# Copyright 2025 Alibaba Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ajet extension for verl DataParallelPPOActor.
Overrides `update_policy` to support `override_ppo_mini_batch_num` and add debug logging.
"""

import logging
import math
import os

import torch
import torch.distributed as dist

from verl import DataProto
from ajet.backbone.verl.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
# [AJET-OPD] On-Policy Distillation loss (ported from verl >=0.8.0 trainer/distillation/losses.py)
from ajet.backbone.verl.distillation import (
    OpdLossConfig,
    compute_distillation_loss,
    compute_forward_kl_topk,
    opd_loss_config_from_meta,
)
# [AJET-OPD/SimCT] cross-tokenizer path
from ajet.backbone.verl.simct import student_virtual_logits, simct_reverse_kl
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
# ajet/backbone/verl/seqlen_balancing.py
from ajet.backbone.verl.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ajet.tuner_lib.experimental.interchange_utils import http_push_verbose_log
from verl.workers.actor.dp_actor import DataParallelPPOActor

__all__ = ["AjetDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AjetDataParallelPPOActor(DataParallelPPOActor):
    """DataParallelPPOActor with ajet-specific modifications:

    1. Supports `override_ppo_mini_batch_num` to control the number of optimizer steps per train-batch-step.
    2. Adds debug print for tensor shapes during training.
    3. Override `prepare_dynamic_batch`
    """

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        # print(f"len(micro_batches) = {len(micro_batches)}")
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs



    def _forward_micro_batch_with_logits(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """[AJET-OPD] Dense (non-remove-padding) forward that ALSO returns the response-token logits,
        for the ``forward_kl_topk`` distillation loss (logits-processor path).

        Returns:
            log_probs: (bs, response_len)
            logits:    (bs, response_len, vocab_size) — on the grad graph, used to differentiate
                       the top-k forward KL against the student distribution.
            entropys:  (bs, response_len) only if ``calculate_entropy``.

        NOTE: intentionally bypasses remove_padding / ulysses SP / fused kernels so the dense
        ``[bs, response_len, V]`` logits exist. forward_kl_topk at long context / large vocab is
        memory-heavy — prefer an estimator mode (kl/k1/k3/...) unless you need the distributional
        signal.
        """
        if self.use_remove_padding or self.use_ulysses_sp or self.use_fused_kernels:
            logger.warning(
                "[AJET-OPD] forward_kl_topk uses a dense forward (remove_padding/ulysses/fused ignored)."
            )
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bs, response_len, vocab_size)
            log_probs = logprobs_from_logits(logits, micro_batch["responses"])
            outputs = {"log_probs": log_probs, "logits": logits}
            if calculate_entropy:
                outputs["entropys"] = entropy_from_logits(logits)
            return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # [AJET] Optional: estimate the GPU-memory limit of ppo_max_token_len_per_gpu and raise.
        # Triggered by env AGENTJET_FIND_MAX_PPO_TOKEN_LEN. Intercepts the *first* real PPO update
        # (model + grads + optimizer already resident, so the measurement is realistic). See
        # ajet.utils.find_max_ppo_token_len. It never returns.
        if os.environ.get("AGENTJET_FIND_MAX_PPO_TOKEN_LEN"):
            from ajet.utils.find_max_ppo_token_len import find_max_ppo_token_len_per_gpu
            find_max_ppo_token_len_per_gpu(self, data)

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        # [AJET-OPD] resolve OPD config from meta_info (populated by the trainer driver; the Ray
        # worker actor only receives the actor_rollout_ref subtree, not the ajet namespace).
        self._opd_cfg: OpdLossConfig = opd_loss_config_from_meta(data.meta_info.get("opd"), self.config)
        self._opd_simct = self._opd_cfg.enabled and self._opd_cfg.loss_mode == "simct"  # [AJET-OPD/SimCT]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self._opd_cfg.enabled and not self._opd_simct and "teacher_logprobs" in data.batch.keys():
            select_keys.append("teacher_logprobs")
            if self._opd_cfg.use_topk and "teacher_ids" in data.batch.keys():
                select_keys.append("teacher_ids")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # [AJET] per-sample loss weight (episode-level loss normalization).
        # Present only when ajet.trainer_common.loss_weight_normalization_episode_level
        # is enabled; absent => every sample weighted equally (default behaviour).
        if "loss_weight" in data.batch.keys():
            select_keys.append("loss_weight")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if self._opd_simct and "simct_specs" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("simct_specs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # [AJET] Support override_ppo_mini_batch_num to control the number of optimizer steps
        if self.config.override_ppo_mini_batch_num > 0:
            mini_batch_split_size = math.ceil(data.batch.batch_size[0] / self.config.override_ppo_mini_batch_num)
        else:
            mini_batch_split_size = self.config.ppo_mini_batch_size

        mini_batches = data.split(mini_batch_split_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # [AJET-OPD/SimCT] SimCT uses dynamic batching like every other mode.
                # prepare_dynamic_batch reindexes non_tensor_batch (incl. simct_specs) by the SAME
                # permutation it applies to the samples, so per-sample spec<->logits alignment is
                # preserved -- no need to force split(1). (Forcing split(1) previously bypassed the
                # token-balanced micro-batch count that keeps FSDP forward allgathers rank-uniform,
                # which caused the NCCL _ALLGATHER_BASE timeout / desync.)
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                num_micro_batches = len(micro_batches)
                for micro_batch_idx, micro_batch in enumerate(micro_batches, 1):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    # [AJET] Episode-level loss-weight normalization.
                    loss_weight = model_inputs.get("loss_weight", None)
                    if loss_weight is not None:
                        loss_weight = loss_weight.to(advantages.dtype)
                        advantages = advantages * loss_weight
                    # [AJET] Debug logging for tensor shapes
                    input_ids = model_inputs["input_ids"]
                    _shape_msg = f'[Update Policy] -> Micro batch shape, input_ids {input_ids.shape}, response {response_mask.shape} @{micro_batch_idx}/{num_micro_batches}'
                    print(_shape_msg)
                    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                        http_push_verbose_log(_shape_msg, tag="update_policy")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.override_ppo_mini_batch_num > 0:
                        loss_scale_factor = response_mask.shape[0] / mini_batch_split_size
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    loss_scale_factor *= self.config.loss_extra_scale_ratio  # [AJET] Extra scaling for loss if needed

                    # all return: (bsz, response_length)
                    # [AJET-OPD] forward_kl_topk needs the student logits alive on the grad graph
                    # (to differentiate the top-k forward KL), so use the dense with-logits forward.
                    opd_need_logits = (
                        self._opd_cfg.enabled
                        and (self._opd_cfg.use_topk or self._opd_simct)
                        and (
                            (self._opd_simct and "simct_specs" in micro_batch.non_tensor_batch)
                            or ("teacher_logprobs" in model_inputs and "teacher_ids" in model_inputs)
                        )
                    )
                    if opd_need_logits:
                        outputs = self._forward_micro_batch_with_logits(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        logits_for_opd = outputs["logits"]
                    else:
                        outputs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        logits_for_opd = None
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        # [AJET] apply the per-token episode-level loss weight to
                        # the KL term as well (same weight/shape used for the
                        # policy-gradient term above), so each episode contributes
                        # equally to the KL loss too.
                        if loss_weight is not None:
                            kld = kld * loss_weight
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # [AJET-OPD] On-Policy Distillation loss (teacher signal), layered on top of the
                    # task policy loss. Mirrors verl >=0.8.0 distillation_ppo_loss: the divergence is
                    # applied under the configured gradient mode (supervised agg_loss, or PG with
                    # advantage = -KL), then combined with the task loss via distillation_loss_coef.
                    if self._opd_cfg.enabled and "teacher_logprobs" in model_inputs:
                        opd_data = {
                            "teacher_logprobs": model_inputs["teacher_logprobs"].to(log_prob.dtype),
                            "response_mask": response_mask,
                            "old_log_probs": old_log_prob,
                        }
                        if rollout_is_weights is not None:
                            opd_data["rollout_is_weights"] = rollout_is_weights
                        opd_model_output = {"log_probs": log_prob}
                        if opd_need_logits and logits_for_opd is not None:
                            topk_out = compute_forward_kl_topk(
                                student_logits=logits_for_opd,
                                teacher_topk_log_probs=model_inputs["teacher_logprobs"].to(logits_for_opd.dtype),
                                teacher_topk_ids=model_inputs["teacher_ids"].long(),
                                cfg=self._opd_cfg,
                                response_mask=response_mask,
                            )
                            opd_model_output.update(topk_out)
                        distill_loss, opd_metrics = compute_distillation_loss(
                            self._opd_cfg, opd_model_output, opd_data
                        )
                        if self._opd_cfg.use_task_rewards:
                            policy_loss = policy_loss + distill_loss * self._opd_cfg.distillation_loss_coef
                        else:
                            policy_loss = distill_loss * self._opd_cfg.distillation_loss_coef
                        micro_batch_metrics.update(opd_metrics)
                        metrics.setdefault("actor/distill_loss", 0.0)
                        _dl_val = distill_loss.detach().item() if torch.is_tensor(distill_loss) else float(distill_loss)
                        metrics["actor/distill_loss"] += _dl_val * loss_scale_factor

                    # [AJET-OPD/SimCT] cross-tokenizer distillation: student virtual logits from
                    # white-box student logits (response-aligned → spec's response-relative positions
                    # map directly), reverse-KL vs teacher virtual logits (precomputed driver-side from
                    # the remote-vLLM prompt_logprobs). Per-sample specs travel via non_tensor_batch.
                    if self._opd_simct and "simct_specs" in micro_batch.non_tensor_batch and logits_for_opd is not None:
                        specs = micro_batch.non_tensor_batch["simct_specs"]
                        mb = logits_for_opd.shape[0]
                        distill_loss = logits_for_opd.new_zeros(())
                        n_seg_total = 0
                        for s in range(mb):
                            spec = specs[s]
                            if spec is None:
                                continue
                            sv = student_virtual_logits(spec, logits_for_opd[s])  # [num_seg, virtual_dim]
                            tv = spec.teacher_virtual.to(device=sv.device, dtype=sv.dtype).detach()
                            distill_loss = distill_loss + simct_reverse_kl(sv, tv)  # scalar (sum over segs)
                            n_seg_total += spec.num_segments
                        distill_loss = distill_loss / max(n_seg_total, 1)  # per-segment mean
                        if self._opd_cfg.use_task_rewards:
                            policy_loss = policy_loss + distill_loss * self._opd_cfg.distillation_loss_coef
                        else:
                            policy_loss = distill_loss * self._opd_cfg.distillation_loss_coef
                        micro_batch_metrics["opd/distill_loss"] = distill_loss.detach().item()
                        micro_batch_metrics["opd/segments_per_sample"] = float(n_seg_total) / max(mb, 1)
                        metrics.setdefault("actor/distill_loss", 0.0)
                        metrics["actor/distill_loss"] += distill_loss.detach().item() * loss_scale_factor

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)
                print(f'-> optimizer_step !')
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
