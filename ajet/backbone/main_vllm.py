import atexit
import os
import sys
from types import SimpleNamespace

import hydra
from openai import AsyncOpenAI, OpenAI

from ajet.backbone.warm_up import warm_up_process
from ajet.task_rollout.native_parallel_worker import VerlRolloutManager
from ajet.utils.launch_utils import set_loguru_default_color
from ajet.schema.logprob import TokenAndProb
from ajet.utils.core_env_vars import get_runtime_env
from loguru import logger

set_loguru_default_color()


class TokenAndProbVllmDebug(TokenAndProb):
    def __init__(self, t):
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        token_id = int(t.token.split("token_id:")[-1])
        logprob = t.logprob
        try:
            decoded_string = bytes(t.bytes).decode("utf-8")
        except Exception:
            decoded_string = "<cannot decode>" + str(t.bytes)
        super().__init__(token_id=token_id, logprob=logprob, decoded_string=decoded_string)


class ChatCompletionScheduler:
    def __init__(self, url, config):
        from transformers import AutoTokenizer

        self.url = url
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ajet.model.path)
        self.chat_scheduler = SimpleNamespace(
            model_name="dummy-model-name",
            weighted_addresses="dummy-weighted-addresses",
            completion_callback=SimpleNamespace(tokenizer=self.tokenizer),
        )

    def submit_chat_completions(self, messages, sampling_params, request_id, tools=[]):
        client = OpenAI(
            base_url=self.url,
            api_key="token-abc123",
        )
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.config.ajet.rollout.max_response_length_in_one_turn,
        )
        sampling_params["temperature"] = self.config.ajet.rollout.val_kwargs.temperature
        sampling_params["top_k"] = self.config.ajet.rollout.val_kwargs.top_k
        sampling_params["top_p"] = self.config.ajet.rollout.val_kwargs.top_p

        sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})

        if tools:
            completion = client.chat.completions.create(
                model=self.config.ajet.model.path,
                messages=messages,
                tools=tools,
                extra_body=sampling_params,
            )
        else:
            completion = client.chat.completions.create(
                model=self.config.ajet.model.path,
                messages=messages,
                extra_body=sampling_params,
            )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

        # sometimes tool use message has no content field
        if "content" not in message:
            message["content"] = ""

        messages.append(
            {
                "role": message["role"],
                "request_id": completion.id,
                "content": message["content"],
                "tool_calls": message.get("tool_calls", None),
                "tokens": [TokenAndProbVllmDebug(t) for t in completion.choices[0].logprobs.content],  # type: ignore
            }
        )
        return messages

    async def submit_chat_completions_async(self, messages, sampling_params, request_id, tools=[]):
        client = AsyncOpenAI(
            base_url=self.url,
            api_key="token-abc123",
        )
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.config.ajet.rollout.max_response_length_in_one_turn,
        )
        sampling_params["temperature"] = self.config.ajet.rollout.val_kwargs.temperature
        sampling_params["top_k"] = self.config.ajet.rollout.val_kwargs.top_k
        sampling_params["top_p"] = self.config.ajet.rollout.val_kwargs.top_p

        sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})

        if tools:
            completion = await client.chat.completions.create(
                model=self.config.ajet.model.path,
                messages=messages,
                tools=tools,
                extra_body=sampling_params,
            )
        else:
            completion = await client.chat.completions.create(
                model=self.config.ajet.model.path,
                messages=messages,
                extra_body=sampling_params,
            )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

        # sometimes tool use message has no content field
        if "content" not in message:
            message["content"] = ""

        messages.append(
            {
                "role": message["role"],
                "request_id": completion.id,
                "content": message["content"],
                "tool_calls": message.get("tool_calls", None),
                "tokens": [TokenAndProbVllmDebug(t) for t in completion.choices[0].logprobs.content],  # type: ignore
            }
        )
        return messages


def run(config):
    from ajet.task_reader import RouterTaskReader

    # --------- fast adjustment for debugging ---------
    warm_up_process(config)
    max_parallel = config.ajet.debug.debug_max_parallel
    n_task = config.ajet.debug.debug_first_n_tasks
    vllm_port = config.ajet.debug.debug_vllm_port

    # --------- init ---------
    async_rollout_manager = ChatCompletionScheduler(config=config, url=f"http://localhost:{vllm_port}/v1")
    parallel_env = VerlRolloutManager(
        config=config,
        async_rollout_manager=async_rollout_manager,
        max_parallel=max_parallel,
        max_llm_retries=3,
        llm_mode="remote",
        tokenizer=async_rollout_manager.tokenizer,
    )

    task_reader = RouterTaskReader(
        config.ajet.task_reader.type,
        config.ajet.task_reader,
    )
    tasks = task_reader.get_validation_tasks()
    logger.info(tasks[:n_task])
    ctx_tracker = parallel_env.rollout(tasks=tasks[:n_task], mode="sample", epoch="1")  # "sample" or "validate"
    _ = parallel_env.to_dataproto(ctx_tracker)


@hydra.main(
    config_path="ajet/default_config",
    config_name="ajet_default",
    version_base=None,
)
def main(config):
    from omegaconf import OmegaConf

    OmegaConf.resolve(config)
    runtime_env = get_runtime_env(config)
    os.environ.update(runtime_env["env_vars"])
    # atexit.register(lambda: print("Process exiting, performing cleanup..."))

    if config.ajet.enable_experimental_interchange_server:
        from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import start_interchange_server

        start_interchange_server(config)

    def companion_launch():
        import torch

        from ajet.utils.smart_daemon import LaunchCommandWhenAbsent

        logger.info("Launching companion process for async LLM server...")
        model_path = config.ajet.model.path
        tensor_parallel_size = config.ajet.debug.debug_tensor_parallel_size
        n_avail_gpus = torch.cuda.device_count()
        if tensor_parallel_size > n_avail_gpus:
            logger.info(f"Warning: tensor_parallel_size {tensor_parallel_size} is greater than available GPUs {n_avail_gpus}. Setting tensor_parallel_size to {n_avail_gpus}.")
            tensor_parallel_size = n_avail_gpus
        gpu_memory_utilization = config.actor_rollout_ref.rollout.gpu_memory_utilization
        max_num_seqs = config.actor_rollout_ref.rollout.max_num_seqs
        max_model_len = config.ajet.rollout.max_model_len
        seed = config.ajet.debug.debug_vllm_seed
        vllm_port = config.ajet.debug.debug_vllm_port
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                f"{model_path}",
                "--tensor-parallel-size",
                f"{tensor_parallel_size}",
                "--dtype",
                "auto",
                "--enforce-eager",
                "--gpu-memory-utilization",
                f"{gpu_memory_utilization}",
                "--disable-custom-all-reduce",
                "--max-num-seqs",
                f"{max_num_seqs}",
                "--max-model-len",
                f"{max_model_len}",
                "--load-format",
                "auto",
                "--enable-chunked-prefill",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--enable-prefix-caching",
                "--seed",
                f"{seed}",
                "--port",
                f"{vllm_port}",
            ],
            dir="./",
            tag="external_vllm_server",
        )
        companion.launch(
            launch_wait_time=1800,
            success_std_string="Application startup complete",
            env_dict={**os.environ},
        )

    companion_launch()

    run(config)


if __name__ == "__main__":
    main()
