"""Programmatic training entry point for AgentJet.

This class mirrors the CLI launcher by materializing a YAML config and
spawning a subprocess to run the existing training pipeline. The goal is to
keep the public surface minimal while reusing the mature CLI code paths.
"""

from __future__ import annotations

import os
import time
import yaml

from typing import Any, Callable, List, Union, cast
from loguru import logger
from ajet.default_config.ajet_config_schema import Config
from ajet.utils.config_utils import (
    expand_ajet_hierarchical_config,
    read_ajet_hierarchical_config,
)
from ajet.utils.dynamic_import import cls_to_path
from beast_logger import print_dict


def override_current_yaml_value_if_given(override_value, current_value):
    if override_value is not None:
        return override_value
    else:
        return current_value

def _set_nested_attr(obj, attr_path: str, value):
    keys = attr_path.split(".")
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)

def _get_nested_attr(obj, attr_path: str):
    for key in attr_path.split("."):
        obj = getattr(obj, key)
    return obj

class AgentJetJob:
    """Programmatic interface for configuring ( Arguments + YAML -->  New YAML ) and launching AgentJet training jobs.

    Args:
        base_yaml_config: Path to base YAML configuration file. If None, uses default config (at ./ajet/default_config/ajet_swarm_default.yaml).
        experiment_dir: Directory where experiment outputs will be saved.
        project_name: Name of the project for organizing experiments.
        experiment_name: Unique name for this specific experiment run.
        logging: "swanlab", "tensorboard", etc
        n_gpu: Number of GPUs to use per node for training.
        model: Path or identifier of the model to train.
        algorithm: Advantage estimator algorithm (e.g., 'gae', 'vtrace').
        num_repeat: Tell swarm server how many repeated sample it should expect for a same task (same means task_id is identical).
        batch_size: Training batch size for the model (the watermark to empty buffer pool and update llm weight).
        swarm_mode: Whether to enable swarm mode for distributed sample collection.
        swarm_mode_sample_collection_method: Stop-condition the swarm server uses to decide when the
            current batch of collected samples is "enough" and the next weight update can proceed.
            One of:
              - "rollout_until_finish_enough_episodes": stop once ``total_episodes >= batch_size * num_repeat``.
                Each episode counts individually regardless of its ``task_id``; cheapest, but a GRPO
                group may end up with fewer than ``num_repeat`` episodes.
              - "rollout_until_finish_enough_tasks" (default): stop once ``batch_size`` distinct
                ``task_id``s have each accumulated at least ``num_repeat`` completed episodes. Guarantees
                every GRPO group is fully populated.
              - "rollout_until_finish_enough_non_dummy_tasks": like the above, but only counts tasks
                whose ``num_repeat`` episodes do *not* all share the same reward. Tasks with uniform
                reward (e.g. all 0 or all 1) produce zero advantage under GRPO and are skipped —
                useful when the dataset contains many too-easy or too-hard prompts.
        max_env_worker: an estimation about how many episodes will be running in parallel (all swarm clients combined).
        backbone: Training backbone framework (e.g., 'verl').
        max_prompt_length: Maximum token length for input prompts (token length before the first llm-generated token, default 3000).
        max_response_length: Maximum token length for model responses (token length after the first llm-generated token, default 15000).
        max_response_length_in_one_turn: Maximum token length for model response in one turn (default 4096, should be <= max_response_length).
        max_model_len: Maximum total token length (prompt + response) the model can handle (bigger => more GPU memory), default 18000.
        mini_batch_num: Number of mini-batches to split training batch into (how many mini steps, i.e. how many times the `optimizer.step` should be executed, per big train batch).
        lora_rank: LoRA rank for low-rank adaptation (set > 0 to enable LoRA training, default 0 means disabled).
        lora_alpha: LoRA alpha scaling factor (default 16).
        lora_target_modules: Target modules for LoRA adaptation (default 'all-linear').
        lora_load_format: Load format for LoRA weights (default 'auto').
        layered_summon: Enable layered summon for LoRA (default False).
        gpu_memory_utilization: GPU memory utilization for vLLM engine (default 0.85).
        lr: Learning rate for optimizer (default 1e-6).
        ppo_epochs: Number of PPO epochs per update (default 1).
        compute_madness_checklist: List of madness checks to monitor LLM's abnormal behaviors during rollout (default ["nonsense"], detect infinite repeat such as "但但但但但但但但但但....").
    """

    def __init__(
        self,
        ensure_new_experiment: bool = False,
        base_yaml_config: str | None = None,
        experiment_dir: str | None = None,
        project_name: str | None = None,
        experiment_name: str | None = None,
        logging: str | None = None,
        n_gpu: int | None = None,
        model: str | None = None,
        algorithm: str | None = None,
        num_repeat: int | None = None,
        batch_size: int | None = None,
        swarm_mode: bool | None = None,
        swarm_mode_sample_collection_method: str | None = None,
        max_env_worker: int | None = None,
        backbone: str | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        max_response_length_in_one_turn: int | None = None,
        max_model_len: int | None = None,
        mini_batch_num: int | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_target_modules: str | None = None,
        lora_load_format: str | None = None,
        layered_summon: bool | None = None,
        gpu_memory_utilization: float | None = None,
        lr: float | None = None,
        ppo_epochs: int | None = None,
        compute_madness_checklist: List[str] | None = None,
    ) -> None:

        if base_yaml_config is None:
            base_yaml_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "default_config/ajet_swarm_default.yaml"))
        else:
            logger.warning(f"Reading config from {base_yaml_config}.")
            time.sleep(1)
        if not os.path.exists(base_yaml_config):
            raise ValueError(f"Configuration yaml is absent! {base_yaml_config}")

        # Validate: max_prompt_length, max_response_length, max_model_len must all be None or all be non-None
        length_params = [max_prompt_length, max_response_length, max_model_len, max_response_length_in_one_turn]
        if not (all(p is None for p in length_params) or all(p is not None for p in length_params)):
            raise ValueError("(`max_prompt_length`, `max_response_length`, `max_model_len`, `max_response_length_in_one_turn`) must all be None or all be non-None")

        self.config_as_dict: dict = self.build_job_from_yaml(base_yaml_config)
        self.config = Config.update_from_dict_recursive(Config(), self.config_as_dict)
        self.ensure_new_experiment = ensure_new_experiment

        self.base_yaml_config: str = cast(str, base_yaml_config)    # currently may be None, but will be set later
        self.experiment_dir: str = cast(str, experiment_dir)
        self.project_name: str = cast(str, project_name)
        self.experiment_name: str = cast(str, experiment_name)

        if self.ensure_new_experiment:
            # add timestamp suffix to experiment_name to ensure it's new every time, if ensure_new_experiment is True.
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if self.experiment_name is not None:
                self.experiment_name += f"_{timestamp}"
            else:
                self.experiment_name = self.config.ajet.experiment_name
                if self.experiment_name != "read_yaml_name":
                    self.experiment_name += f"_{timestamp}"

        self.logging: str = cast(str, logging)
        self.n_gpu: int = cast(int, n_gpu)
        self.model: str = cast(str, model)
        self.algorithm: str = cast(str, algorithm)
        self.num_repeat: int = cast(int, num_repeat)
        self.batch_size: int = cast(int, batch_size)
        self.swarm_mode: bool = cast(bool, swarm_mode)
        self.swarm_mode_sample_collection_method: str = cast(str, swarm_mode_sample_collection_method)
        self.max_env_worker: int = cast(int, max_env_worker)
        self.backbone: str = cast(str, backbone)
        self.max_prompt_length: int = cast(int, max_prompt_length)
        self.max_response_length_in_one_turn: int = cast(int, max_response_length_in_one_turn)
        self.max_response_length: int = cast(int, max_response_length)
        self.max_model_len: int = cast(int, max_model_len)
        self.mini_batch_num: int = cast(int, mini_batch_num)
        self.lora_rank: int = cast(int, lora_rank)
        self.lora_alpha: int = cast(int, lora_alpha)
        self.lora_target_modules: str = cast(str, lora_target_modules)
        self.lora_load_format: str = cast(str, lora_load_format)
        self.layered_summon: bool = cast(bool, layered_summon)
        self.gpu_memory_utilization: float = cast(float, gpu_memory_utilization)
        self.lr: float = cast(float, lr)
        self.ppo_epochs: int = cast(int, ppo_epochs)
        self.compute_madness_checklist: List[str] = cast(List[str], compute_madness_checklist)

        # see `ajet/default_config/ajet_swarm_default.yaml`
        overrides = {
            # left: [yaml key navigation]                  right: [AgentJetJob self attr]
            "ajet.experiment_dir":                          "experiment_dir",
            "ajet.project_name":                            "project_name",
            "ajet.experiment_name":                         "experiment_name",
            "ajet.trainer_common.logger":                   "logging",
            "ajet.model.path":                              "model",
            "ajet.trainer_common.n_gpus_per_node":          "n_gpu",
            "ajet.trainer_common.algorithm.adv_estimator":  "algorithm",
            "ajet.rollout.num_repeat":                      "num_repeat",
            "ajet.data.train_batch_size":                   "batch_size",
            "ajet.enable_swarm_mode":                       "swarm_mode",
            "ajet.swarm_mode_sample_collection_method":     "swarm_mode_sample_collection_method",
            "ajet.rollout.max_env_worker":                  "max_env_worker",
            "ajet.backbone":                                "backbone",
            "ajet.data.max_prompt_length":                  "max_prompt_length",
            "ajet.data.max_response_length":                "max_response_length",
            "ajet.rollout.max_response_length_in_one_turn": "max_response_length_in_one_turn",
            "ajet.rollout.max_model_len":                   "max_model_len",
            "ajet.trainer_common.mini_batch_num":           "mini_batch_num",
            "ajet.lora.lora_rank":                          "lora_rank",
            "ajet.lora.lora_alpha":                         "lora_alpha",
            "ajet.lora.target_modules":                     "lora_target_modules",
            "ajet.lora.load_format":                        "lora_load_format",
            "ajet.lora.layered_summon":                     "layered_summon",
            "ajet.rollout.gpu_memory_utilization":          "gpu_memory_utilization",
            "ajet.trainer_common.optim.lr":                 "lr",
            "ajet.trainer_common.ppo_epochs":               "ppo_epochs",
            "ajet.rollout.compute_madness_checklist":       "compute_madness_checklist",
        }

        # if any value given in kwargs, override the corresponding value in config
        for attr_path, override_val in overrides.items():
            # get value from yaml config
            # >> e.g. current_model = self.config.model.path
            current_val = _get_nested_attr(self.config, attr_path)

            # if override_val (given in __init__) is not None, use it to override the value from yaml config
            # >> e.g. new_model = self.model if (self.model is not None) else current_model
            new_val = override_current_yaml_value_if_given(getattr(self, override_val), current_val)

            # write final value to `self.config``
            # >> e.g. self.config.model.path = new_model
            _set_nested_attr(self.config, attr_path, new_val)

            # write final value to `self`
            # >> e.g. self.model = new_model
            setattr(self, override_val, new_val)


        assert self.max_prompt_length + self.max_response_length <= self.max_model_len, "illegal token length"
        assert self.max_response_length_in_one_turn <= self.max_response_length

        # Validate: when lora_rank > 0, load_format must be safetensors
        if self.lora_rank > 0:
            if self.lora_load_format != "safetensors":
                raise ValueError(f"When lora_rank > 0, lora_load_format must be 'safetensors', got '{self.lora_load_format}'")
            if not self.layered_summon:
                raise ValueError("When lora_rank > 0, layered_summon must be True")
            if self.lr is None:
                raise ValueError("lr should be provided for lora training")
            if self.lr <= 1e-5:
                raise ValueError(f"lr should usually be greater than 1e-5 for lora training, got {self.lr}")

        if self.backbone == "trinity":
            raise NotImplementedError("Trinity backbone is not yet supported in AgentJetJob.")

        primary_attributes = {key: getattr(self, key) for key in overrides.values()}

        print_dict(primary_attributes)



    def build_job_from_yaml(self, yaml_path: str | None) -> dict:
        self.config_as_dict = read_ajet_hierarchical_config(
            yaml_path,
            write_to=None,
        )
        self.config_as_dict = expand_ajet_hierarchical_config(self.config_as_dict, write_to=None)
        logger.info(f"Built AgentJet job config: {yaml_path}")
        return self.config_as_dict


    def dump_job_as_yaml(self, yaml_path: str) -> str:
        if os.path.dirname(yaml_path):
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f, sort_keys=False)
        logger.info(f"Saved training config to {yaml_path}")
        return yaml_path


    def set_workflow(
        self, workflow: Union[str, Callable[..., Any]], ensure_reward_in_workflow: bool = False
    ) -> "AgentJetJob":
        self.config.ajet.rollout.user_workflow = cls_to_path(workflow)
        # TODO: validate workflow outputs contain reward
        # ensure_reward_in_workflow
        return self


    def set_data(
        self,
        type: str,
        dataset_path: str,
        training_split: str = "train",
        validation_split: str = "test",
    ) -> "AgentJetJob":
        """Configure the task reader. Defaults to HuggingFace datasets."""

        # available types:
        # `env_service` or `jsonl_dataset_file` or `huggingface_dat_repo` or `data_generation` or `random_dummy`

        if type in {"hf", "huggingface", "huggingface_dat_repo"}:
            self.config.ajet.task_reader.type = "huggingface_dat_repo"
            self.config.ajet.task_reader.huggingface_dat_repo.dataset_path = dataset_path
            self.config.ajet.task_reader.huggingface_dat_repo.training_split = training_split
            self.config.ajet.task_reader.huggingface_dat_repo.validation_split = validation_split
        elif type in {"random_dummy", "dummy"}:
            self.config.ajet.task_reader.type = "random_dummy"
        else:
            raise NotImplementedError(
                f"Please edit yaml to directly set up task reader of type {type}."
            )

        return self
