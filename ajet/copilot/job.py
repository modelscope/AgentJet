"""Programmatic training entry point for AgentJet.

This class mirrors the CLI launcher by materializing a YAML config and
spawning a subprocess to run the existing training pipeline. The goal is to
keep the public surface minimal while reusing the mature CLI code paths.
"""

from __future__ import annotations

import os
import time
import yaml
import tempfile

from types import SimpleNamespace
from typing import Any, Callable, Union, cast
from loguru import logger
from ajet.default_config.ajet_default import Config
from ajet.utils.config_utils import (
    expand_ajet_hierarchical_config,
    prepare_experiment_config,
    read_ajet_hierarchical_config,
)
from ajet.utils.dynamic_import import cls_to_path
from ajet.utils.launch_utils import (
    execute_training_process,
    check_avail_gpu,
    get_backbone_target,
    setup_environment_vars,
)


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
    """
    arg: base_yaml_config + **kwargs (yaml config, then override with kwargs)
    arg: base_yaml_config (yaml config)
    arg: **kwargs (yaml config, then override with kwargs)
    """

    def __init__(
        self,
        base_yaml_config: str | None = None,
        experiment_dir: str | None = None,
        project_name: str | None = None,
        experiment_name: str | None = None,
        n_gpu: int | None = None,
        model: str | None = None,
        algorithm: str | None = None,
        num_repeat: int | None = None,
        batch_size: int | None = None,
        swarm_mode: bool | None = None,
        swarm_mode_sample_collection_method: str | None = None,
        max_env_worker: int | None = None,
        backbone: str | None = None,
    ) -> None:

        if base_yaml_config is None:
            base_yaml_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "default_config/ajet_ts_default.yaml"))
        else:
            logger.warning(f"Reading config from {base_yaml_config}.")
            time.sleep(1)
        self.config_as_dict: dict = self.build_job_from_yaml(base_yaml_config)
        self.config = Config.update_from_dict_recursive(Config(), self.config_as_dict)

        self.base_yaml_config: str = cast(str, base_yaml_config)    # currently may be None, but will be set later
        self.experiment_dir: str = cast(str, experiment_dir)
        self.project_name: str = cast(str, project_name)
        self.experiment_name: str = cast(str, experiment_name)
        self.n_gpu: int = cast(int, n_gpu)
        self.model: str = cast(str, model)
        self.algorithm: str = cast(str, algorithm)
        self.num_repeat: int = cast(int, num_repeat)
        self.batch_size: int = cast(int, batch_size)
        self.swarm_mode: bool = cast(bool, swarm_mode)
        self.swarm_mode_sample_collection_method: str = cast(str, swarm_mode_sample_collection_method)
        self.max_env_worker: int = cast(int, max_env_worker)
        self.backbone: str = cast(str, backbone)

        # see `ajet/default_config/ajet_ts_default.yaml`
        overrides = {
            "ajet.experiment_dir":                          "experiment_dir",
            "ajet.project_name":                            "project_name",
            "ajet.experiment_name":                         "experiment_name",
            "ajet.model.path":                              "model",
            "ajet.trainer_common.n_gpus_per_node":          "n_gpu",
            "ajet.trainer_common.algorithm.adv_estimator":  "algorithm",
            "ajet.rollout.num_repeat":                      "num_repeat",
            "ajet.data.train_batch_size":                   "batch_size",
            "ajet.enable_swarm_mode":                       "swarm_mode",
            "ajet.swarm_mode_sample_collection_method":     "swarm_mode_sample_collection_method",
            "ajet.rollout.max_env_worker":                  "max_env_worker",
            "ajet.backbone":                                "backbone",
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

        if self.backbone == "trinity":
            raise NotImplementedError("Trinity backbone is not yet supported in AgentJetJob.")


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

