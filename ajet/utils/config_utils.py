# -*- coding: utf-8 -*-

import os
import shutil
import time
import yaml
import hydra.errors
from functools import cache

from beast_logger import print_dict
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig

from ajet.utils.config_computer import split_keys_and_operators

DEFAULT_DIR = "saved_experiments"

def fix_hydra_searchpath_and_create_copy_when_needed(yaml_fp):
    """Fix Hydra search paths if they don't exist by trying with base directory."""
    abs_yaml_fp = os.path.abspath(yaml_fp)
    with open(abs_yaml_fp, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    if yaml_content and 'hydra' in yaml_content and 'searchpath' in yaml_content['hydra']:
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        modified = False
        for i, path in enumerate(yaml_content['hydra']['searchpath']):
            if path.startswith('file://'):
                rel_path = path[7:]
                if not os.path.exists(rel_path):
                    fixed_path = os.path.join(base_dir, rel_path)
                    if os.path.exists(fixed_path):
                        logger.warning(f"Cannot find `{os.path.abspath(rel_path)}`, but find `{os.path.abspath(fixed_path)}`, override original config ...")
                        yaml_content['hydra']['searchpath'][i] = f'file://{fixed_path}'
                        modified = True
        if modified:
            with open(abs_yaml_fp + ".patch.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f)
            return abs_yaml_fp + ".patch.yaml"
    return abs_yaml_fp


def read_ajet_config(yaml_fp):
    """Load a Hydra configuration relative to this module."""
    yaml_fp = read_ajet_yaml_fp = fix_hydra_searchpath_and_create_copy_when_needed(yaml_fp)
    yaml_fp = os.path.relpath(
        yaml_fp, os.path.dirname(__file__)
    )  # do not try to understand this line, hydra is too weird

    def load_hydra_config(config_path: str, config_name: str) -> DictConfig:
        with initialize(config_path=config_path, version_base=None):
            try:
                cfg = compose(config_name=config_name, overrides=[])
            except hydra.errors.MissingConfigException as e:
                logger.error(f"Configuration default files not found (please check {read_ajet_yaml_fp})")
                raise e
            return cfg

    dir_path = os.path.dirname(yaml_fp)
    file_name = os.path.basename(yaml_fp)
    return load_hydra_config(config_path=dir_path, config_name=file_name)


@cache
def read_ajet_config_with_cache(yaml_fp):
    """Load a Hydra configuration relative to this module with caching."""
    return read_ajet_config(yaml_fp)


def dump_yaml_config(cfg: DictConfig, yaml_fp: str):
    """Persist the provided OmegaConf config to ``yaml_fp``."""
    from omegaconf import OmegaConf

    with open(yaml_fp, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    return yaml_fp


class NotFound(object): pass

def _dive_to_fetch_value(config, dotted_key):
    keys = dotted_key.split(".")
    value = config
    for key in keys:
        value = value.get(key, NotFound)
        if value is None:
            break
        if value is NotFound:
            break
    if value is NotFound:
        raise ValueError(f"[Warning]: Cannot find value for key: {dotted_key} in {config}")
    return value


def _dive_to_set_value(config, dotted_key, value):
    keys = dotted_key.split(".")
    sub_config = config
    for key in keys[:-1]:
        if key not in sub_config:
            sub_config[key] = {}
        sub_config = sub_config[key]
    sub_config[keys[-1]] = value


def align_parameters(from_config_fp, to_config_fp, convertion_json_fg, backbone):
    """Align configuration values based on a conversion map.

    Parameters
    ----------
    from_config_fp : str
        Source YAML path to read values from.
    to_config_fp : str
        Destination YAML path that is updated in place.
    convertion_json_fg : str
        JSON path mapping dotted keys between configs.
    backbone : str
        Backbone identifier used for framework-specific alignment.
    """
    # read yaml files
    with open(from_config_fp, "r", encoding="utf-8") as file:
        from_config = yaml.safe_load(file)
    with open(to_config_fp, "r", encoding="utf-8") as file:
        to_config = yaml.safe_load(file)

    # read convertion json
    import json

    with open(convertion_json_fg, "r", encoding="utf-8") as file:
        convertion_json = json.load(file)

    logger.success("----------------------------------------------------")
    # align trinity.* to to_config
    if ("trinity" in from_config) and backbone == "trinity":
        trinity_config = from_config["trinity"]

        def recursive_copy(src_dict, dst_dict, parent_key=""):
            for key, value in src_dict.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    if key not in dst_dict:
                        dst_dict[key] = {}
                    recursive_copy(value, dst_dict[key], full_key)
                else:
                    dst_dict[key] = value

        recursive_copy(trinity_config, to_config)

    # align based on convertion_json
    for from_key, to_keys in convertion_json.items():
        if from_key.startswith("("):
            # special argument that need A.S.T. computation
            # e.g. "(min(ajet.rollout.max_env_worker, 128) // ajet.rollout.n_vllm_engine)": "explorer.runner_per_model"
            keys_array, config_computer = split_keys_and_operators(from_key, [])
            value = config_computer({k: _dive_to_fetch_value(from_config, k) for k in keys_array})
        else:
            # normal argument
            value = _dive_to_fetch_value(from_config, from_key)

        # multiple to_keys support
        to_keys = to_keys if isinstance(to_keys, list) else [to_keys]

        # set and override config value
        for to_key in to_keys:
            _dive_to_set_value(to_config, to_key, value)
            logger.success(
                f"[Note]: Aligned parameter from [{from_key}] to [{to_key}] with value: [{value}]"
            )

    # backbone specific safe guard
    to_config = config_safe_guard(to_config, backbone)

    # save to_config_fp
    with open(to_config_fp, "w", encoding="utf-8") as file:
        yaml.dump(to_config, file)

    # logger.success(f"Saved aligned configuration to {to_config_fp}")
    print_dict({"Note": f"Saved aligned configuration to {to_config_fp}"}, header="Final Configuration")


def config_safe_guard(config: dict, backbone: str) -> dict:
    # special: logger
    if backbone == "verl" and isinstance(config["trainer"]["logger"], str):
        config["trainer"]["logger"] = ["console", config["trainer"]["logger"]]

    # special: trinity train_batch_size
    if backbone == "trinity":
        train_batch_size = config["buffer"]["train_batch_size"]
        world_size = config["cluster"]["gpu_per_node"] * config["cluster"]["node_num"]
        vllm_world_size = (
            config["explorer"]["rollout_model"]["tensor_parallel_size"]
            * config["explorer"]["rollout_model"]["engine_num"]
        )
        fsdp_world_size = world_size - vllm_world_size

        # if train_batch_size % fsdp_world_size != 0, train_batch_size + until divisible
        if fsdp_world_size > 0 and train_batch_size % fsdp_world_size != 0:
            new_train_batch_size = train_batch_size
            while new_train_batch_size % fsdp_world_size != 0:
                new_train_batch_size += 1
            logger.warning(
                f"[Warning]: trinity backbone detected, but train_batch_size {train_batch_size} is not divisible by fsdp_world_size {fsdp_world_size}. Automatically adjust train_batch_size to {new_train_batch_size}."
            )
            config["buffer"]["train_batch_size"] = new_train_batch_size

    return config


def read_ajet_hierarchical_config(
    yaml_fp, experiment_name=None, backbone=None, write_to=None, experiment_dir=None, override_param_callback=None
):
    if yaml_fp is None:
        config = {
            "ajet": {},
            "hydra": {
                "searchpath": [
                    "file://ajet/default_config",
                    "file://ajet/default_config/verl",
                    "file://ajet/default_config/trinity",
                ]
            },
            "defaults": [
                "verl_default",
                "trinity_default",
                "ajet_default",
                "_self_",
            ],
        }
    else:
        with open(yaml_fp, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    if experiment_name is not None:
        config["ajet"]["experiment_name"] = experiment_name
    if (experiment_dir is not None):
        config["ajet"]["experiment_dir"] = experiment_dir
    if backbone is not None:
        config["ajet"]["backbone"] = backbone

    # remove extra config of verl for trinity
    if backbone == "debug":
        if "trinity_default" in config["defaults"]:
            config["defaults"].remove("trinity_default")
            config["hydra"]["searchpath"].remove("file://ajet/default_config/trinity")
    # remove extra config of verl for trinity
    if backbone == "trinity":
        if "verl_default" in config["defaults"]:
            config["defaults"].remove("verl_default")
            config["hydra"]["searchpath"].remove("file://ajet/default_config/verl")
    # remove extra config of trinity for verl
    if backbone == "verl":  # or args.backbone == "debug"
        if "trinity_default" in config["defaults"]:
            config["defaults"].remove("trinity_default")
            config["hydra"]["searchpath"].remove("file://ajet/default_config/trinity")

    if override_param_callback is not None:
        config = override_param_callback(config)

    if write_to:
        with open(write_to, "w", encoding="utf-8") as file:
            yaml.dump(config, file)
    return config


def expand_ajet_hierarchical_config(config, write_to=None):
    # create temp yaml file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_yaml:
        yaml_path = temp_yaml.name
        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(config, file)
        full_config = read_ajet_config(yaml_path)
        yaml_path = dump_yaml_config(full_config, yaml_fp=yaml_path)
        # put inherit info back
        with open(yaml_path, "r", encoding="utf-8") as file:
            config_final = yaml.safe_load(file)
        config_final["defaults"] = config["defaults"]
        config_final["hydra"] = config["hydra"]

    if write_to:
        with open(write_to, "w", encoding="utf-8") as file:
            yaml.dump(config_final, file)

    return config_final


def _validate_input_yaml_no_overlap_with_auto_convertion_config(input_yaml_config, config_final):
    """Validate that input yaml doesn't contain keys in fields such as `actor_rollout_ref` that will be override by `ajet` field values."""
    import json
    import re

    jsonc_path = os.path.join(os.path.dirname(__file__), "..", "default_config", "verl", "config_auto_convertion_verl.jsonc")
    with open(jsonc_path, "r", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(r'//.*', '', content)
        convertion_json = json.loads(content)

    errors = []
    for from_key, to_keys in convertion_json.items():
        to_keys = to_keys if isinstance(to_keys, list) else [to_keys]
        for to_key in to_keys:
            try:
                input_value = _dive_to_fetch_value(input_yaml_config, to_key)
            except ValueError:
                continue
            final_value = _dive_to_fetch_value(config_final, to_key)
            if str(input_value) != str(final_value):
                errors.append(
                    f"  - Key '{to_key}': input_yaml value = {input_value}, "
                    f"but ajet config sets it to = {final_value}"
                )

    if errors:
        error_msg = (
            "We found a configuration conflict between AgentJet and Verl! Input yaml contains keys that conflict with ajet default config values:\n"
            + "\n".join(errors)
            + "\nPlease use ajet.xxx to assign training parameters instead."
        )
        raise ValueError(error_msg)


def prepare_experiment_config(yaml_path, exp_base_dir, backbone, override_param_callback=None, storage=True):
    """
    Prepare experiment configuration by reading YAML, setting up backup directories,
    and copying necessary files for the experiment.

    Args:
        yaml_path: Path to the YAML configuration file
        exp_base_dir: Directory where experiment artifacts and backups should be stored
        backbone: Backbone identifier that controls config munging

    Returns:
        tuple: (yaml_backup_dst, exe_exp_base, exp_name, config_final)
    """
    assert yaml_path.endswith(".yaml"), "Configuration file must be a YAML file"
    exp_base = os.path.dirname(yaml_path)

    if not os.path.exists(exp_base):
        raise FileNotFoundError(f"Configuration file not found: {exp_base}")

    ## 0. read yaml & get experiment_name
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = input_yaml_config = yaml.safe_load(file)
    try:
        exp_name = config.get("ajet").get("experiment_name")
    except Exception:
        raise ValueError(f"Please set ajet field in yaml file. Current yaml:\n{config}")
    if exp_name is None or exp_name == "read_yaml_name":
        exp_name = os.path.basename(yaml_path).replace(".yaml", "")
        # add timestamp to exp_name
        timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
        exp_name = f"{exp_name}_{timestamp}"
    else:
        exp_name = exp_name.replace("|", "-")

    backup_dir = os.path.abspath(os.path.join(exp_base_dir, exp_name, "backup"))
    yaml_backup_dst = os.path.join(exp_base_dir, exp_name, "yaml_backup.yaml")
    yaml_backup_dst = os.path.abspath(yaml_backup_dst)
    exe_exp_base = os.path.dirname(yaml_backup_dst)

    if storage:
        logger.info("----------------------------------------")
        logger.info(f"Experiment Name: {exp_name}")
        logger.info(f"Experiment Backup Dir: {backup_dir}")
        logger.info(f"Experiment Yaml Dir: {yaml_backup_dst}")
        logger.info("----------------------------------------")

    ## 1. check exp_base/backup exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    else:
        total_seconds = 5
        for i in range(total_seconds):
            logger.warning(
                f"Warning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds..."
            )
            time.sleep(1)

    ## 2. copy files to backup
    BACK_TARGETS = os.environ.get("BACK_TARGETS", "").split(",")
    BACK_TARGETS = [p for p in BACK_TARGETS if os.path.exists(p)]

    for backup_target in BACK_TARGETS:
        logger.info(
            f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}"
        )
        shutil.copytree(
            backup_target,
            os.path.join(backup_dir, os.path.basename(backup_target)),
            dirs_exist_ok=True,
        )

    ## 3. copy yaml to backup
    yaml_backup_src = yaml_path
    shutil.copyfile(yaml_backup_src, yaml_backup_dst)

    ## 4. edit new yaml
    experiment_dir = f"{exp_base_dir}/{exp_name}"
    config = read_ajet_hierarchical_config(
        yaml_backup_dst,
        experiment_name=exp_name,
        backbone=backbone,
        write_to=yaml_backup_dst,
        experiment_dir=experiment_dir,
        override_param_callback=override_param_callback
    )
    config_final = expand_ajet_hierarchical_config(config, write_to=yaml_backup_dst)

    _validate_input_yaml_no_overlap_with_auto_convertion_config(input_yaml_config, config_final)

    if not storage:
        shutil.rmtree(os.path.join(exp_base_dir, exp_name))

    return yaml_backup_dst, exe_exp_base, exp_name, config_final
