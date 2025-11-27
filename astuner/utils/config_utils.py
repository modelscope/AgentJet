import os
import shutil
import time

import yaml
from best_logger import print_dict
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig


def read_astune_config(yaml_fp):
    """Load a Hydra configuration relative to this module."""
    yaml_fp = os.path.relpath(
        yaml_fp, os.path.dirname(__file__)
    )  # do not try to understand this line, hydra is too weird

    def load_hydra_config(config_path: str, config_name: str) -> DictConfig:
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name, overrides=[])
            return cfg

    dir_path = os.path.dirname(yaml_fp)
    file_name = os.path.basename(yaml_fp)
    return load_hydra_config(config_path=dir_path, config_name=file_name)


def dump_yaml_config(cfg: DictConfig, yaml_fp: str):
    """Persist the provided OmegaConf config to ``yaml_fp``."""
    from omegaconf import OmegaConf

    with open(yaml_fp, "w") as f:
        OmegaConf.save(cfg, f)
    return yaml_fp


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
    with open(from_config_fp, "r") as file:
        from_config = yaml.safe_load(file)
    with open(to_config_fp, "r") as file:
        to_config = yaml.safe_load(file)

    # read convertion json
    import json

    with open(convertion_json_fg, "r") as file:
        convertion_json = json.load(file)

    logger.success("----------------------------------------------------")

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
                    logger.info(
                        f"[Note]: Aligned parameter from [trinity.{full_key}] to [{full_key}] with value: [{value}]"
                    )

        recursive_copy(trinity_config, to_config)

    logger.success("----------------------------------------------------")
    time.sleep(1)

    for from_key, to_keys in convertion_json.items():
        # get value from from_config
        keys = from_key.split(".")
        value = from_config
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        if value is None:
            logger.warning(
                f"[Warning]: Cannot find value for key: {from_key} in {from_config_fp}, skip aligning {to_keys}"
            )
            continue

        to_keys = to_keys if isinstance(to_keys, list) else [to_keys]
        for to_key in to_keys:
            keys = to_key.split(".")
            sub_config = to_config
            for key in keys[:-1]:
                if key not in sub_config:
                    sub_config[key] = {}
                sub_config = sub_config[key]
            sub_config[keys[-1]] = value
            logger.success(
                f"[Note]: Aligned parameter from [{from_key}] to [{to_key}] with value: [{value}]"
            )

    logger.success("----------------------------------------------------")
    time.sleep(1)

    # save to_config_fp
    with open(to_config_fp, "w") as file:
        yaml.dump(to_config, file)
    # logger.success(f"Saved aligned configuration to {to_config_fp}")
    print_dict({"Note": f"Saved aligned configuration to {to_config_fp}"})


def read_astune_hierarchical_config(
    yaml_fp, exp_name, backbone, write_to=None, exp_dir="launcher_record"
):
    with open(yaml_fp, "r") as file:
        config = yaml.safe_load(file)
    config["astuner"]["experiment_name"] = exp_name
    config["astuner"]["experiment_dir"] = os.path.join(exp_dir, exp_name)
    config["astuner"]["backbone"] = backbone
    # remove extra config of verl for trinity
    if backbone == "debug":
        config["defaults"].remove("ppo_trainer")
        config["defaults"].remove("trinity_default")
        config["hydra"]["searchpath"].remove("file://astuner/default_config/trinity")
        config["hydra"]["searchpath"].remove("file://external/verl/verl/trainer/config")
    # remove extra config of verl for trinity
    if backbone == "trinity":
        config["defaults"].remove("ppo_trainer")
        config["defaults"].remove("verl_default")
        config["hydra"]["searchpath"].remove("file://external/verl/verl/trainer/config")
        config["hydra"]["searchpath"].remove("file://astuner/default_config/verl")
    # remove extra config of trinity for verl
    if backbone == "verl":  #  or args.backbone == "debug"
        config["defaults"].remove("trinity_default")
        config["hydra"]["searchpath"].remove("file://astuner/default_config/trinity")
    if write_to:
        with open(write_to, "w") as file:
            yaml.dump(config, file)
    return config


def expand_astune_hierarchical_config(config, write_to=None):
    # create temp yaml file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_yaml:
        yaml_path = temp_yaml.name
        with open(yaml_path, "w") as file:
            yaml.dump(config, file)
        full_config = read_astune_config(yaml_path)
        yaml_path = dump_yaml_config(full_config, yaml_fp=yaml_path)
        # put inherit info back
        with open(yaml_path, "r") as file:
            config_final = yaml.safe_load(file)
        config_final["defaults"] = config["defaults"]
        config_final["hydra"] = config["hydra"]

    if write_to:
        with open(write_to, "w") as file:
            yaml.dump(config_final, file)

    return config_final


def prepare_experiment_config(yaml_path, exp_dir, backbone):
    """
    Prepare experiment configuration by reading YAML, setting up backup directories,
    and copying necessary files for the experiment.

    Args:
        yaml_path: Path to the YAML configuration file
        exp_dir: Directory where experiment artifacts and backups should be stored
        backbone: Backbone identifier that controls config munging

    Returns:
        tuple: (yaml_backup_dst, exe_exp_base, exp_name, config_final)
    """
    assert yaml_path.endswith(".yaml"), "Configuration file must be a YAML file"
    exp_base = os.path.dirname(yaml_path)

    if not os.path.exists(exp_base):
        raise FileNotFoundError(f"Configuration file not found: {exp_base}")

    ## 0. read yaml & get experiment_name
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    exp_name = config.get("astuner").get("experiment_name")
    if exp_name is None or exp_name == "read_yaml_name":
        if exp_name is not None:
            exp_name = exp_name.replace("|", "-")
        exp_name = os.path.basename(yaml_path).replace(".yaml", "")
    else:
        exp_name = exp_name.replace("|", "-")

    backup_dir = os.path.join(exp_dir, exp_name, "backup")
    yaml_backup_dst = os.path.join(exp_dir, exp_name, "yaml_backup.yaml")
    exe_exp_base = os.path.dirname(yaml_backup_dst)
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
    config = read_astune_hierarchical_config(
        yaml_backup_dst, exp_name, backbone, write_to=yaml_backup_dst, exp_dir=exp_dir
    )
    config_final = expand_astune_hierarchical_config(config, write_to=yaml_backup_dst)

    return yaml_backup_dst, exe_exp_base, exp_name, config_final
