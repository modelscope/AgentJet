import os
from hydra import initialize, compose
from omegaconf import DictConfig

def read_astune_config(yaml_fp):
    yaml_fp = os.path.relpath(yaml_fp, os.path.dirname(__file__))   # do not try to understand this line, hydra is too weird

    def load_hydra_config(config_path: str, config_name: str) -> DictConfig:
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name, overrides=[])
            return cfg

    dir_path = os.path.dirname(yaml_fp)
    file_name = os.path.basename(yaml_fp)
    return load_hydra_config(config_path=dir_path, config_name=file_name)

def dump_yaml_config(cfg: DictConfig, yaml_fp: str):
    from omegaconf import OmegaConf
    with open(yaml_fp, 'w') as f:
        OmegaConf.save(cfg, f)
    return yaml_fp