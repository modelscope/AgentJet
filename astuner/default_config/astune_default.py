from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AstunerAlgorithm:
    adv_estimator: Optional[str] = None


@dataclass
class AstunerTrainerCommon:
    n_gpus_per_node: Optional[int] = None
    algorithm: AstunerAlgorithm = field(default_factory=AstunerAlgorithm)


@dataclass
class AstunerModel:
    path: Optional[str] = None


@dataclass
class AstunerData:
    max_prompt_length: Optional[int] = None
    max_response_length: Optional[int] = None
    train_batch_size: Optional[int] = None


@dataclass
class AstunerRollout:
    agentscope_workflow: Optional[str] = None


@dataclass
class AstunerTaskReader:
    type: Optional[str] = None
    huggingface_dat_repo: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AstunerDefaultConfig:
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_dir: Optional[str] = None
    backbone: Optional[str] = None

    model: AstunerModel = field(default_factory=AstunerModel)
    data: AstunerData = field(default_factory=AstunerData)
    rollout: Optional[AstunerRollout] = field(default_factory=AstunerRollout)
    trainer_common: AstunerTrainerCommon = field(default_factory=AstunerTrainerCommon)
    task_reader: AstunerTaskReader = field(default_factory=AstunerTaskReader)


@dataclass
class Config:
    astuner: AstunerDefaultConfig = field(default_factory=AstunerDefaultConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @staticmethod
    def update_from_dict_recursive(config_as_dataclass, config_as_dict: Dict[str, Any]) -> None:
        # read and assign
        for key in config_as_dict.keys():
            target_value = config_as_dict[key]
            if isinstance(target_value, dict):
                if hasattr(config_as_dataclass, key):
                    setattr(
                        config_as_dataclass,
                        key,
                        Config.update_from_dict_recursive(
                            getattr(config_as_dataclass, key), target_value
                        ),
                    )
                else:
                    setattr(config_as_dataclass, key, target_value)
            else:
                if hasattr(config_as_dataclass, key):
                    setattr(config_as_dataclass, key, target_value)
                else:
                    setattr(config_as_dataclass, key, target_value)
        return config_as_dataclass
