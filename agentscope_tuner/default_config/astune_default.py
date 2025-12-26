from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AstunerAlgorithm:
    adv_estimator: str = "grpo"


@dataclass
class AstunerTrainerCommon:
    n_gpus_per_node: int = 8
    algorithm: AstunerAlgorithm = field(default_factory=AstunerAlgorithm)


@dataclass
class AstunerModel:
    path: str = "/path/to/model/such/as/Qwen/Qwen2___5-14B-Instruct"


@dataclass
class AstunerData:
    max_prompt_length: int = 3000
    max_response_length: int = 15000
    train_batch_size: int = 32


@dataclass
class AstunerRollout:
    agentscope_workflow: str = "tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow"
    n_vllm_engine: int = 1
    tensor_model_parallel_size: int = 1


@dataclass
class HuggingfaceDatRepo:
    dataset_path: str = "gsm8k"
    training_split: str = "train"
    validation_split: str = "validation"


@dataclass
class AstunerTaskReader:
    type: str = "huggingface_dat_repo"
    huggingface_dat_repo: HuggingfaceDatRepo = field(default_factory=HuggingfaceDatRepo)


@dataclass
class AstunerDefaultConfig:
    project_name: str = "astune_default_project"
    experiment_name: str = "read_yaml_name"
    experiment_dir: str = "auto"
    backbone: str = "debug"

    model: AstunerModel = field(default_factory=AstunerModel)
    data: AstunerData = field(default_factory=AstunerData)
    rollout: AstunerRollout = field(default_factory=AstunerRollout)
    trainer_common: AstunerTrainerCommon = field(default_factory=AstunerTrainerCommon)
    task_reader: AstunerTaskReader = field(default_factory=AstunerTaskReader)


@dataclass
class Config:
    astuner: AstunerDefaultConfig = field(default_factory=AstunerDefaultConfig)

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries."""
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, "__dataclass_fields__"):
                result[key] = Config._to_dict(value)
            else:
                result[key] = value
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary, including extra attributes."""
        return Config._to_dict(self)

    @staticmethod
    def update_from_dict_recursive(config_as_dataclass, config_as_dict: Dict[str, Any]) -> "Config":
        # read and assign
        for key in config_as_dict.keys():
            target_value = config_as_dict[key]
            if isinstance(target_value, dict):
                if hasattr(config_as_dataclass, key):
                    if isinstance(getattr(config_as_dataclass, key), dict):
                        setattr(config_as_dataclass, key, target_value)
                        continue
                    else:
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
                setattr(config_as_dataclass, key, target_value)
        return config_as_dataclass
