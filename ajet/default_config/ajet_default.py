from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AjetAlgorithm:
    adv_estimator: str = "grpo"


@dataclass
class AjetTrainerCommon:
    n_gpus_per_node: int = 8
    algorithm: AjetAlgorithm = field(default_factory=AjetAlgorithm)


@dataclass
class AjetModel:
    path: str = "/path/to/model/such/as/Qwen/Qwen2___5-14B-Instruct"


@dataclass
class AjetData:
    max_prompt_length: int = 3000
    max_response_length: int = 15000
    train_batch_size: int = 32


@dataclass
class AjetRollout:
    agentscope_workflow: str = "tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow"
    n_vllm_engine: int = 1
    tensor_model_parallel_size: int = 1


@dataclass
class HuggingfaceDatRepo:
    dataset_path: str = "gsm8k"
    training_split: str = "train"
    validation_split: str = "validation"


@dataclass
class AjetTaskReader:
    type: str = "huggingface_dat_repo"
    huggingface_dat_repo: HuggingfaceDatRepo = field(default_factory=HuggingfaceDatRepo)


@dataclass
class AjetDefaultConfig:
    project_name: str = "ajet_default_project"
    experiment_name: str = "read_yaml_name"
    experiment_dir: str = "auto"
    backbone: str = "debug"

    model: AjetModel = field(default_factory=AjetModel)
    data: AjetData = field(default_factory=AjetData)
    rollout: AjetRollout = field(default_factory=AjetRollout)
    trainer_common: AjetTrainerCommon = field(default_factory=AjetTrainerCommon)
    task_reader: AjetTaskReader = field(default_factory=AjetTaskReader)


@dataclass
class Config:
    ajet: AjetDefaultConfig = field(default_factory=AjetDefaultConfig)

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
