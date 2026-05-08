from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AjetAlgorithm:
    adv_estimator: str = "grpo"


@dataclass
class AjetOptim:
    lr: float = 1e-6


@dataclass
class AjetTrainerCommon:
    n_gpus_per_node: int = 8
    algorithm: AjetAlgorithm = field(default_factory=AjetAlgorithm)
    optim: AjetOptim = field(default_factory=AjetOptim)
    use_kl_loss: bool = True
    use_kl_in_reward: bool = False
    kl_penalty_type: str = "kl"
    ppo_epochs: int = 1
    val_print_to_markdown_file_path: str | None = None
    train_print_to_markdown_file_path: str | None = None
    total_training_steps: int | None = None
    test_freq: int = 20
    save_freq: int = 20
    total_epochs: int = 50
    val_pass_n: int = 4
    val_before_train: bool = False


@dataclass
class AjetModel:
    path: str = "/path/to/model/such/as/Qwen/Qwen2___5-14B-Instruct"


@dataclass
class AjetData:
    max_prompt_length: int = 3000
    max_response_length: int = 15000
    # Note that this value is ignored when swarm_mode_sample_collection_method="rollout_until_all_clients_agree_sync_weight"
    train_batch_size: int = 32


@dataclass
class AjetRollout:
    user_workflow: str = "tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow"
    n_vllm_engine: int = 1  # this argument is NOT effective when NOT using trinity
    tensor_model_parallel_size: int = 1
    max_num_seqs: int = 64
    num_repeat: int = 8
    gpu_memory_utilization: float = 0.85
    compute_madness_checklist: List[str] = field(default_factory=list)


@dataclass
class AjetLora:
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str = "all-linear"
    load_format: str = "auto"
    layered_summon: bool = False


@dataclass
class AjetInterchangeServer:
    interchange_method: str = "ipc"
    interchange_server_port: Any = "auto"
    num_fastapi_process: int = 1
    max_fastapi_threads: int = 512
    max_inference_tracker_threads: int = 64
    already_started: bool = False


@dataclass
class HuggingfaceDatRepo:
    dataset_path: str = "gsm8k"
    dataset_name: str | None = None
    training_split: str = "train"
    validation_split: str = "validation"
    http_proxy_address: str = ""


@dataclass
class JsonlTrainingFp:
    file_path: str = ""
@dataclass
class JsonlDatasetFile:
    training: JsonlTrainingFp = field(default_factory=JsonlTrainingFp)
    validation: JsonlTrainingFp = field(default_factory=JsonlTrainingFp)


@dataclass
class AjetTaskReader:
    type: str = "huggingface_dat_repo"
    huggingface_dat_repo: HuggingfaceDatRepo = field(default_factory=HuggingfaceDatRepo)
    jsonl_dataset_file: JsonlDatasetFile = field(default_factory=JsonlDatasetFile)

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
    lora: AjetLora = field(default_factory=AjetLora)
    enable_swarm_mode: bool = True
    swarm_mode_sample_collection_method: str = "rollout_until_finish_enough_tasks"
    execute_test: bool = False
    interchange_server: AjetInterchangeServer = field(default_factory=AjetInterchangeServer)

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
