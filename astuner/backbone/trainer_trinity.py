import asyncio
import os
from typing import Dict, List, Literal, Optional, cast

import datasets
import openai
import swanlab
from loguru import logger
from transformers import AutoTokenizer
from trinity.buffer.reader import READER
from trinity.buffer.reader.file_reader import TaskFileReader, _HFBatchReader
from trinity.buffer.schema import FORMATTER
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import Task as TrinityTask
from trinity.common.workflows.workflow import Workflow
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, Monitor

from astuner.backbone.warm_up import warm_up_process
from astuner.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from astuner.schema.trajectory import Sample
from astuner.task_rollout.native_parallel_worker import DynamicRolloutManager
from astuner.utils.config_utils import read_astune_config_with_cache
from astuner.utils.testing_utils import _test_if_test_mode


def get_astune_config_from_trinity_side():
    yaml_path = os.environ.get("ASTUNER_CONFIG_REDIRECT", None)
    if yaml_path is None:
        raise ValueError("ASTUNER_CONFIG_REDIRECT is not set in environment variables")
    astune_config = read_astune_config_with_cache(yaml_path)
    return astune_config


class TrinityRolloutManager(DynamicRolloutManager):
    def __init__(
        self,
        is_eval,
        task,
        llm_handle,
        tokenizer,
        config,
        llm_mode: Literal["local", "remote", "trinity"] = "trinity",
        **kwargs,
    ):
        self.is_eval = is_eval
        self.task = task
        self.tokenizer = tokenizer
        self.config = config
        self.llm_mode = llm_mode

        super().__init__(
            config=self.config,
            async_rollout_manager=llm_handle,
            max_parallel=1,
            max_llm_retries=1,
            tokenizer=tokenizer,
            llm_mode=llm_mode,
            **kwargs,
        )

    def convert_task(self, task: TrinityTask):
        from astuner.schema.task import Task

        d = {}
        for vip_key in ["main_query", "task_id", "env_type", "metadata", "init_messages"]:
            if vip_key not in task.raw_task:
                raise ValueError(f"Key {vip_key} not found in task.raw_task")
            d[vip_key] = task.raw_task[vip_key]
        return Task(**d)

    def thread_worker(self):
        observation_window = {
            "stop": [False],
            "step": [0],
            "token": [0],
        }
        astune_task = self.convert_task(self.task)
        return self.rollout_env_worker(
            task=astune_task,
            task_batch_index=0,
            task_tag=f"T{astune_task.task_id}#R",
            mode="sample" if not self.is_eval else "validate",
            task_thread_index=0,
            observation_window=observation_window,
        )

    async def run_in_new_thread(self) -> MultiAgentContextTracker:
        return cast(
            MultiAgentContextTracker,
            await asyncio.to_thread(self.thread_worker),
        )


@WORKFLOWS.register_module("astuner_workflow")
class ASTunerWorkflowWrap(Workflow):
    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: TrinityTask,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.task = task
        self.model_client = model.get_openai_async_client()
        self.is_eval = task.is_eval
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]

    async def run_async(self):
        astune_config = get_astune_config_from_trinity_side()
        warm_up_process(astune_config)
        tracker = await TrinityRolloutManager(
            is_eval=self.is_eval,
            task=self.task,
            llm_handle=self.model_client,
            tokenizer=AutoTokenizer.from_pretrained(self.model_client.model_path),
            config=astune_config,
        ).run_in_new_thread()

        sample_final = []
        try:
            sample_arr = tracker.group_tokenize()
        except Exception as e:
            raise e
        finally:
            tracker.generate_log(global_step="NA")
        sample_final += sample_arr

        exps = []
        for _, sample in enumerate(sample_final):
            sample: Sample
            input_ids = sample.input_ids
            prompt_ids = sample.prompt_ids
            response_ids = sample.response_ids
            response_loss_mask = sample.response_loss_mask

            logprobs = sample.response_logprobs
            reward = sample.step_reward  # reward scalar

            metrics = {
                "success_rate": tracker.reward_structure.success_rate,
                "madness": tracker.reward_structure.madness,
            }

            if (
                len(response_ids) + len(prompt_ids) == len(input_ids)
                and len(logprobs) == len(response_ids)
                and len(logprobs) > 0
            ):
                exp = Experience(
                    tokens=input_ids,  # [seq_length] prompt + response
                    prompt_length=len(
                        prompt_ids
                    ),  # Length of the prompt in tokens, used for generating attention masks
                    logprobs=logprobs,  # [resp_length]
                    reward=reward,  #
                    # advantages=None,
                    # returns=None,
                    info={},
                    metrics=metrics,  # for wandb logging (must be string:float)
                    response_text="",  # optional
                    prompt_text="",  # optional
                    #### for multi-turn experiences
                    action_mask=response_loss_mask,  # 1 stands for training, 0 stands for ignoring
                    messages=sample.messages,  #
                    # tools,
                    #### for dpo experiences
                    # chosen,
                    # rejected,
                    # chosen_messages,
                    # rejected_messages,
                    #### for multi-modal data
                    # multi_modal_inputs
                )
                exps += [exp]
            else:
                logger.exception("Data length mismatch when converting sample to experience.")
        return exps


try:

    @READER.register_module("astuner")
    class ASTunerTaskReader(TaskFileReader):
        def __init__(self, config):
            self.config = config
            self.read_batch_size = config.batch_size
            self.split = config.split

            astune_config = get_astune_config_from_trinity_side()

            from astuner.task_reader import RouterTaskReader, task_to_standard_dataset

            task_reader = RouterTaskReader(
                astune_config.astuner.task_reader.type,
                astune_config.astuner.task_reader,
            )

            dataset_segments = []
            if "train" in self.split:
                dataset_segments.append(task_to_standard_dataset(task_reader.get_training_tasks()))
            if "val" in self.split:
                dataset_segments.append(
                    task_to_standard_dataset(task_reader.get_validation_tasks())
                )
            if not dataset_segments:
                raise ValueError(
                    f"Unsupported split '{self.split}'. Expected to contain 'train' or 'val'."
                )

            concatenated_dataset = (
                dataset_segments[0]
                if len(dataset_segments) == 1
                else datasets.concatenate_datasets(dataset_segments)
            )

            self.dataset = _HFBatchReader(
                concatenated_dataset,
                name=self.config.name,
                default_batch_size=self.read_batch_size,
                total_epochs=self.config.total_epochs if not self.config.is_eval else 1,
                offset=self.config.index,
                drop_last=not self.config.is_eval,
                total_steps=self.config.total_steps,
                enable_progress_bar=self.config.enable_progress_bar,
            )
            self.formatter = FORMATTER.get("task")(self.config)

        def read(self, batch_size: Optional[int] = None) -> List:
            batch_size = batch_size or self.read_batch_size
            tasks = []
            samples, indices = self.dataset.read_batch(batch_size)
            for sample in samples:
                task = self.formatter.format(sample)
                tasks.append(task)
            return tasks

except Exception:
    pass


@MONITOR.register_module("swanlab")
class SwanlabMonitor(Monitor):
    """Monitor with SwanLab.

    This monitor integrates with SwanLab (https://swanlab.cn/) to track experiments.

    Supported monitor_args in config.monitor.monitor_args:
                - api_key (Optional[str]): API key for swanlab.login(). If omitted, will read from env
                    (SWANLAB_API_KEY, SWANLAB_APIKEY, SWANLAB_KEY, SWANLAB_TOKEN) or assume prior CLI login.
        - workspace (Optional[str]): Organization/username workspace.
        - mode (Optional[str]): "cloud" | "local" | "offline" | "disabled".
        - logdir (Optional[str]): Local log directory when in local/offline modes.
        - experiment_name (Optional[str]): Explicit experiment name. Defaults to "{name}_{role}".
        - description (Optional[str]): Experiment description.
        - tags (Optional[List[str]]): Tags to attach. Role and group are appended automatically.
        - id (Optional[str]): Resume target run id (21 chars) when using resume modes.
        - resume (Optional[Literal['must','allow','never']|bool]): Resume policy.
        - reinit (Optional[bool]): Whether to re-init on repeated init() calls.
    """

    def __init__(self, project: str, group: str, name: str, role: str, config) -> None:
        assert (
            swanlab is not None
        ), "swanlab is not installed. Please install it to use SwanlabMonitor."

        monitor_args = (
            (config.monitor.monitor_args or {})
            if config and getattr(config, "monitor", None)
            else {}
        )

        # Optional API login via code if provided; otherwise try environment, then rely on prior `swanlab login`.
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            try:
                swanlab.login(api_key=api_key, save=True)
            except Exception:
                # Best-effort login; continue to init which may still work if already logged in
                pass
        else:
            raise RuntimeError("Swanlab API key not found in environment variable SWANLAB_API_KEY.")

        # Compose tags (ensure list and include role/group markers)
        tags = monitor_args.get("tags") or []
        if isinstance(tags, tuple):
            tags = list(tags)
        if role and role not in tags:
            tags.append(role)
        if group and group not in tags:
            tags.append(group)

        # Determine experiment name
        exp_name = monitor_args.get("experiment_name") or f"{name}_{role}"
        self.exp_name = exp_name

        # Prepare init kwargs, passing only non-None values to respect library defaults
        init_kwargs = {
            "project": project,
            "workspace": monitor_args.get("workspace"),
            "experiment_name": exp_name,
            "description": monitor_args.get("description"),
            "tags": tags or None,
            "logdir": monitor_args.get("logdir"),
            "mode": monitor_args.get("mode") or "cloud",
            "settings": monitor_args.get("settings"),
            "id": monitor_args.get("id"),
            "resume": monitor_args.get("resume"),
            "reinit": monitor_args.get("reinit"),
        }
        # Strip None values to avoid overriding swanlab defaults
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        self.logger = swanlab.init(**init_kwargs)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

        run_info = self.logger.public.json()
        self.data_dashboard_url = run_info["cloud"]["experiment_url"]

    def log_table(self, table_name: str, experiences_table, step: int):
        assert (
            swanlab is not None
        ), "swanlab is not installed. Please install it to use SwanlabMonitor."

        # Convert pandas DataFrame to SwanLab ECharts Table
        headers: List[str] = list(experiences_table.columns)
        # Ensure rows are native Python types
        rows: List[List[object]] = experiences_table.astype(object).values.tolist()
        try:
            tbl = swanlab.echarts.Table()
            tbl.add(headers, rows)
            swanlab.log({table_name: tbl}, step=step)
        except Exception:
            # Fallback: log as CSV string if echarts table is unavailable
            csv_str = experiences_table.to_csv(index=False)
            swanlab.log({table_name: csv_str}, step=step)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        # SwanLab doesn't use commit flag; keep signature for compatibility
        assert (
            swanlab is not None
        ), "swanlab is not installed. Please install it to use SwanlabMonitor."
        swanlab.log(data, step=step)
        self.console_logger.info(f"Step {step}: {data}")

        astune_config = get_astune_config_from_trinity_side()
        experiment_dir = astune_config.astuner.experiment_dir
        trinity_log = f"{experiment_dir}/{self.exp_name}.log"

        with open(trinity_log, "a") as f:
            f.write(f"Step {step}: {data}\n")

        if astune_config.astuner.execute_test:  # apply a test probe
            if "critic/score/mean" in data:
                return
            if "experience_pipeline/group_advantages/reward_mean/mean" not in data:
                return
            test_robot_data = {}
            test_robot_data["step"] = step
            test_robot_data["data_dashboard_url"] = self.data_dashboard_url
            test_robot_data["reward_for_test_robot"] = data[
                "experience_pipeline/group_advantages/reward_mean/mean"
            ]
            _test_if_test_mode(key="reward_probe", value=test_robot_data, config=astune_config)

    def close(self) -> None:
        try:
            # Prefer run.finish() if available
            if hasattr(self, "logger") and hasattr(self.logger, "finish"):
                self.logger.finish()
            else:
                # Fallback to global finish
                swanlab.finish()
        except Exception:
            pass

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}
