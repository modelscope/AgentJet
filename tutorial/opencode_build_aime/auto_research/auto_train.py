# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Auto Research Client
Configurable for batch_size, max_response_length_in_one_turn, and KL-regularization experiments

python -m tutorial.opencode_build_aime.auto_research.auto_train
"""

import os
import argparse
import time
import statistics
from inspect import signature
from urllib.parse import urlparse
from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader, HuggingFaceTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from tutorial.opencode_build_aime import download_data
from tqdm import tqdm


DEFAULT_PROJECT_NAME = "subject14_aime_baseline_group_8_bs32"

# Framework-ablation: the OpenAI-SDK agent loop is agent_run_v3 (the baseline); the
# langchain / agentscope / rawhttp variants reuse v3's sandbox tool + reward and
# differ only in the LLM client. `--framework` picks which one drives the trainer;
# data, prompt, tool, reward, and all hyperparameters stay identical across arms.
_FRAMEWORK_MODULES = {
    "openai": "tutorial.opencode_build_aime.agent_run_v3",
    "langchain": "tutorial.opencode_build_aime.agent_run_langchain",
    "agentscope": "tutorial.opencode_build_aime.agent_run_agentscope",
    "rawhttp": "tutorial.opencode_build_aime.agent_run_rawhttp",
}

# API-ablation: which OpenAI endpoint shape does the openai-sdk client hit?
#   - "chat"      → POST /v1/chat/completions  (legacy default; reuse v3 directly)
#   - "responses" → POST /v1/responses          (new; uses agent_run_responses)
#   - "messages"  → POST /v1/messages           (Anthropic Messages API; uses agent_run_messages)
# All three share the same model, prompt, tool, reward, and dataset.
_API_MODULES = {
    "chat": "tutorial.opencode_build_aime.agent_run_v3",
    "responses": "tutorial.opencode_build_aime.agent_run_responses",
    "messages": "tutorial.opencode_build_aime.agent_run_messages",
}


def resolve_execute_agent(framework: str):
    import importlib
    module_path = _FRAMEWORK_MODULES.get(framework)
    if module_path is None:
        raise ValueError(f"Unknown framework {framework!r}; choose from {sorted(_FRAMEWORK_MODULES)}")
    return importlib.import_module(module_path).execute_agent


def resolve_execute_agent_by_api(api: str):
    """Pick the agent module by OpenAI API surface (chat vs responses).

    Takes precedence over `--framework` when both are given: the API ablation
    is the actual variable under test, while `--framework` is kept for the
    separate framework-ablation arms (langchain / agentscope / rawhttp), all
    of which currently use the chat-completions shape.
    """
    import importlib
    module_path = _API_MODULES.get(api)
    if module_path is None:
        raise ValueError(f"Unknown api {api!r}; choose from {sorted(_API_MODULES)}")
    return importlib.import_module(module_path).execute_agent


def scalar_reward(reward: float | list[float] | None) -> float:
    if not isinstance(reward, (int, float)):
        raise TypeError(f"Expected scalar reward, got {type(reward).__name__}: {reward!r}")
    return float(reward)


def agentjet_job_kwargs_from_args(args: argparse.Namespace) -> dict:
    job_arg_names = set(signature(AgentJetJob.__init__).parameters) - {"self"}
    return {name: value for name, value in vars(args).items() if name in job_arg_names}


def extract_swarm_port(swarm_url: str) -> int:
    parsed = urlparse(swarm_url)
    if parsed.port is None:
        raise ValueError(f"Swarm URL must include an explicit port: {swarm_url}")
    return parsed.port


def validate_length_config(
    max_prompt_length: int,
    max_response_length: int,
    max_response_length_in_one_turn: int,
    max_model_len: int,
):
    if max_prompt_length + max_response_length > max_model_len:
        raise ValueError(
            "Invalid length config: max_prompt_length + max_response_length must be <= max_model_len"
        )
    if max_response_length_in_one_turn > max_response_length:
        raise ValueError(
            "Invalid length config: max_response_length_in_one_turn must be <= max_response_length"
        )


def load_eval_tasks(test_dataset: str, label: str = "") -> list:
    eval_tasks = []
    if os.path.exists(test_dataset):
        eval_reader = HuggingFaceTaskReader(
            AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=test_dataset))
        )
        for t in eval_reader.generate_training_tasks():
            eval_tasks.append(t)
        print(f"[INFO] Loaded {len(eval_tasks)} eval tasks from {label or test_dataset}")
    else:
        print(f"[WARN] Eval dataset not found: {test_dataset}. Skipping {label or test_dataset}.")
    return eval_tasks


class AIMEAutoResearchEval:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.framework = getattr(args, "framework", "openai")
        self.api = getattr(args, "api", "chat")
        # API ablation takes precedence over framework ablation when set to an
        # OpenAI/Anthropic API surface other than the chat-completions default;
        # otherwise we fall back to the framework picker so the existing
        # langchain/agentscope/rawhttp arms still work.
        if self.api in ("responses", "messages"):
            self.execute_agent = resolve_execute_agent_by_api(self.api)
        else:
            self.execute_agent = resolve_execute_agent(self.framework)
        self.swarm_url = args.swarm_url or os.getenv("AJET_SWARM_URL", "http://localhost:10086")
        self.result_dir = args.result_dir
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.test_datasets = {
            # "AIME-2024": os.path.join(data_dir, "aime-2024.parquet"),
            "AIME-2025": os.path.join(data_dir, "aime-2025.parquet"),
            "AIME-2026": os.path.join(data_dir, "aime-2026.parquet"),
            "DAPO-Math-Tiny-Val": os.path.join(data_dir, "dapo-math-tiny-val.parquet"),
        }
        self.swarm_worker: SwarmClient | None = None
        self.ajet_job: AgentJetJob | None = None
        self.eval_tasks_by_set: dict[str, list[Task]] = {}
        self.eval_interval = args.eval_interval
        self.eval_k = args.eval_k
        self.max_env_worker = args.max_env_worker

    def setup_eval_tasks(self):
        eval_downloaders = {
            "AIME-2024": download_data.ensure_aime_2024,
            "AIME-2025": download_data.ensure_aime_2025,
            "AIME-2026": download_data.ensure_aime_2026,
        }
        for label, path in self.test_datasets.items():
            if not os.path.exists(path):
                downloader = eval_downloaders.get(label)
                if downloader is None:
                    print(f"[WARN] {label} parquet missing at {path} and no downloader registered. Skipping.")
                    continue
                print(f"[INFO] {label} parquet missing, downloading...")
                try:
                    downloader()
                except Exception as e:
                    print(f"[WARN] Failed to download {label}: {e}")
                    continue
            tasks = load_eval_tasks(path, label=label)
            if tasks:
                self.eval_tasks_by_set[label] = tasks

    def eval_rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None, "setup() must be called before eval_rollout()"
        assert self.ajet_job is not None, "AgentJet job must be initialized before eval_rollout()"
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=120, episode_type="eval"
        )
        try:
            workflow_output = self.execute_agent(task, api_baseurl_key, self.ajet_job)
            return scalar_reward(workflow_output.reward)
        finally:
            self.swarm_worker.abort_episode(episode_uuid)

    def run_eval(self, n_global_step: int):
        if not self.eval_tasks_by_set:
            return
        eval_log_path = os.path.join(self.result_dir, "eval_results.log")

        for label, eval_tasks in self.eval_tasks_by_set.items():
            self._run_eval_one(n_global_step, label, eval_tasks, eval_log_path)

    def _run_eval_one(self, n_global_step: int, label: str, eval_tasks: list, eval_log_path: str):
        k = self.eval_k
        total_rollouts = len(eval_tasks) * k
        print(f"\n[EVAL @ step {n_global_step}] Running {label} eval on {len(eval_tasks)} tasks x {k} (pass@{k})...")
        per_task_rewards = [[] for _ in eval_tasks]
        pbar = tqdm(total=total_rollouts, desc=f"EVAL {label} @ step {n_global_step}")

        with ThreadPoolExecutor(max_workers=self.max_env_worker) as eval_executor:
            future_to_idx = {
                eval_executor.submit(self.eval_rollout, t): i
                for i, t in enumerate(eval_tasks)
                for _ in range(k)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    per_task_rewards[idx].append(fut.result())
                except Exception as e:
                    print(f"[EVAL] future error: {e}")
                pbar.update(1)
        pbar.close()

        flat = [r for rs in per_task_rewards for r in rs if r is not None]
        if flat:
            avg = sum(flat) / len(flat)
            std_reward = statistics.pstdev(flat) if len(flat) > 1 else 0.0
            pass1 = sum(1 for r in flat if r > 0) / len(flat)
            num_all_success_tasks = sum(
                1 for rs in per_task_rewards if rs and all((r is not None and r > 0) for r in rs)
            )
            solved_tasks = [rs for rs in per_task_rewards if any((r is not None and r > 0) for r in rs)]
            num_pass_n_tasks = len(solved_tasks)
            passk = num_pass_n_tasks / len(per_task_rewards)
            summary = (
                f"[EVAL @ step {n_global_step}] {label}  mean_reward={avg:.4f}  std_reward={std_reward:.4f}  "
                f"task_pass_rate@1={pass1*100:.2f}%  task_pass_rate@{k}={passk*100:.2f}%  "
                f"n_tasks={len(per_task_rewards)}  n_rollouts={len(flat)}"
            )
            print(summary)
            with open(eval_log_path, "a") as f:
                f.write(summary + "\n")

            val_result_path = os.path.join(self.result_dir, "val_results.md")
            with open(val_result_path, "a") as f:
                f.write(f"\n## Step {n_global_step}\n")
                f.write(f"- dataset: {label}\n")
                f.write(f"- pass_n: {k}\n")
                f.write(f"- total_tasks: {len(per_task_rewards)}\n")
                f.write(f"- num_all_success_tasks: {num_all_success_tasks}\n")
                f.write(f"- num_pass_n_tasks: {num_pass_n_tasks}\n")
                f.write(f"- task_pass_rate@1: {pass1*100:.2f}%\n")
                f.write(f"- task_pass_rate@{k}: {passk*100:.2f}%\n")
                f.write(f"- mean_reward: {avg:.4f}\n")
                f.write(f"- std_reward: {std_reward:.4f}\n")
                f.write(f"- n_rollouts: {len(flat)}\n")
        else:
            print(f"[EVAL @ step {n_global_step}] {label}  no valid rewards")


class AIMEAutoResearchTrainer(AIMEAutoResearchEval):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.batch_size = args.batch_size
        self.resolved_yaml_path = args.resolved_yaml_path or os.path.join(args.result_dir, "resolved_swarm_config.yaml")
        self.prepare_only = args.prepare_only
        self.total_training_steps = args.total_training_steps
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.train_dataset = os.path.join(data_dir, "dapo-math-17k.parquet")
        self.dataset: RouterTaskReader | None = None
        self.grpo_n = args.grpo_repeat

        os.makedirs(args.result_dir, exist_ok=True)
        model_path = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B")
        validate_length_config(
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_response_length_in_one_turn=args.max_response_length_in_one_turn,
            max_model_len=args.max_model_len,
        )
        job_kwargs = agentjet_job_kwargs_from_args(args)
        job_kwargs.update(
            ensure_new_experiment=True,
            experiment_dir=args.result_dir,
            algorithm="grpo",
            model=model_path,
            swarm_mode=True,
            swarm_mode_sample_collection_method="rollout_until_all_clients_agree_sync_weight",
            num_repeat=args.grpo_repeat,
            logging="swanlab",
            compute_madness_checklist=["nonsense", "un-paired-think"],
            val_print_to_markdown_file_path=os.path.join(args.result_dir, "val_results.md"),
            train_print_to_markdown_file_path=os.path.join(args.result_dir, "train_results.md"),
            timeline_compare_level='token',
            tensor_model_parallel_size=1,
        )
        self.ajet_job = AgentJetJob(**job_kwargs)
        self.ajet_job.config.ajet.trainer_common.loss_weight_normalization_episode_level = args.loss_weight_normalization_episode_level
        self.ajet_job.config.ajet.trainer_common.advantage_estimation_episode_level = args.advantage_estimation_episode_level

        # [OPD] On-Policy Distillation config (off by default; enable via --opd-enabled).
        if getattr(args, "opd_enabled", False):
            cfg = self.ajet_job.config.ajet
            cfg.teacher_model.teacher_opd_enabled = True
            cfg.teacher_model.teacher_model_vllm_url = args.opd_teacher_url
            cfg.teacher_model.teacher_model_name = args.opd_teacher_name
            cfg.teacher_model.teacher_model_api_key = args.opd_teacher_api_key or "EMPTY"
            cfg.teacher_model.teacher_topk = args.opd_topk
            cfg.teacher_model.teacher_tokenizer_path = args.opd_teacher_tokenizer
            cfg.opd.loss_mode = args.opd_loss_mode
            cfg.opd.use_policy_gradient = args.opd_use_policy_gradient
            cfg.opd.use_task_rewards = args.opd_use_task_rewards
            cfg.opd.distillation_loss_coef = args.opd_coef
            print(f"[OPD] enabled: teacher={args.opd_teacher_name} @ {args.opd_teacher_url} "
                  f"loss_mode={args.opd_loss_mode} topk={args.opd_topk} "
                  f"use_pg={args.opd_use_policy_gradient} use_task={args.opd_use_task_rewards} coef={args.opd_coef}")

    def setup(self):
        if not os.path.exists(self.train_dataset):
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_dataset}\n"
                "Please run: proxychains python -m tutorial.opencode_build_aime.download_data"
            )

        self.ajet_job.dump_job_as_yaml(self.resolved_yaml_path)

        if self.prepare_only:
            return

        self.dataset = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=self.train_dataset)
            )
        )

        self.swarm_worker = SwarmClient(self.swarm_url, verbose=False, agentjet_job=self.ajet_job)
        if os.getenv("SYNC_CODE", "0") == "1":
            self.swarm_worker.sync_train_code_from_dir(os.getcwd(), force_restart=True)
        self.swarm_worker.auto_sync_train_config_and_start_engine(
            self.ajet_job,
            force_restart=os.getenv("AJET_SWARM_RESTART", "0") == "1"
        )
        self.setup_eval_tasks()

    def rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None, "setup() must be called before rollout()"
        assert self.ajet_job is not None, "AgentJet job must be initialized before rollout()"
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120)
        workflow_output = self.execute_agent(task, api_baseurl_key, self.ajet_job)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return scalar_reward(workflow_output.reward)

    def train(self):
        assert self.swarm_worker is not None and self.dataset is not None, "setup() must be called before train()"
        if not os.getenv("SKIP_INITIAL_EVAL", False): self.run_eval(0)

        max_parallel = 128
        executor = PeriodicDrainThreadPoolExecutor(
            workers=self.batch_size * self.grpo_n,
            max_parallel=max_parallel,
            auto_retry=True,
        )

        num_epochs = 10000
        last_eval_step = 0
        for epoch in range(num_epochs):
            for _, task in enumerate(self.dataset.generate_training_tasks()):
                for _ in range(self.grpo_n):
                    _, drained_results = executor.submit_with_periodic_drain(
                        fn=self.rollout, task=task
                    )
                    if drained_results:
                        # when `self.batch_size * self.grpo_n` episode has completed
                        self.swarm_worker.agree_sync_weight()

                n_global_step = self.swarm_worker.get_global_step()

                time_to_eval = n_global_step >= last_eval_step + self.eval_interval
                if time_to_eval:
                    self.run_eval(n_global_step)
                    last_eval_step = n_global_step

                if n_global_step >= self.total_training_steps:
                    break

            if n_global_step >= self.total_training_steps:
                break

        finish_flag = os.path.join(self.result_dir, "finish.flag")
        with open(finish_flag, "w") as f:
            f.write(f"Training completed at {time.time()}\n")

        print("\n[INFO] Training complete!")

    def run(self):
        self.setup()
        if self.prepare_only:
            print(f"[INFO] Prepared run artifacts only. Resolved config written to {self.resolved_yaml_path}")
            return
        self.train()


def main():
    parser = argparse.ArgumentParser(description="AIME Auto Research Swarm Training")
    parser.add_argument("--batch-size", default=16, type=int, required=True, help="Training batch size")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Experiment name for this run")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory to store results")
    parser.add_argument("--swarm-url", type=str, default="http://localhost:10086",
                        help="Swarm server URL")
    parser.add_argument("--project-name", type=str, default=DEFAULT_PROJECT_NAME,
                        help="Shared project name used for this research line")
    parser.add_argument("--framework", type=str, default="openai",
                        choices=sorted(_FRAMEWORK_MODULES),
                        help="Client framework for the agent loop "
                             "(openai=agent_run_v3 baseline / langchain / agentscope / rawhttp). "
                             "The only variable in the framework-ablation experiment.")
    parser.add_argument("--api", type=str, default="chat",
                        choices=sorted(_API_MODULES),
                        help="LLM API surface to drive the agent with: "
                             "'chat' (POST /v1/chat/completions, default), "
                             "'responses' (POST /v1/responses), or "
                             "'messages' (POST /v1/messages, Anthropic Messages API). "
                             "When 'responses' or 'messages' is selected, --framework is ignored.")
    parser.add_argument("--resolved-yaml-path", type=str, default=None,
                        help="Optional output path for the fully resolved swarm config yaml")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build the config, dump the resolved yaml, and exit without training")
    parser.add_argument("--max-response-length-in-one-turn", type=int, default=10000,
                        help="Max response length in one turn")
    parser.add_argument("--max-prompt-length", type=int, default=3000,
                        help="Maximum prompt length")
    parser.add_argument("--max-response-length", type=int, default=20000,
                        help="Maximum total response length")
    parser.add_argument("--max-model-len", type=int, default=23000,
                        help="Maximum total model context length")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1,
                        help="Tensor-parallel size for the vLLM rollout engine")
    parser.add_argument("--total-training-steps", type=int, default=100,
                        help="Hard cap on total training steps")
    parser.add_argument("--n-gpu", type=int, default=8,
                        help="Number of GPUs reserved for the swarm server")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes reserved for training")
    parser.add_argument("--max-env-worker", type=int, default=128,
                        help="Estimated number of parallel environment workers")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N global steps")
    parser.add_argument("--eval-k", type=int, default=4,
                        help="Number of rollouts per eval task (pass@k)")
    parser.add_argument("--grpo-repeat", type=int, default=8,
                        help="GRPO num_repeat per training task")
    parser.add_argument("--ppo-epochs", type=int, default=1,
                        help="Number of PPO epochs per update")
    parser.add_argument("--mini-batch-num", type=int, default=1,
                        help="Number of mini-batches per PPO update")
    parser.add_argument("--use-kl-loss", action=argparse.BooleanOptionalAction, default=True,
                        help="Add KL-divergence regularization to the actor's policy loss "
                             "(use --no-use-kl-loss to disable)")
    parser.add_argument("--use-kl-in-reward", action=argparse.BooleanOptionalAction, default=False,
                        help="Subtract a KL penalty from the reward signal during advantage "
                             "computation (use --no-use-kl-in-reward to disable)")
    parser.add_argument("--kl-penalty-type", type=str, default="kl",
                        choices=["kl", "abs", "mse", "low_var_kl", "full"],
                        help="KL divergence estimator used for the reward-shaping path "
                             "when --use-kl-in-reward is enabled")
    parser.add_argument("--loss-weight-normalization-episode-level", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Weight loss contributions so each episode contributes equally")
    parser.add_argument("--advantage-estimation-episode-level", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Compute GRPO advantage statistics at episode level")
    # ---- OPD (On-Policy Distillation) ----
    parser.add_argument("--opd-enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable OPD teacher distillation (ajet.teacher_model.teacher_opd_enabled)")
    parser.add_argument("--opd-teacher-url", type=str, default="",
                        help="Remote teacher vLLM OpenAI base URL, e.g. http://22.5.158.93:8000/v1")
    parser.add_argument("--opd-teacher-name", type=str, default="",
                        help="Served model name on the teacher vLLM server")
    parser.add_argument("--opd-teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--opd-teacher-tokenizer", type=str, default="",
                        help="Path to teacher tokenizer (only needed for SimCT cross-tokenizer OPD)")
    parser.add_argument("--opd-loss-mode", type=str, default="k3",
                        help="OPD loss: kl|k1|abs|mse|k2|low_var_kl|k3|k1+|k3+|kl+|forward_kl_topk|simct")
    parser.add_argument("--opd-topk", type=int, default=0,
                        help="Teacher topk (0=estimator sampled-token logprob; >0=top-k forward KL / SimCT candidates)")
    parser.add_argument("--opd-use-policy-gradient", action=argparse.BooleanOptionalAction, default=False,
                        help="OPD-PG (advantage=-KL) vs supervised backprop")
    parser.add_argument("--opd-use-task-rewards", action=argparse.BooleanOptionalAction, default=True,
                        help="Blend OPD with task PPO loss (else distillation only)")
    parser.add_argument("--opd-coef", type=float, default=1.0, help="distillation_loss_coef")
    args = parser.parse_args()

    trainer = AIMEAutoResearchTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
