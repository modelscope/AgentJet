# -*- coding: utf-8 -*-
"""
AppWorld swarm client (driver) for example_cocktail_rl_v2.

python -m tutorial.example_cocktail_rl_v2.train_appworld_as_swarm_client_0
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task
from ajet.utils.env_service_client.env_client_ng import EnvClient

from tutorial.example_cocktail_rl_v2.cocktail_v2_config import CocktailV2Config, cocktail_v2_config_from_env
from tutorial.example_cocktail_rl_v2.cocktail_v2_runner import CocktailSwarmRunner


# ---------------- Engine config (was cocktail_rl_conf.yaml) ----------------

def build_cocktail_ajet_job(cfg: CocktailV2Config) -> AgentJetJob:
    """Construct the AgentJetJob that drives the swarm engine.

    Every value is read from `cfg`. There are no hardcoded constants in this
    function -- CocktailV2Config is the single source of truth for the entire
    engine config. Fields not exposed as AgentJetJob kwargs are set on
    `ajet_job.config.ajet.*` after construction and shipped to the engine via
    `Config.to_dict()`.
    """
    ajet_job = AgentJetJob(
        # base_yaml_config=None -> use ajet/default_config/ajet_swarm_default.yaml
        project_name=cfg.project_name,
        experiment_name=cfg.experiment_name,
        experiment_dir=cfg.experiment_dir,
        model=cfg.model_path,
        algorithm=cfg.algorithm,
        num_repeat=cfg.grpo_n,
        # batch_size is ignored under rollout_until_all_clients_agree_sync_weight,
        # but we mirror cfg.total_batch_size so the dumped engine config reads coherently.
        batch_size=cfg.total_batch_size,
        swarm_mode=cfg.swarm_mode,
        swarm_mode_sample_collection_method=cfg.swarm_mode_sample_collection_method,
        max_env_worker=cfg.max_env_worker,
        max_prompt_length=cfg.max_prompt_length,
        max_response_length=cfg.max_response_length,
        max_response_length_in_one_turn=cfg.max_response_length_in_one_turn,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        compute_madness_checklist=list(cfg.compute_madness_checklist),
        n_gpu=cfg.n_gpu,
        logging=cfg.logging,
        use_kl_loss=cfg.use_kl_loss,
        use_kl_in_reward=cfg.use_kl_in_reward,
        kl_penalty_type=cfg.kl_penalty_type,
        total_training_steps=cfg.total_training_steps,
        timeline_compare_level="token",
    )

    # Fields not exposed as AgentJetJob kwargs.
    rollout = ajet_job.config.ajet.rollout
    rollout.temperature = cfg.temperature
    rollout.force_disable_toolcalls = cfg.force_disable_toolcalls
    rollout.agent_madness_reward = cfg.agent_madness_reward
    rollout.tensor_model_parallel_size = cfg.tensor_model_parallel_size
    rollout.multi_turn = {
        "max_sample_per_task": cfg.multi_turn_max_sample_per_task,
        "max_steps": cfg.max_steps,
    }

    trainer = ajet_job.config.ajet.trainer_common
    trainer.save_freq = cfg.save_freq
    trainer.test_freq = cfg.test_freq
    trainer.total_epochs = cfg.total_epochs
    trainer.nnodes = cfg.nnodes
    trainer.val_pass_n = cfg.val_pass_n
    trainer.val_before_train = cfg.val_before_train

    ajet_job.config.ajet.debug = {
        "debug_max_parallel": cfg.debug_max_parallel,
        "debug_first_n_tasks": cfg.debug_first_n_tasks,
    }

    return ajet_job


# ---------------- AppWorld task / runner glue ----------------

def _get_appworld_tasks(env_url: str, env_type: str, split: str) -> List[Task]:
    env_client = EnvClient(base_url=env_url)
    task_id_array = env_client.get_env_profile(env_type, split=split)
    if len(task_id_array) == 0:
        raise ValueError(
            f"No task_id found for env_type={env_type}, split={split}, "
            f"check connection to {env_url}"
        )
    return [
        Task(
            main_query="[not defined]",
            init_messages=[],
            task_id=str(task_id),
            env_type=env_type,
            metadata={},
        )
        for task_id in task_id_array
    ]


class ShuffledTaskDataset:
    def __init__(self, tasks: List[Task]):
        self.tasks = list(tasks)

    def generate_training_tasks(self) -> Iterator[Task]:
        pool = list(self.tasks)
        random.shuffle(pool)
        for t in pool:
            yield t


class AppWorldRunner(CocktailSwarmRunner):
    ROLE = "client_0"
    IS_DRIVER = True
    CLIENT_LABEL = "appworld"

    def __init__(self, v2_config: CocktailV2Config):
        super().__init__(v2_config)
        ap = v2_config.appworld
        self.env_url: str = ap.env_url
        self.env_type: str = ap.env_type
        self.training_split: str = ap.training_split
        self.validation_split: str = ap.validation_split
        self.max_steps: int = v2_config.max_steps
        self.EPISODE_TIMEOUT = ap.episode_timeout

    def build_ajet_job(self) -> Optional[AgentJetJob]:
        return build_cocktail_ajet_job(self.config)

    def setup_data(self) -> None:
        train_tasks = _get_appworld_tasks(self.env_url, self.env_type, self.training_split)
        print(f"[INFO] AppWorld training: {len(train_tasks)} tasks (split={self.training_split})")
        self.dataset = ShuffledTaskDataset(train_tasks)

        eval_tasks = _get_appworld_tasks(self.env_url, self.env_type, self.validation_split)
        print(f"[INFO] AppWorld eval: {len(eval_tasks)} tasks (split={self.validation_split})")
        self.eval_tasks_by_set = {self.validation_split: eval_tasks}

    def rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=self.EPISODE_TIMEOUT
        )
        out = self._execute(task, api_baseurl_key)
        self.swarm_worker.end_episode(task, episode_uuid, out)
        return out.reward

    def eval_rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=self.EPISODE_TIMEOUT, episode_type="eval"
        )
        try:
            out = self._execute(task, api_baseurl_key)
            return out.reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)

    def is_success(self, reward: float) -> bool:
        # Mirrors EnvServiceJudge partial-credit shaping: full success requires
        # raw_reward >= 1, which corresponds to final_reward >= 1.0 here.
        return reward >= 1.0

    def _execute(self, task: Task, api_baseurl_key):
        import asyncio
        from tutorial.example_appworld_swarm.appworld_swarm import ExampleAgentScopeWorkflow
        wf = ExampleAgentScopeWorkflow(
            env_url=self.env_url,
            env_type=self.env_type,
            max_steps=self.max_steps,
        )
        return asyncio.run(wf.execute(task, api_baseurl_key))


def main():
    cfg = cocktail_v2_config_from_env()
    runner = AppWorldRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
