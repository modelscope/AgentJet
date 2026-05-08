# -*- coding: utf-8 -*-
"""
AIME swarm client (follower) for example_cocktail_rl_v2.

python -m tutorial.example_cocktail_rl_v2.train_aime_as_swarm_client_1
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.schema.task import Task
from ajet.task_reader import HuggingFaceTaskReader, RouterTaskReader

from tutorial.example_cocktail_rl_v2.cocktail_v2_config import (
    CocktailV2Config,
    cocktail_v2_config_from_env,
)
from tutorial.example_cocktail_rl_v2.cocktail_v2_runner import CocktailSwarmRunner
from tutorial.opencode_build_aime import download_data
from tutorial.opencode_build_aime.agent_run_v3 import execute_agent as _execute_aime_agent


_THIS_DIR = os.path.dirname(__file__)


@dataclass
class _AimeAgentConfig:
    """Duck-types the subset of AgentJetJob that execute_agent reads."""
    model: str
    max_response_length: int


def _load_eval_tasks(test_dataset: str, label: str = "") -> List[Task]:
    eval_tasks: List[Task] = []
    if not os.path.exists(test_dataset):
        print(f"[WARN] Eval dataset not found: {test_dataset}. Skipping {label or test_dataset}.")
        return eval_tasks

    eval_reader = HuggingFaceTaskReader(
        AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=test_dataset))
    )
    for t in eval_reader.generate_training_tasks():
        eval_tasks.append(t)
    print(f"[INFO] Loaded {len(eval_tasks)} eval tasks from {label or test_dataset}")
    return eval_tasks




class AimeRunner(CocktailSwarmRunner):
    ROLE = "client_1"
    IS_DRIVER = False
    CLIENT_LABEL = "aime"

    def __init__(self, v2_config: CocktailV2Config):
        super().__init__(v2_config)
        am = v2_config.aime
        self.EPISODE_TIMEOUT = am.episode_timeout
        self.agent_config = _AimeAgentConfig(
            model="dummy",
            max_response_length=v2_config.max_response_length,
        )

        data_dir = os.path.join(_THIS_DIR, "..", "opencode_build_aime", "data")
        self.train_dataset = os.path.join(data_dir, am.train_dataset_filename)
        self.test_datasets = {
            label: os.path.join(data_dir, fname)
            for label, fname in am.test_dataset_filenames.items()
        }

    def setup_data(self) -> None:
        if not os.path.exists(self.train_dataset):
            raise FileNotFoundError(
                f"AIME training dataset missing: {self.train_dataset}\n"
                "Please run: proxychains python -m tutorial.opencode_build_aime.download_data"
            )

        train_reader = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=self.train_dataset)
            ),
        )
        self.dataset = train_reader

        eval_downloaders = {
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
            tasks = _load_eval_tasks(path, label=label)
            if tasks:
                self.eval_tasks_by_set[label] = tasks

    def rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=self.EPISODE_TIMEOUT
        )
        out = _execute_aime_agent(task, api_baseurl_key, self.agent_config)
        self.swarm_worker.end_episode(task, episode_uuid, out)
        return out.reward

    def eval_rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=self.EPISODE_TIMEOUT, episode_type="eval"
        )
        try:
            out = _execute_aime_agent(task, api_baseurl_key, self.agent_config)
            return out.reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)

    def is_success(self, reward: float) -> bool:
        return reward > 0


def main():
    cfg = cocktail_v2_config_from_env()
    runner = AimeRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
