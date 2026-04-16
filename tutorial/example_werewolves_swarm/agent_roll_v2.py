# -*- coding: utf-8 -*-
"""
Multi-model Werewolves Training with LoRA Support and Flexible Role Assignment.

This module supports training multiple models with different role assignments,
enabling experiments like:
- Training different models for different role groups (villager/seer vs witch/hunter)
- Training separate models for each special role
- Training werewolf team with multiple independent models
- Random role assignment across models

Usage:
    python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config exp1

Or import and configure programmatically:
    from tutorial.example_werewolves_swarm.agent_roll_v2 import MultiModelWerewolvesTrainer
    trainer = MultiModelWerewolvesTrainer(model_configs=[...])
    trainer.run()
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger

from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_config_schema import AjetTaskReader
from ajet.tuner_lib.experimental.swarm_client import SwarmClient


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL_14B = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct"
DEFAULT_MODEL_7B = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPPONENT_URL = "http://22.17.31.54:2888/v1"
DEFAULT_OPPONENT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

ALL_ROLES = ["werewolf", "villager", "seer", "witch", "hunter"]
GOOD_ROLES = ["villager", "seer", "witch", "hunter"]
BAD_ROLES = ["werewolf"]


class Faction(Enum):
    GOOD = "good"
    BAD = "bad"


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class LoraConfig:
    """LoRA configuration for a model."""
    enabled: bool = True
    rank: int = 32
    alpha: int = 32
    target_modules: str = "all-linear"


@dataclass
class ModelConfig:
    """Configuration for a single trainable model.

    For roles with multiple instances (e.g., werewolf, villager), you can use
    `role_indices` to specify which specific instances this model controls.
    For example:
        - roles=["werewolf"], role_indices={"werewolf": [0]} -> first werewolf only
        - roles=["werewolf"], role_indices={"werewolf": [1, 2]} -> second and third werewolves
        - roles=["villager"], role_indices=None -> all villagers (default behavior)
    """
    model_id: str  # e.g., "M1", "M2"
    swarm_url: str  # e.g., "http://localhost:10086"
    model_path: str = DEFAULT_MODEL_14B
    roles: List[str] = field(default_factory=list)  # e.g., ["hunter", "villager"]
    role_indices: Optional[Dict[str, List[int]]] = None  # e.g., {"werewolf": [0]} for first werewolf
    n_gpu: int = 4
    batch_size: int = 32
    lora: LoraConfig = field(default_factory=LoraConfig)
    experiment_name: str = ""

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"werewolves_{self.model_id.lower()}"
        # Validate roles
        for role in self.roles:
            if role not in ALL_ROLES:
                raise ValueError(f"Invalid role: {role}. Must be one of {ALL_ROLES}")


@dataclass
class ExperimentConfig:
    """Configuration for the entire multi-model experiment."""
    model_configs: List[ModelConfig]
    opponent_model: str = DEFAULT_OPPONENT_MODEL
    opponent_url: str = DEFAULT_OPPONENT_URL
    num_epochs: int = 10000
    grpo_n: int = 6
    max_parallel: int = 64
    discard_episode_timeout: int = 240
    project_name: str = "werewolves_multi_model"
    # Random player split mode: at each episode start, randomly split
    # good-side players among trainable models (ignoring role-based assignment)
    random_player_split: bool = False

    def __post_init__(self):
        # Validate that all trainable roles are from the same faction
        all_trainable_roles = set()
        for mc in self.model_configs:
            all_trainable_roles.update(mc.roles)

        has_good = any(r in GOOD_ROLES for r in all_trainable_roles)
        has_bad = any(r in BAD_ROLES for r in all_trainable_roles)

        if has_good and has_bad:
            raise ValueError(
                "Cannot train good and bad roles simultaneously. "
                f"Found: {all_trainable_roles}"
            )

        self.faction = Faction.BAD if has_bad else Faction.GOOD


# ============================================================================
# Multi-Model Werewolves Game Executor
# ============================================================================

class MultiModelWerewolvesGame:
    """
    Execute werewolves game with multiple trainable models.

    Each model controls a specific set of roles, while non-trainable roles
    are controlled by the opponent model.

    Supports two assignment modes:
    1. Role-based: All instances of a role use the same model (default)
    2. Index-based: Specific role instances assigned to specific models
       (for experiments where each werewolf has its own model)
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        swarm_clients: Dict[str, SwarmClient],
        opponent_model: str,
        opponent_url: str,
        random_player_split: bool = False,
    ):
        self.model_configs = model_configs
        self.swarm_clients = swarm_clients
        self.opponent_model = opponent_model
        self.opponent_url = opponent_url
        self.random_player_split = random_player_split

        # Build role -> model_id mapping (for roles without index constraints)
        self.role_to_model: Dict[str, str] = {}
        # Build (role, index) -> model_id mapping (for indexed assignments)
        self.role_index_to_model: Dict[Tuple[str, int], str] = {}

        if not random_player_split:
            for mc in model_configs:
                for role in mc.roles:
                    if mc.role_indices and role in mc.role_indices:
                        # Index-based assignment
                        for idx in mc.role_indices[role]:
                            self.role_index_to_model[(role, idx)] = mc.model_id
                    else:
                        # Role-based assignment (all instances)
                        self.role_to_model[role] = mc.model_id

    def get_trainable_targets(self) -> List[str]:
        """Get all trainable roles across all models."""
        targets = set()
        for mc in self.model_configs:
            targets.update(mc.roles)
        return list(targets)

    async def execute(
        self,
        task: Task,
        model_api_keys: Dict[str, OpenaiBaseUrlAndApiKey],  # model_id -> api_key
    ):
        """
        Execute the werewolves game with multi-model support.

        Returns dict mapping model_id -> WorkflowOutput.
        """
        import agentscope
        from agentscope.agent import ReActAgent
        from agentscope.formatter import OpenAIMultiAgentFormatter
        from agentscope.model import OpenAIChatModel

        from tutorial.example_werewolves.start import get_official_agent_prompt
        from tutorial.example_werewolves.game import BadGuyException, werewolves_game
        from ajet import WorkflowOutput

        assert agentscope.__version__ == "1.0.7", "Please use AgentScope version 1.0.7"

        trainable_targets = self.get_trainable_targets()

        # Determine faction for reward calculation
        is_training_werewolves = "werewolf" in trainable_targets

        # Make and shuffle roles
        roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
        task_id = task.metadata["random_number"]
        np.random.seed(int(task_id))
        np.random.shuffle(roles)

        # Count occurrences of each role to track indices
        role_counters: Dict[str, int] = {}

        # Track which model each player uses
        player_to_model: Dict[int, str] = {}

        # For random_player_split mode: randomly assign good-side players to models
        player_to_model_split: Dict[int, str] = {}
        if self.random_player_split:
            # Identify all good-side player indices
            good_player_indices = [i for i, role in enumerate(roles) if role in GOOD_ROLES]
            # Shuffle and split 50/50
            np.random.shuffle(good_player_indices)
            half = len(good_player_indices) // 2
            model_ids = [mc.model_id for mc in self.model_configs]
            for i, player_idx in enumerate(good_player_indices):
                # First half -> M1, second half -> M2
                model_id_for_player = model_ids[0] if i < half else model_ids[1]
                player_to_model_split[player_idx] = model_id_for_player
            logger.info(f"Random player split: M1={[p for p, m in player_to_model_split.items() if m == model_ids[0]]}, "
                       f"M2={[p for p, m in player_to_model_split.items() if m == model_ids[1]]}")

        # Initialize agents
        players = []
        for i, role in enumerate(roles):
            # Get the index of this role instance (0, 1, 2 for werewolves, etc.)
            role_idx = role_counters.get(role, 0)
            role_counters[role] = role_idx + 1

            # Determine model_id based on assignment mode
            if self.random_player_split:
                # In random split mode, use player-based assignment
                model_id = player_to_model_split.get(i)
            else:
                # Try to find model: first by (role, index), then by role only
                model_id = self.role_index_to_model.get((role, role_idx))
                if model_id is None:
                    model_id = self.role_to_model.get(role)

            if model_id is None:
                # Non-trainable role - use opponent model
                model_for_agent = OpenAIChatModel(
                    stream=False,
                    api_key="no_api_key",
                    generate_kwargs={"temperature": 0.01},
                    model_name=self.opponent_model,
                    client_args={"base_url": self.opponent_url},
                )
            else:
                # Trainable role - use corresponding swarm model
                api_key = model_api_keys[model_id]
                model_for_agent = OpenAIChatModel(
                    stream=False,
                    api_key=api_key.api_key,
                    generate_kwargs={"temperature": 0.7},
                    model_name="default",
                    client_args={"base_url": api_key.base_url},
                )
                player_to_model[i] = model_id

            agent = ReActAgent(
                name=f"Player{i + 1}",
                sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
                model=model_for_agent,
                formatter=OpenAIMultiAgentFormatter(),
                max_iters=3 if role in trainable_targets else 5,
            )
            agent.set_console_output_enabled(False)
            players.append(agent)

        # Execute game
        try:
            good_guy_win = await werewolves_game(players, roles)

            # Calculate reward
            if is_training_werewolves:
                raw_reward = 1.0 if not good_guy_win else 0.0
            else:
                raw_reward = 1.0 if good_guy_win else 0.0

            is_success = raw_reward > 0
            logger.info(f"Game finished - Winner: {'Good' if good_guy_win else 'Bad'}, "
                       f"Training {'werewolves' if is_training_werewolves else 'good guys'}, "
                       f"Reward: {raw_reward}")

        except (BadGuyException, Exception) as e:
            logger.exception(f"Error during game: {e}")
            raw_reward = -0.1
            is_success = False

        # Create WorkflowOutput for each model
        results = {}
        for mc in self.model_configs:
            results[mc.model_id] = WorkflowOutput(reward=raw_reward, is_success=is_success)

        return results


# ============================================================================
# Multi-Model Trainer
# ============================================================================

class MultiModelWerewolvesTrainer:
    """
    Trainer for multi-model werewolves experiments.

    Supports training multiple models with LoRA, each controlling
    different roles in the game.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.swarm_clients: Dict[str, SwarmClient] = {}
        self.jobs: Dict[str, AgentJetJob] = {}

    def setup(self):
        """Initialize all swarm clients and jobs."""
        logger.info(f"Setting up {len(self.config.model_configs)} models...")

        jobs_to_start = []

        for mc in self.config.model_configs:
            # Create AgentJetJob with LoRA params passed directly
            job = AgentJetJob(
                base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
                algorithm="grpo",
                project_name=self.config.project_name,
                experiment_name=mc.experiment_name,
                n_gpu=mc.n_gpu,
                model=mc.model_path,
                batch_size=mc.batch_size,
                num_repeat=self.config.grpo_n,
                max_env_worker=128,
                # LoRA parameters - pass directly to AgentJetJob
                lora_rank=mc.lora.rank if mc.lora.enabled else None,
                lora_alpha=mc.lora.alpha if mc.lora.enabled else None,
                lora_target_modules=mc.lora.target_modules if mc.lora.enabled else None,
                lr=3e-4,
                layered_summon=True,
            )

            self.jobs[mc.model_id] = job

            # Create SwarmClient
            client = SwarmClient(mc.swarm_url)
            self.swarm_clients[mc.model_id] = client
            jobs_to_start.append((client, job))

            logger.info(f"Model {mc.model_id}: roles={mc.roles}, "
                       f"lora={'enabled' if mc.lora.enabled else 'disabled'}, "
                       f"swarm={mc.swarm_url}")

        # Start all engines concurrently
        logger.info("Starting all swarm engines...")
        SwarmClient.async_and_start_multi_engine(jobs_to_start, force_restart=True)
        logger.info("All engines started successfully!")

    def run(self):
        """Run the training loop."""
        self.setup()

        dataset = RouterTaskReader(
            reader_type="random_dummy",
            reader_config=AjetTaskReader()
        )

        game_executor = MultiModelWerewolvesGame(
            model_configs=self.config.model_configs,
            swarm_clients=self.swarm_clients,
            opponent_model=self.config.opponent_model,
            opponent_url=self.config.opponent_url,
            random_player_split=self.config.random_player_split,
        )

        def rollout(task: Task):
            """Execute one rollout episode across all models."""
            # Begin episodes for all models
            episodes: Dict[str, Tuple[str, OpenaiBaseUrlAndApiKey]] = {}
            model_api_keys: Dict[str, OpenaiBaseUrlAndApiKey] = {}

            for model_id, client in self.swarm_clients.items():
                episode_uuid, api_key = client.begin_episode(
                    discard_episode_timeout=self.config.discard_episode_timeout
                )
                episodes[model_id] = (episode_uuid, api_key)
                model_api_keys[model_id] = api_key

            # Execute game
            results = asyncio.run(game_executor.execute(task, model_api_keys))

            # End episodes for all models
            for model_id, client in self.swarm_clients.items():
                episode_uuid, _ = episodes[model_id]
                client.end_episode(task, episode_uuid, results[model_id])
                client.print_rollout_stat()

        # Calculate total workers needed
        total_batch_size = max(mc.batch_size for mc in self.config.model_configs)
        total_workers = total_batch_size * self.config.grpo_n

        executor = PeriodicDrainThreadPoolExecutor(
            workers=total_workers,
            max_parallel=self.config.max_parallel,
            auto_retry=True
        )

        logger.info(f"Starting training loop: {self.config.num_epochs} epochs, "
                   f"{total_workers} workers, {self.config.max_parallel} parallel")

        for epoch in range(self.config.num_epochs):
            for _, task in enumerate(dataset.generate_training_tasks()):
                for _ in range(self.config.grpo_n):
                    executor.submit_with_periodic_drain(fn=rollout, task=task)


# ============================================================================
# Predefined Experiment Configurations
# ============================================================================

VERSION = "v3"


def get_exp1_config() -> ExperimentConfig:
    """
    Experiment 1: Two models for good guys.
    - M1 (14B-LoRA): hunter, villager
    - M2 (14B-LoRA): witch, seer
    - Opponents (235B): werewolf
    """
    return ExperimentConfig(
        model_configs=[
            ModelConfig(
                model_id="M1",
                swarm_url="http://localhost:10086",
                model_path=DEFAULT_MODEL_14B,
                roles=["hunter", "villager"],
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp1_m1_ht_vl_{VERSION}",
            ),
            ModelConfig(
                model_id="M2",
                swarm_url="http://localhost:10087",
                model_path=DEFAULT_MODEL_14B,
                roles=["witch", "seer"],
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp1_m2_wt_sr_{VERSION}",
            ),
        ],
        project_name="werewolves_exp1_two_model_good",
    )


def get_exp2_config() -> ExperimentConfig:
    """
    Experiment 2: Three models for good guys.
    - M1 (14B-LoRA, 3 GPUs): villager
    - M2 (14B-LoRA, 3 GPUs): seer, witch
    - M3 (14B-LoRA, 2 GPUs): hunter
    - Opponents (235B): werewolf
    """
    return ExperimentConfig(
        model_configs=[
            ModelConfig(
                model_id="M1",
                swarm_url="http://localhost:10086",
                model_path=DEFAULT_MODEL_14B,
                roles=["villager"],
                n_gpu=3,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp2_m1_vl_{VERSION}",
            ),
            ModelConfig(
                model_id="M2",
                swarm_url="http://localhost:10087",
                model_path=DEFAULT_MODEL_14B,
                roles=["seer", "witch"],
                n_gpu=3,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp2_m2_sr_wt_{VERSION}",
            ),
            ModelConfig(
                model_id="M3",
                swarm_url="http://localhost:10088",
                model_path=DEFAULT_MODEL_14B,
                roles=["hunter"],
                n_gpu=2,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp2_m3_ht_{VERSION}",
            ),
        ],
        project_name="werewolves_exp2_three_model_good",
    )


def get_exp3_config() -> ExperimentConfig:
    """
    Experiment 3: Three models for werewolves.
    - M1 (14B-LoRA, 3 GPUs): werewolf 1 (index 0)
    - M2 (14B-LoRA, 3 GPUs): werewolf 2 (index 1)
    - M3 (14B-LoRA, 2 GPUs): werewolf 3 (index 2)
    - Opponents (235B): villager, seer, witch, hunter

    Uses role_indices to assign each werewolf instance to a different model.
    """
    return ExperimentConfig(
        model_configs=[
            ModelConfig(
                model_id="M1",
                swarm_url="http://localhost:10086",
                model_path=DEFAULT_MODEL_14B,
                roles=["werewolf"],
                role_indices={"werewolf": [0]},  # First werewolf only
                n_gpu=3,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp3_m1_ww1_{VERSION}",
            ),
            ModelConfig(
                model_id="M2",
                swarm_url="http://localhost:10087",
                model_path=DEFAULT_MODEL_14B,
                roles=["werewolf"],
                role_indices={"werewolf": [1]},  # Second werewolf only
                n_gpu=3,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp3_m2_ww2_{VERSION}",
            ),
            ModelConfig(
                model_id="M3",
                swarm_url="http://localhost:10088",
                model_path=DEFAULT_MODEL_14B,
                roles=["werewolf"],
                role_indices={"werewolf": [2]},  # Third werewolf only
                n_gpu=2,
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp3_m3_ww3_{VERSION}",
            ),
        ],
        project_name="werewolves_exp3_three_model_ww",
    )


def get_exp4_config() -> ExperimentConfig:
    """
    Experiment 4: Two models with random 50/50 split of good-side players.
    - M1 (14B-LoRA): randomly selected 50% of good-side players per episode
    - M2 (14B-LoRA): remaining 50% of good-side players
    - Opponents (235B): werewolf

    At the start of each episode, the 6 good-side players (3 villagers,
    1 seer, 1 witch, 1 hunter) are randomly split: 3 players go to M1,
    3 players go to M2. This is player-based, not role-based assignment.
    """
    return ExperimentConfig(
        model_configs=[
            ModelConfig(
                model_id="M1",
                swarm_url="http://localhost:10086",
                model_path=DEFAULT_MODEL_14B,
                roles=GOOD_ROLES,  # All good roles (for validation only)
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp4_m1_half_{VERSION}",
            ),
            ModelConfig(
                model_id="M2",
                swarm_url="http://localhost:10087",
                model_path=DEFAULT_MODEL_14B,
                roles=GOOD_ROLES,  # All good roles (for validation only)
                lora=LoraConfig(enabled=True, rank=32, alpha=32),
                experiment_name=f"werewolves_exp4_m2_half_{VERSION}",
            ),
        ],
        project_name="werewolves_exp4_random_split",
        random_player_split=True,  # Enable random player-based assignment
    )


# ============================================================================
# Main Entry Point
# ============================================================================

EXPERIMENT_CONFIGS = {
    "exp1": get_exp1_config,
    "exp2": get_exp2_config,
    "exp3": get_exp3_config,
    "exp4": get_exp4_config,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-model Werewolves Training")
    parser.add_argument(
        "--config",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        default="exp1",
        help="Experiment configuration to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    args = parser.parse_args()

    config = EXPERIMENT_CONFIGS[args.config]()
    if args.epochs is not None:
        config.num_epochs = args.epochs

    logger.info(f"Running experiment: {args.config}")
    logger.info(f"Faction: {config.faction.value}")
    logger.info(f"Models: {len(config.model_configs)}")

    trainer = MultiModelWerewolvesTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
