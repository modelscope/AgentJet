# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Agent Rollout Script

This script trains a language model on the DAPO-Math-17k dataset
using AgentJet Swarm for distributed RL training.

Usage:
    # First, start the swarm server:
    ajet-swarm start

    # Then run this script:
    python -m tutorial.opencode_build_aime.agent_roll

Environment Variables:
    AJET_SWARM_URL: Swarm server URL (default: http://localhost:10086)
    REMOTE_MODEL_PATH: Path to the model to train
    REMOTE_BATCH_SIZE: Training batch size (default: 32)
    REMOTE_ALLOCATE_GPU_PER_NODE: Number of GPUs per node (default: 8)
"""

import os
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.task_reader import RouterTaskReader
from tutorial.opencode_build_aime.agent_run import execute_agent


# ==================== Local Configurations ====================
# These settings control the client-side behavior

LOCAL_GRPO_N = 4  # GRPO group size (number of rollouts per task)
LOCAL_NUM_EPOCH = 10000  # Number of training epochs

# Dataset paths - will be created by download_data.py
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCAL_TRAIN_DATASET = os.path.join(LOCAL_DATA_DIR, "dapo-math-17k.parquet")
LOCAL_TEST_DATASET = os.path.join(LOCAL_DATA_DIR, "aime-2024.parquet")

# Swarm server URL
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

# ==================== Remote Configurations ====================
# These settings are sent to the swarm server

REMOTE_BATCH_SIZE = int(os.getenv("REMOTE_BATCH_SIZE", "32"))
REMOTE_ALLOCATE_GPU_PER_NODE = int(os.getenv("REMOTE_ALLOCATE_GPU_PER_NODE", "8"))
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct"
)

# Validate swarm URL
assert AJET_SWARM_URL != "http://swarm-server-ip:10086", \
    "Please set AJET_SWARM_URL to your swarm server's URL"


def main():
    """Main training loop."""

    # Check dataset exists
    if not os.path.exists(LOCAL_TRAIN_DATASET):
        print(f"[ERROR] Training dataset not found: {LOCAL_TRAIN_DATASET}")
        print("Please run: proxychains python -m tutorial.opencode_build_aime.download_data")
        return

    print("=" * 70)
    print("AIME Math Swarm Training")
    print("=" * 70)
    print(f"  Swarm URL:    {AJET_SWARM_URL}")
    print(f"  Model:        {REMOTE_MODEL_PATH}")
    print(f"  Dataset:      {LOCAL_TRAIN_DATASET}")
    print(f"  Batch Size:   {REMOTE_BATCH_SIZE}")
    print(f"  GRPO N:       {LOCAL_GRPO_N}")
    print(f"  GPUs/Node:    {REMOTE_ALLOCATE_GPU_PER_NODE}")
    print("=" * 70)

    # Initialize dataset reader
    dataset = RouterTaskReader(
        reader_type="huggingface_dat_repo",
        reader_config=AjetTaskReader(
            huggingface_dat_repo=HuggingfaceDatRepo(
                dataset_path=LOCAL_TRAIN_DATASET
            )
        )
    )

    # Connect to swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)

    # Configure and start the training engine
    ajet_job = AgentJetJob(
        algorithm="grpo",
        project_name="ajet-aime",
        experiment_name="dapo-math-17k-grpo",
        n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_MODEL_PATH,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
    )

    # Print configuration for verification
    print("\n[INFO] Training configuration:")
    print(ajet_job.config.to_dict())

    # Sync configuration and start engine
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        force_restart=True,
    )

    def rollout(task) -> float | None:
        """Execute a single episode rollout."""
        # Begin episode - get API credentials from swarm server
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=300  # 5 minutes timeout for long reasoning
        )

        # Execute agent and compute reward
        workflow_output = execute_agent(task, api_baseurl_key)

        # Report output back to swarm server
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

        # Print progress
        swarm_worker.print_rollout_stat()

        return workflow_output.reward

    # Main training loop
    print("\n[INFO] Starting training loop...")
    next_batch = []

    for epoch in range(LOCAL_NUM_EPOCH):
        print(f"\n[EPOCH {epoch + 1}/{LOCAL_NUM_EPOCH}]")

        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(LOCAL_GRPO_N):
                next_batch.append(task)

                # When batch is full, execute all episodes
                if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                    print(f"\n[BATCH] Executing {len(next_batch)} episodes...")
                    episode_results = run_episodes_until_all_complete(
                        next_batch,
                        func=rollout,
                        auto_retry=True
                    )

                    # Print batch statistics
                    rewards = [r for r in episode_results if r is not None]
                    if rewards:
                        avg_reward = sum(rewards) / len(rewards)
                        print(f"[BATCH COMPLETE] Avg reward: {avg_reward:.4f}")

                    next_batch.clear()

    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()
