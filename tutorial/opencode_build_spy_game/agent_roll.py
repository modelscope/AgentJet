"""
Swarm client for training spy game agent - agent_roll mode.
Civilians (7B model) vs Spies (qwen-max)
"""

import os
import json
import uuid
from pathlib import Path
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.default_config.ajet_default import AjetTaskReader
from ajet.task_reader import RouterTaskReader
from ajet.schema.task import Task
from tutorial.opencode_build_spy_game.agent_run import run_agent_and_compute_reward


# Local configurations (client-side)
LOCAL_GRPO_N = 6  # GRPO group size (rollout.n)
LOCAL_NUM_EPOCH = 100
LOCAL_MAX_PARALLEL = 32
LOCAL_DATASET_PATH = str(Path(__file__).parent / "mock_game_dataset.json")

# Remote configurations (swarm server)
REMOTE_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_BATCH_SIZE = 32  # Small batch size to fit in memory
REMOTE_ALLOCATE_GPU = 8  # Use only 2 GPUs to avoid OOM
REMOTE_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'


class SpyGameDatasetReader:
    """Custom dataset reader for spy game configurations."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def generate_training_tasks(self):
        """Generate training tasks from dataset."""
        for idx, item in enumerate(self.data):
            # Each task needs a unique task_id - use a deterministic ID based on index
            task_id = f"spy_game_task_{idx:04d}"
            yield Task(
                task_id=task_id,  # Required: explicit task_id
                main_query=f"Play spy game episode {idx}",
                metadata={
                    "civilian_word": item["civilian_word"],
                    "spy_word": item["spy_word"],
                    "num_players": item["num_players"],
                    "num_spies": item["num_spies"],
                    "episode_id": idx
                }
            )


def main():
    """Main training loop."""

    # Load dataset
    print(f"Loading dataset from: {LOCAL_DATASET_PATH}")
    dataset_reader = SpyGameDatasetReader(LOCAL_DATASET_PATH)

    # Connect to swarm server
    print(f"Connecting to swarm server: {REMOTE_SWARM_URL}")
    swarm_worker = SwarmClient(REMOTE_SWARM_URL)

    # Configure and start training
    ajet_job = AgentJetJob(
        algorithm="grpo",
        project_name="spy-game-rl",
        logging="swanlab",
        experiment_name="agent_roll_7b_vs_qwen_max",
        n_gpu=REMOTE_ALLOCATE_GPU,
        model=REMOTE_TRAIN_MODEL,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
    )

    print("Starting swarm engine...")
    swarm_worker.auto_sync_train_config_and_start_engine(ajet_job)

    def rollout(task: Task):
        """Execute one episode rollout."""
        try:
            # Begin episode
            episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=300)

            # Execute agent workflow
            workflow_output = run_agent_and_compute_reward(
                task=task,
                base_url=api_baseurl_key.base_url,
                api_key=api_baseurl_key.api_key
            )

            # Report result back to swarm server
            swarm_worker.end_episode(task, episode_uuid, workflow_output)

            # Print status
            print(f"Episode {task.metadata.get('episode_id', '?')}: "
                  f"Winner={workflow_output.metadata.get('winner', '?')}, "
                  f"Reward={workflow_output.reward:.2f}")

            swarm_worker.print_rollout_stat()

            return workflow_output.reward

        except Exception as e:
            print(f"Error in rollout: {e}")
            return None

    # Training loop
    print(f"\nStarting training for {LOCAL_NUM_EPOCH} epochs...")

    for epoch in range(LOCAL_NUM_EPOCH):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{LOCAL_NUM_EPOCH}")
        print(f"{'='*60}")

        next_batch = []
        task_count = 0
        for task in dataset_reader.generate_training_tasks():
            task_count += 1
            # For each task, add it LOCAL_GRPO_N times to the batch
            # These are multiple rollouts of the SAME task for GRPO
            for _ in range(LOCAL_GRPO_N):
                next_batch.append(task)

            # Debug logging
            if task_count <= 5:
                print(f"[DEBUG] Added task {task_count} (episode_id={task.metadata.get('episode_id')}), batch size now: {len(next_batch)}")

            # When we have enough tasks in batch, execute them
            if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                # Execute batch with retry logic
                episode_results = run_episodes_until_all_complete(
                    next_batch,
                    func=rollout,
                    auto_retry=True
                )

                # Print batch statistics
                valid_results = [r for r in episode_results if r is not None]
                if valid_results:
                    avg_reward = sum(valid_results) / len(valid_results)
                    num_tasks = len(next_batch) // LOCAL_GRPO_N
                    print(f"\nBatch completed: {len(valid_results)}/{len(next_batch)} episodes "
                          f"({num_tasks} tasks x {LOCAL_GRPO_N} episodes), Avg reward: {avg_reward:.3f}")

                next_batch.clear()

        # Process any remaining tasks in the batch at end of epoch
        if len(next_batch) > 0:
            episode_results = run_episodes_until_all_complete(
                next_batch,
                func=rollout,
                auto_retry=True
            )
            valid_results = [r for r in episode_results if r is not None]
            if valid_results:
                avg_reward = sum(valid_results) / len(valid_results)
                num_tasks = len(next_batch) // LOCAL_GRPO_N
                print(f"\nFinal batch completed: {len(valid_results)}/{len(next_batch)} episodes "
                      f"({num_tasks} tasks x {LOCAL_GRPO_N} episodes), Avg reward: {avg_reward:.3f}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    # Optionally stop the engine (commented out to keep it running)
    # swarm_worker.stop_engine()


if __name__ == "__main__":
    main()
