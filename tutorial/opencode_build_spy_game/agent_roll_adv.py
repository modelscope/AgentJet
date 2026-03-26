"""
Swarm client for adversarial training - agent_roll_adv mode.
Team A (civilians): 7B model from swarm server 1
Team B (spies): 7B model from swarm server 2
Both teams train simultaneously in competitive setting.
"""

import os
import json
from pathlib import Path
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.schema.task import Task
from tutorial.opencode_build_spy_game.agent_run_adv import run_agent_and_compute_reward


# Local configurations (client-side)
LOCAL_GRPO_N = 4  # GRPO group size (rollout.n)
LOCAL_NUM_EPOCH = 100
LOCAL_MAX_PARALLEL = 16
LOCAL_DATASET_PATH = str(Path(__file__).parent / "mock_game_dataset.json")

# Remote configurations for swarm server 1 (civilian team)
REMOTE_SWARM_URL_1 = os.getenv("AJET_SWARM_URL_1", "http://localhost:10086")
REMOTE_BATCH_SIZE_1 = 16
REMOTE_ALLOCATE_GPU_1 = 4
REMOTE_TRAIN_MODEL_1 = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

# Remote configurations for swarm server 2 (spy team)
REMOTE_SWARM_URL_2 = os.getenv("AJET_SWARM_URL_2", "http://localhost:10087")
REMOTE_BATCH_SIZE_2 = 16
REMOTE_ALLOCATE_GPU_2 = 4
REMOTE_TRAIN_MODEL_2 = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'


class SpyGameDatasetReader:
    """Custom dataset reader for spy game configurations."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def generate_training_tasks(self):
        """Generate training tasks from dataset."""
        for idx, item in enumerate(self.data):
            yield Task(
                main_query=f"Play adversarial spy game episode {idx}",
                metadata={
                    "civilian_word": item["civilian_word"],
                    "spy_word": item["spy_word"],
                    "num_players": item["num_players"],
                    "num_spies": item["num_spies"],
                    "episode_id": idx
                }
            )


def main():
    """Main adversarial training loop."""

    # Load dataset
    print(f"Loading dataset from: {LOCAL_DATASET_PATH}")
    dataset_reader = SpyGameDatasetReader(LOCAL_DATASET_PATH)

    # Connect to swarm server 1 (civilian team)
    print(f"Connecting to swarm server 1 (civilians): {REMOTE_SWARM_URL_1}")
    swarm_worker_1 = SwarmClient(REMOTE_SWARM_URL_1)

    ajet_job_1 = AgentJetJob(
        algorithm="grpo",
        project_name="spy-game-rl-adv",
        experiment_name="civilians_team_7b",
        n_gpu=REMOTE_ALLOCATE_GPU_1,
        model=REMOTE_TRAIN_MODEL_1,
        batch_size=REMOTE_BATCH_SIZE_1,
        num_repeat=LOCAL_GRPO_N,
    )

    print("Starting swarm engine 1 (civilians)...")
    swarm_worker_1.auto_sync_train_config_and_start_engine(ajet_job_1)

    # Connect to swarm server 2 (spy team)
    print(f"Connecting to swarm server 2 (spies): {REMOTE_SWARM_URL_2}")
    swarm_worker_2 = SwarmClient(REMOTE_SWARM_URL_2)

    ajet_job_2 = AgentJetJob(
        algorithm="grpo",
        project_name="spy-game-rl-adv",
        experiment_name="spies_team_7b",
        n_gpu=REMOTE_ALLOCATE_GPU_2,
        model=REMOTE_TRAIN_MODEL_2,
        batch_size=REMOTE_BATCH_SIZE_2,
        num_repeat=LOCAL_GRPO_N,
    )

    print("Starting swarm engine 2 (spies)...")
    swarm_worker_2.auto_sync_train_config_and_start_engine(ajet_job_2)

    def rollout(task: Task):
        """Execute one adversarial episode rollout."""
        try:
            # Begin episode for both teams
            episode_uuid_1, api_baseurl_key_1 = swarm_worker_1.begin_episode(discard_episode_timeout=300)
            episode_uuid_2, api_baseurl_key_2 = swarm_worker_2.begin_episode(discard_episode_timeout=300)

            # Execute adversarial agent workflow
            workflow_output_civilians, workflow_output_spies = run_agent_and_compute_reward(
                task=task,
                base_url_civilians=api_baseurl_key_1.base_url,
                api_key_civilians=api_baseurl_key_1.api_key,
                base_url_spies=api_baseurl_key_2.base_url,
                api_key_spies=api_baseurl_key_2.api_key
            )

            # Report results back to both swarm servers
            swarm_worker_1.end_episode(task, episode_uuid_1, workflow_output_civilians)
            swarm_worker_2.end_episode(task, episode_uuid_2, workflow_output_spies)

            # Print status
            winner = workflow_output_civilians.metadata.get('winner', '?')
            print(f"Episode {task.metadata.get('episode_id', '?')}: "
                  f"Winner={winner}, "
                  f"Civilian_Reward={workflow_output_civilians.reward:.2f}, "
                  f"Spy_Reward={workflow_output_spies.reward:.2f}")

            # Print rollout statistics
            print("Civilian team stats:")
            swarm_worker_1.print_rollout_stat()
            print("Spy team stats:")
            swarm_worker_2.print_rollout_stat()

            # Return average reward for logging
            civilian_reward = workflow_output_civilians.reward
            spy_reward = workflow_output_spies.reward
            if civilian_reward is None or spy_reward is None:
                return None
            if not isinstance(civilian_reward, float) or not isinstance(spy_reward, float):
                return None
            return (civilian_reward + spy_reward) / 2.0

        except Exception as e:
            print(f"Error in adversarial rollout: {e}")
            return None

    # Training loop
    print(f"\nStarting adversarial training for {LOCAL_NUM_EPOCH} epochs...")

    for epoch in range(LOCAL_NUM_EPOCH):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{LOCAL_NUM_EPOCH}")
        print(f"{'='*60}")

        next_batch = []
        for task in dataset_reader.generate_training_tasks():
            # For each task, add it LOCAL_GRPO_N times to the batch
            # These are multiple rollouts of the SAME task for GRPO
            for _ in range(LOCAL_GRPO_N):
                next_batch.append(task)

            # When we have enough tasks in batch, execute them
            if len(next_batch) >= (REMOTE_BATCH_SIZE_1 * LOCAL_GRPO_N):
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
                          f"({num_tasks} tasks x {LOCAL_GRPO_N} episodes), Avg combined reward: {avg_reward:.3f}")

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
                      f"({num_tasks} tasks x {LOCAL_GRPO_N} episodes), Avg combined reward: {avg_reward:.3f}")

    print("\n" + "="*60)
    print("Adversarial training completed!")
    print("="*60)

    # Optionally stop the engines (commented out to keep them running)
    # swarm_worker_1.stop_engine()
    # swarm_worker_2.stop_engine()


if __name__ == "__main__":
    main()
