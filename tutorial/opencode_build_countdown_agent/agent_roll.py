# ------- AI GENERATED --------
# ------- [Read tutorial/opencode_build_countdown_agent.prompt.md] --------

"""
CountDown Agent Training Script (Swarm Client)

This script connects to the AgentJet Swarm server and trains the countdown agent.

Usage:
    python -m tutorial.opencode_build_countdown_agent.agent_roll

Before running:
    1. Start the swarm server: ajet-swarm start
    2. Ensure the dataset is generated: python tutorial/opencode_build_countdown_agent/generate_countdown_dataset.py
    3. Update the configuration variables below to match your setup
"""

from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import (
    SwarmClient,
    run_episodes_until_all_complete,
)
from ajet.default_config.ajet_config_schema import (
    AjetTaskReader,
    JsonlDatasetFile,
    JsonlTrainingFp,
)
from ajet.task_reader import RouterTaskReader
from .agent_run import run_agent_and_compute_reward


# --------- Configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # GRPO group size (number of rollouts per task)
LOCAL_NUM_EPOCH = 100  # Number of training epochs
LOCAL_DATASET_PATH = "./tutorial/opencode_build_countdown_agent/countdown_dataset/train.jsonl"
REMOTE_SWARM_URL = "http://localhost:10086"  # Swarm server URL

# --------- Configurations that take effect remotely (on swarm server) -------------
REMOTE_BATCH_SIZE = 16  # Batch size for training (as specified by user)
REMOTE_ALLOCATE_GPU_PER_NODE = 8  # Number of GPUs to use (as specified by user)
REMOTE_TRAIN_MODEL = (
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct"
)


def main():
    """
    Main training loop for CountDown agent.
    """

    # Load the CountDown dataset
    print(f"Loading dataset from: {LOCAL_DATASET_PATH}")
    dataset = RouterTaskReader(
        reader_type="jsonl_dataset_file",
        reader_config=AjetTaskReader(
            jsonl_dataset_file=JsonlDatasetFile(
                training=JsonlTrainingFp(file_path=LOCAL_DATASET_PATH)
            )
        ),
    )

    # Connect to swarm server and configure training
    print(f"Connecting to swarm server at: {REMOTE_SWARM_URL}")
    swarm_worker = SwarmClient(REMOTE_SWARM_URL)

    # Configure and start the training engine
    print("Configuring training parameters...")
    yaml_job = AgentJetJob(
        algorithm="grpo",  # Using GRPO (Group Relative Policy Optimization)
        project_name="countdown-agent",
        experiment_name="countdown_solver_7b",
        n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_TRAIN_MODEL,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
    )

    print("Starting swarm engine...")
    swarm_worker.auto_sync_train_config_and_start_engine(yaml_job)

    print("\n" + "=" * 80)
    print("Training started!")
    print(f"Model: {REMOTE_TRAIN_MODEL}")
    print(f"GPUs: {REMOTE_ALLOCATE_GPU_PER_NODE}")
    print(f"Batch size: {REMOTE_BATCH_SIZE}")
    print(f"GRPO group size: {LOCAL_GRPO_N}")
    print(f"Epochs: {LOCAL_NUM_EPOCH}")
    print("=" * 80 + "\n")

    def rollout(task):
        """
        Execute a single episode (rollout) of the agent.

        Args:
            task: The countdown problem to solve

        Returns:
            The reward obtained (or None on failure)
        """
        try:
            # Begin episode and get API credentials
            episode_uuid, api_baseurl_key = swarm_worker.begin_episode()

            # Execute agent and compute reward
            workflow_output = run_agent_and_compute_reward(
                task, api_baseurl_key.base_url, api_baseurl_key.api_key
            )

            # Report results back to swarm server
            swarm_worker.end_episode(task, episode_uuid, workflow_output)

            # Print rollout statistics
            swarm_worker.print_rollout_stat()

            return workflow_output.reward

        except Exception as e:
            print(f"Error during rollout: {e}")
            return None

    # Training loop
    next_batch = []
    total_episodes = 0

    for epoch in range(LOCAL_NUM_EPOCH):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{LOCAL_NUM_EPOCH}")
        print(f"{'=' * 80}\n")

        for task_idx, task in enumerate(dataset.generate_training_tasks()):
            # For each task, perform LOCAL_GRPO_N rollouts (GRPO group)
            for _ in range(LOCAL_GRPO_N):
                next_batch.append(task)

                # When batch is full, execute all episodes
                if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                    print(f"\nExecuting batch of {len(next_batch)} episodes...")

                    # Execute episodes with automatic retry on failure
                    episode_results = run_episodes_until_all_complete(
                        next_batch, func=rollout, auto_retry=True
                    )

                    total_episodes += len(next_batch)

                    # Print batch results
                    successful = sum(
                        1 for r in episode_results if r is not None and r > 0
                    )
                    avg_reward = (
                        sum(r for r in episode_results if r is not None)
                        / len(episode_results)
                        if episode_results
                        else 0
                    )

                    print(f"\nBatch completed:")
                    print(f"  Total episodes: {len(next_batch)}")
                    print(f"  Successful: {successful}")
                    print(f"  Average reward: {avg_reward:.3f}")
                    print(f"  Total episodes so far: {total_episodes}")

                    next_batch.clear()

        print(f"\nEpoch {epoch + 1} completed!")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Total episodes executed: {total_episodes}")
    print("=" * 80)

    return None


if __name__ == "__main__":
    main()
