"""
AppWorld React Agent Training Script

This script sets up the training loop for the AppWorld React agent using AgentJet Swarm.

Usage:
    python -m tutorial.opencode_build_appworld_react.agent_roll
"""

import os
import subprocess
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.utils.env_service_client.env_client_ng import EnvClient
from ajet.schema.task import Task
from tutorial.opencode_build_appworld_react.agent_run import run_agent_and_compute_reward


# ==================== Configuration ====================

# Local configurations (client-side)
LOCAL_GRPO_N = 4  # GRPO group size (number of rollouts per task)
LOCAL_NUM_EPOCH = 1000  # Number of training epochs
LOCAL_MAX_PARALLEL = 8  # Maximum parallel episodes

# Remote configurations (server-side)
REMOTE_SWARM_URL = "http://localhost:10086"  # Swarm server URL
REMOTE_BATCH_SIZE = 32  # Batch size for training
REMOTE_ALLOCATE_GPU_PER_NODE = 8  # Number of GPUs to use
REMOTE_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

# Environment service configuration
ENV_SERVICE_URL = "http://localhost:8080"  # Environment service URL
ENV_TYPE = "appworld"  # Environment type

# AppWorld setup paths
APPWORLD_PACK_URL = "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v3.tar.gz"
APPWORLD_INSTALL_PATH = "/tmp/pack_all_in_one"


# ==================== Helper Functions ====================

def setup_appworld():
    """
    Download and setup AppWorld environment.
    This should be run before starting the training.
    """
    print("Setting up AppWorld environment...")

    # Set environment variables
    os.environ["APPWORLD_PATH"] = APPWORLD_INSTALL_PATH
    os.environ["APPWORLD_SCRIPT"] = "bash EnvService/env_sandbox/appworld.sh"

    # Check if already installed
    if os.path.exists(APPWORLD_INSTALL_PATH):
        print(f"AppWorld already installed at {APPWORLD_INSTALL_PATH}")
        return

    # Download and extract AppWorld
    print("Downloading AppWorld...")
    subprocess.run(
        ["wget", APPWORLD_PACK_URL, "-O", "/tmp/appworld_pack_v3.tar.gz"],
        check=True
    )

    print("Extracting AppWorld...")
    subprocess.run(
        ["tar", "-xzf", "/tmp/appworld_pack_v3.tar.gz", "-C", "/tmp"],
        check=True
    )

    print("AppWorld setup complete!")


def get_task_list_from_env(env_service_url: str, env_type: str, split: str = "train") -> list[str]:
    """
    Get list of available tasks from the environment service.

    Args:
        env_service_url: URL of the environment service
        env_type: Type of environment (e.g., "appworld")
        split: Dataset split ("train", "test", etc.)

    Returns:
        List of task IDs
    """
    env_client = EnvClient(base_url=env_service_url)
    task_ids = env_client.get_env_profile(env_type=env_type, split=split)
    return task_ids


def create_task_from_id(task_id: str, env_type: str) -> Task:
    """
    Create a Task object from a task ID.

    Args:
        task_id: The task identifier
        env_type: Type of environment

    Returns:
        Task object
    """
    return Task(
        task_id=task_id,
        env_type=env_type,
        main_query="",  # Will be set by environment
        init_messages=[],
        metadata={"source": "appworld"}
    )


# ==================== Main Training Function ====================

def main():
    """
    Main training loop for AppWorld React agent.
    """

    # Setup AppWorld environment
    print("=" * 60)
    print("AppWorld React Agent Training")
    print("=" * 60)

    try:
        setup_appworld()
    except Exception as e:
        print(f"Warning: AppWorld setup failed: {e}")
        print("Make sure AppWorld is properly installed before running training.")

    # Get task list from environment service
    print("\nFetching task list from environment service...")
    try:
        task_ids = get_task_list_from_env(ENV_SERVICE_URL, ENV_TYPE, split="train")
        print(f"Found {len(task_ids)} tasks")
    except Exception as e:
        print(f"Error: Failed to get task list: {e}")
        print("Make sure the environment service is running at {ENV_SERVICE_URL}")
        return

    if not task_ids:
        print("Error: No tasks found. Please check environment service.")
        return

    # Initialize swarm client
    print("\nConnecting to swarm server...")
    swarm_worker = SwarmClient(REMOTE_SWARM_URL)

    # Configure and start training engine
    print("Configuring training engine...")
    yaml_job = AgentJetJob(
        algorithm="grpo",
        project_name="appworld-react-agent",
        experiment_name="qwen2.5-7b-appworld",
        n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_TRAIN_MODEL,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
    )

    swarm_worker.auto_sync_train_config_and_start_engine(yaml_job)
    print("Training engine started!")

    # Define rollout function
    def rollout(task: Task) -> float | None:
        """
        Execute a single episode rollout.

        Args:
            task: The task to execute

        Returns:
            Reward value or None if failed
        """
        try:
            # Begin episode
            episode_uuid, api_baseurl_key = swarm_worker.begin_episode()

            # Execute agent
            workflow_output = run_agent_and_compute_reward(
                task=task,
                base_url=api_baseurl_key.base_url,
                api_key=api_baseurl_key.api_key,
                env_service_url=ENV_SERVICE_URL
            )

            # Report output back to swarm server
            swarm_worker.end_episode(task, episode_uuid, workflow_output)

            # Print rollout statistics
            swarm_worker.print_rollout_stat()

            reward = workflow_output.reward
            if isinstance(reward, list):
                return reward[0] if reward else 0.0
            return reward if reward is not None else 0.0
        except Exception as e:
            print(f"Episode failed: {e}")
            return None

    # Training loop
    print("\nStarting training loop...")
    print(f"Configuration:")
    print(f"  - GRPO N: {LOCAL_GRPO_N}")
    print(f"  - Batch Size: {REMOTE_BATCH_SIZE}")
    print(f"  - Max Epochs: {LOCAL_NUM_EPOCH}")
    print(f"  - Model: {REMOTE_TRAIN_MODEL}")
    print("=" * 60)

    next_batch = []
    total_episodes = 0

    try:
        for epoch in range(LOCAL_NUM_EPOCH):
            print(f"\nEpoch {epoch + 1}/{LOCAL_NUM_EPOCH}")

            # Iterate through tasks
            for task_id in task_ids:
                # Create task object
                task = create_task_from_id(task_id, ENV_TYPE)

                # Rollout GRPO_N times for this task
                for _ in range(LOCAL_GRPO_N):
                    next_batch.append(task)

                    # Execute batch when ready
                    if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                        print(f"\nExecuting batch of {len(next_batch)} episodes...")

                        episode_results = run_episodes_until_all_complete(
                            next_batch,
                            func=rollout,
                            auto_retry=True
                        )

                        total_episodes += len(next_batch)

                        # Print statistics
                        successful_episodes = sum(1 for r in episode_results if r is not None)
                        avg_reward = sum(r for r in episode_results if r is not None) / max(successful_episodes, 1)

                        print(f"Batch complete:")
                        print(f"  - Total episodes: {total_episodes}")
                        print(f"  - Successful: {successful_episodes}/{len(next_batch)}")
                        print(f"  - Average reward: {avg_reward:.4f}")

                        next_batch.clear()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Execute remaining episodes if any
        if next_batch:
            print("\nExecuting remaining episodes...")
            run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)

        print("\nTraining complete!")
        print(f"Total episodes executed: {total_episodes}")


if __name__ == "__main__":
    main()
