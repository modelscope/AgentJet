from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.as_swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.task_reader import RouterTaskReader
from tutorial.example_academic_trans.train_multi_model.trans import execute_agent


# Step 1: Start two swarm servers on different ports:
# ajet-swarm start --swarm-port=10086  # For 7B model
# ajet-swarm start --swarm-port=10087  # For 14B model
# Step 2: python -m tutorial.example_academic_trans.train_multi_model.trans_roll

# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # grpo group size
LOCAL_NUM_EPOCH = 10000
LOCAL_NUM_EPOCH = 1
LOCAL_MAX_PARALLEL = 32
LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/agentjet/agentjet/tmp/arxiv_papers/train.parquet"

# --------- configurations for 7B model (agents 1 and 3) -------------
REMOTE_7B_SWARM_URL = "http://22.16.208.79:10086"  # Change to your swarm remote url
REMOTE_7B_BATCH_SIZE = 32
REMOTE_7B_ALLOCATE_GPU_PER_NODE = 8
REMOTE_7B_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

# --------- configurations for 14B model (agent 2) -------------
REMOTE_14B_SWARM_URL = "http://22.14.56.6:10086"  # Change to your swarm remote url
REMOTE_14B_BATCH_SIZE = 32
REMOTE_14B_ALLOCATE_GPU_PER_NODE = 8
REMOTE_14B_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct'


class WeightUpdatedHalfway(Exception):
    """Raised when the remote side starts updating model weights halfway through an episode."""


def main():

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = LOCAL_DATASET_PATH
            )
        )
    )


    # Hand shake with remote swarm server for 14B model (agent 2)
    swarm_worker_14b = SwarmClient(REMOTE_14B_SWARM_URL)
    swarm_worker_14b.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            project_name="ajet-swarm-academic-trans",
            experiment_name="14b-model",
            n_gpu=REMOTE_14B_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_14B_TRAIN_MODEL,
            batch_size=REMOTE_14B_BATCH_SIZE,
            num_repeat=LOCAL_GRPO_N,
        ),
    )

    # Hand shake with remote swarm server for 7B model (agents 1 and 3)
    swarm_worker_7b = SwarmClient(REMOTE_7B_SWARM_URL)
    swarm_worker_7b.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            project_name="ajet-swarm-academic-trans",
            experiment_name="7b-model",
            n_gpu=REMOTE_7B_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_7B_TRAIN_MODEL,
            batch_size=REMOTE_7B_BATCH_SIZE,
            num_repeat=LOCAL_GRPO_N,
        ),
    )


    def rollout(task):
        """
        Execute the translation workflow using both 7B and 14B models.
        - Agents 1 and 3 use the 7B model (rewarded based on final translation quality)
        - Agent 2 uses the 14B model (rewarded based on proper noun detection quality)
        """
        # Begin episode for 7B model (agents 1 and 3)
        episode_uuid_7b, api_baseurl_key_7b = swarm_worker_7b.begin_episode(discard_episode_timeout=240)
        # Begin episode for 14B model (agent 2)
        episode_uuid_14b, api_baseurl_key_14b = swarm_worker_14b.begin_episode(discard_episode_timeout=240)

        # Execute agent workflow with both models
        # Returns two separate WorkflowOutputs with different rewards
        workflow_output_7b, workflow_output_14b = execute_agent(task, api_baseurl_key_7b, api_baseurl_key_14b)

        # Report output back to swarm remotes with their respective rewards
        swarm_worker_7b.end_episode(task, episode_uuid_7b, workflow_output_7b)
        swarm_worker_14b.end_episode(task, episode_uuid_14b, workflow_output_14b)

        # Print global rollout status across the swarm
        swarm_worker_7b.print_rollout_stat()
        swarm_worker_14b.print_rollout_stat()

        # Return the average reward for logging purposes
        return (workflow_output_7b.reward + workflow_output_14b.reward) / 2.0


    next_batch = []
    for _, task in enumerate(dataset.generate_training_tasks()):
        for _ in range(LOCAL_GRPO_N):
            next_batch.append(task)
            if len(next_batch) >= (REMOTE_7B_BATCH_SIZE * LOCAL_GRPO_N):
                # wait until getting `local_batching_size` next_batch, then execute them with retry logic
                episode_results = run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)
                print(episode_results)
                next_batch.clear()
    return None


if __name__ == "__main__":
    main()
