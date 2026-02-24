from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.as_swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.task_reader import RouterTaskReader
from tutorial.example_academic_trans_skills.trans import execute_agent

# python -m tutorial.example_academic_trans_skills.trans_roll


# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # grpo group size
LOCAL_NUM_EPOCH = 10000
LOCAL_NUM_EPOCH = 1
LOCAL_MAX_PARALLEL = 32
LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/agentjet/agentjet/tmp/arxiv_papers/train.parquet"
REMOTE_SWARM_URL = "http://localhost:10086" # Change to your swarm remote url

# --------- configurations that take effect remotely -------------
REMOTE_BATCH_SIZE = 8
REMOTE_ALLOCATE_GPU_PER_NODE = 8
REMOTE_TRAIN_MODEL_01 = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

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

    # Hand shake with remote swarm server
    swarm_worker = SwarmClient(REMOTE_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            project_name="ajet-swarm-skills",
            experiment_name="academic-translation",
            n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_TRAIN_MODEL_01,
            batch_size=REMOTE_BATCH_SIZE,
            num_repeat=LOCAL_GRPO_N,
        ),
    )

    def rollout(task) -> float | None:
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode()
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        # print global rollout status across the swarm
        swarm_worker.print_rollout_stat()
        return workflow_output.reward

    next_batch = []
    for _, task in enumerate(dataset.generate_training_tasks()):
        for _ in range(LOCAL_GRPO_N):
            next_batch.append(task)
            if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                # wait until getting `local_batching_size` next_batch, then execute them with with retry logic
                episode_results = run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)
                print(episode_results)
                next_batch.clear()
    return None


if __name__ == "__main__":
    main()
