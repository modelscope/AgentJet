from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.task_reader import RouterTaskReader
from tutorial.example_train_multi_model.trans import execute_agent


# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4
LOCAL_NUM_EPOCH = 10000
LOCAL_MAX_PARALLEL = 64
LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/agentjet/agentjet/tmp/arxiv_papers/train.parquet"

# --------- 7B (agents 1, 3) on localhost:10086 ----------
REMOTE_7B_SWARM_URL = "http://localhost:10086"
REMOTE_7B_BATCH_SIZE = 64
REMOTE_7B_ALLOCATE_GPU_PER_NODE = 4
REMOTE_7B_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

# --------- 14B (agent 2) on localhost:10087 ----------
REMOTE_14B_SWARM_URL = "http://localhost:10087"
REMOTE_14B_BATCH_SIZE = 64
REMOTE_14B_ALLOCATE_GPU_PER_NODE = 4
REMOTE_14B_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct'


def _dset(container, key, value):
    if isinstance(container, dict):
        container[key] = value
    else:
        setattr(container, key, value)


def main():
    dataset = RouterTaskReader(
        reader_type="huggingface_dat_repo",
        reader_config=AjetTaskReader(
            huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=LOCAL_DATASET_PATH)
        ),
    )

    job_14b = AgentJetJob(
        algorithm="grpo",
        project_name="ajet-swarm-academic-trans-lora",
        experiment_name="14b-model-lora",
        n_gpu=REMOTE_14B_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_14B_TRAIN_MODEL,
        batch_size=REMOTE_14B_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
        logging="swanlab",
        lora_rank=32,
        lora_alpha=32,
        lora_load_format="safetensors",
        layered_summon=True,
        lr=3e-4,

    )

    job_7b = AgentJetJob(
        algorithm="grpo",
        project_name="ajet-swarm-academic-trans-lora",
        experiment_name="7b-model-lora",
        n_gpu=REMOTE_7B_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_7B_TRAIN_MODEL,
        batch_size=REMOTE_7B_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
        logging="swanlab",
        lora_rank=32,
        lora_alpha=32,
        lora_load_format="safetensors",
        layered_summon=True,
        lr=3e-4,
    )

    # Original (sequential) version:
    # swarm_worker_14b = SwarmClient(REMOTE_14B_SWARM_URL)
    # swarm_worker_14b.auto_sync_train_config_and_start_engine(job_14b, force_restart=True)
    #
    # swarm_worker_7b = SwarmClient(REMOTE_7B_SWARM_URL)
    # swarm_worker_7b.auto_sync_train_config_and_start_engine(job_7b, force_restart=True)
    swarm_worker_14b = SwarmClient(REMOTE_14B_SWARM_URL)
    swarm_worker_7b = SwarmClient(REMOTE_7B_SWARM_URL)
    SwarmClient.async_and_start_multi_engine(
        [(swarm_worker_14b, job_14b), (swarm_worker_7b, job_7b)],
        force_restart=True,
    )

    def rollout(task):
        episode_uuid_7b, api_baseurl_key_7b = swarm_worker_7b.begin_episode(discard_episode_timeout=240)
        episode_uuid_14b, api_baseurl_key_14b = swarm_worker_14b.begin_episode(discard_episode_timeout=240)

        workflow_output_7b, workflow_output_14b = execute_agent(task, api_baseurl_key_7b, api_baseurl_key_14b)

        swarm_worker_7b.end_episode(task, episode_uuid_7b, workflow_output_7b)
        swarm_worker_14b.end_episode(task, episode_uuid_14b, workflow_output_14b)

        swarm_worker_7b.print_rollout_stat()
        swarm_worker_14b.print_rollout_stat()

        return (workflow_output_7b.reward + workflow_output_14b.reward) / 2.0

    executor = PeriodicDrainThreadPoolExecutor(workers=REMOTE_7B_BATCH_SIZE * LOCAL_GRPO_N, max_parallel=LOCAL_MAX_PARALLEL, auto_retry=True)
    for _, task in enumerate(dataset.generate_training_tasks()):
        for _ in range(LOCAL_GRPO_N):
            executor.submit_with_periodic_drain(fn=rollout, task=task)
    return None


if __name__ == "__main__":
    main()
