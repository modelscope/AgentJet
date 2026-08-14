# -*- coding: utf-8 -*-
"""e2b ATB coding-agent RL swarm client (镜像 claudecode_geo3k_swarm).

训练模型 = Qwen3.6-35B-A3B (与 geo3k_swarm_client 一致), 单节点 8×H20.
每个 episode = e2b 沙箱内 claude code (以策略为大脑) 解 ATB v3 任务 + 固定 judge 打分.

参数 (用户指定): 并行最大 16, remote batch_size 4, num_repeat 4.

Run:
    python -m tutorial.e2b_atbench.e2b_atbench_swarm_client
前置: gw_tunnel session 持续跑 ssh 隧道到模型网关; ajet-swarm start --swarm-port=10086.
"""

import os


from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

from tutorial.e2b_atbench.atb_task_reader import AtbDirTaskReader
from tutorial.e2b_atbench.e2b_atbench_agent import _execute_agent

# ── 用户指定参数 ─────────────────────────────────────────────
REMOTE_BATCH_SIZE = 16      # remote batch size (修正 1→4: 原 1×repeat4=4ep→~10sample < world_size16 触发对齐清空 max-empty; 4×4=16ep→~40sample)
NUM_REPEAT = 4             # num_repeat (GRPO 组大小)
MAX_PARALLEL = 64          # 并行最大 episode 数
# ────────────────────────────────────────────────────────────
NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3___6-35B-A3B",
)
REMOTE_ALLOCATE_GPU_PER_NODE = 8   # 每节点 8 卡
REMOTE_NNODES = int(os.getenv("REMOTE_NNODES", "4"))   # 8851+8852+8853+8854 = 32 卡
REMOTE_TENSOR_PARALLEL_SIZE = 8
REMOTE_ULYSSES_SP = int(os.getenv("ULYSSES_SEQUENCE_PARALLEL_SIZE", "2"))
# 用户指定: ppo_max_token_len_per_gpu (PPO 更新每 GPU token 预算). 若 OOM 二分下调.
REMOTE_PPO_MAX_TOKEN_LEN_PER_GPU = int(os.getenv("PPO_MAX_TOKEN_LEN_PER_GPU", "30000"))
REMOTE_GPU_MEMORY_UTILIZATION = float(os.getenv("REMOTE_GPU_MEMORY_UTILIZATION", "0.75"))
E2B_VLLM_TOOL_PARSER = os.getenv("E2B_ATBENCH_VLLM_TOOL_PARSER", "qwen3_coder")

# ATB v3 任务池
CLEAN_TASKS_ROOT = os.getenv(
    "E2B_ATBENCH_CLEAN_TASKS",
    "/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet2/tmp/coding-agent-material/0730_ATBV3/Clean_Tasks",
)
TASK_LIMIT = int(os.getenv("E2B_ATBENCH_TASK_LIMIT", "0")) or None  # 0=不限

ajet_job = AgentJetJob(
    ensure_new_experiment=True,
    experiment_name="e2b_atbench_grpo_alpha_1",
    algorithm="grpo",
    logging="swanlab",
    n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
    nnodes=REMOTE_NNODES,
    model=REMOTE_MODEL_PATH,
    batch_size=REMOTE_BATCH_SIZE,
    num_repeat=NUM_REPEAT,
    swarm_mode_sample_collection_method="rollout_until_all_clients_agree_sync_weight",
    max_env_worker=MAX_PARALLEL,
    tensor_model_parallel_size=REMOTE_TENSOR_PARALLEL_SIZE,
    ulysses_sequence_parallel_size=REMOTE_ULYSSES_SP,
    ppo_max_token_len_per_gpu=REMOTE_PPO_MAX_TOKEN_LEN_PER_GPU,
    gpu_memory_utilization=REMOTE_GPU_MEMORY_UTILIZATION,
    max_num_seqs=1024,   # 用户指定: vLLM 每引擎并行 seq 数 (默认64)
    # 用户指定: 126k 上下文, 单轮 4k, 总 response 96k, prompt 30k
    max_prompt_length=30000,
    max_response_length=96000,
    max_model_len=126000,   # >= prompt(30000)+response(96000)=126000
    max_response_length_in_one_turn=10240,
)
# The Qwen3.6 model emits XML tool calls (`<function=...>`), so keep the
# engine parser and AgentJet's response parser on the same configured format.
ajet_job.config.ajet.rollout.vllm_tool_parser = E2B_VLLM_TOOL_PARSER
# Checkpoint 保存间隔: 每 N 步存一次 (默认 5; ajet 原默认 20). 经 align_parameters
# 映射到 trainer.save_freq, 在 global_steps % save_freq == 0 时触发 _save_checkpoint.
ajet_job.config.ajet.trainer_common.save_freq = int(os.getenv("AJET_SAVE_FREQ", "5"))


def main():
    assert AJET_SWARM_URL != "http://swarm-server-ip:10086", "set AJET_SWARM_URL"

    dataset = AtbDirTaskReader(CLEAN_TASKS_ROOT, limit=TASK_LIMIT)
    # 预览任务数
    tasks_preview = list(dataset.generate_training_tasks())
    print(f"[e2b_atbench] loaded {len(tasks_preview)} ATB tasks from {CLEAN_TASKS_ROOT}")
    assert tasks_preview, f"no ATB tasks found under {CLEAN_TASKS_ROOT}"

    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        force_restart=os.environ.get("FORCE_RESTART_SWARM_ENGINE", "0") == "1",
    )

    exp_dir = swarm_worker.server_experiment_dir() or "saved_experiments"
    os.makedirs(exp_dir, exist_ok=True)
    os.environ.setdefault("LLM_IO_LOG", os.path.join(exp_dir, "cc_dump_tree.log"))

    def rollout(task):
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=3600  # claude code episode 慢, idle 超时给足
        )
        try:
            workflow_output = _execute_agent(task, api_baseurl_key)
        except Exception as e:
            print(f"[e2b_atbench] rollout error task={task.task_id}: {e}")
            workflow_output = None
        if workflow_output is None:
            swarm_worker.abort_episode(episode_uuid)
            return
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

    executor = PeriodicDrainThreadPoolExecutor(
        workers=NUM_REPEAT * REMOTE_BATCH_SIZE, max_parallel=MAX_PARALLEL, auto_retry=True
    )
    task_count = 0
    for _ in range(NUM_EPOCH):
        for task in tasks_preview:
            for _ in range(NUM_REPEAT):
                # A drain boundary represents one fully collected local batch.
                # With rollout_until_all_clients_agree_sync_weight, explicitly
                # acknowledge that batch so the server can coordinate the
                # weight sync across all active clients.
                _, drained_results = executor.submit_with_periodic_drain(
                    fn=rollout,
                    task=task,
                )
                if drained_results:
                    swarm_worker.agree_sync_weight()
            task_count += 1
    return None


if __name__ == "__main__":
    main()
