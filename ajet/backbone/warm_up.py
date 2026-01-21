"""
Process level warm up
"""


import asyncio
import logging
import os
from ajet.utils.async_utils import apply_httpx_aclose_patch, suppress_httpx_aclose_exception
apply_httpx_aclose_patch()
suppress_httpx_aclose_exception()


def init_parallel_rollout_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    if "PROCESS_LEVEL_WARMUP_INIT_LOGGER" in os.environ:
        return
    os.environ["PROCESS_LEVEL_WARMUP_INIT_LOGGER"] = "1"

    from datetime import datetime

    from beast_logger import register_logger

    final_log_path = os.path.join(
        "saved_experiments",
        experiment_name,
        datetime.now().strftime("%Y_%m_%d_%H_%M"),
        # machine host name
        os.uname().nodename,
    )
    os.environ["BEST_LOGGER_PATH"] = final_log_path
    non_console_mods = ["rollout", "token_clip", "bad_case"]
    register_logger(
        mods=["evaluation", "exception", "benchmark"],
        non_console_mods=non_console_mods,
        auto_clean_mods=[],
        base_log_path=final_log_path,
        debug=False,
    )

    target_logger = logging.getLogger("vllm.entrypoints.openai.tool_parsers.hermes_tool_parser")
    target_logger.setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)



def warm_up_task_judge_when_needed(config):
    if config.ajet.task_judge.judge_type == "rubrics_auto_grader":
        from ajet.task_judge.rm_auto_grader_judge import AutoGraderJudge

        judge = AutoGraderJudge(config)
        asyncio.run(judge.generate_rubrics_from_samples())
        asyncio.run(judge.load_rubrics_from_cache())


def clean_up_tmp_ajet_dir(config):
    """Clean up old IPC socket files in /tmp/ajet directory."""
    import time
    if config.ajet.enable_experimental_interchange_server is False:
        return

    tmp_dir = "/tmp/ajet"
    if not os.path.exists(tmp_dir):
        return
    current_time = time.time()
    ttl = 4 * 3600
    try:
        for filename in os.listdir(tmp_dir):
            if not filename.endswith(".sock"):
                continue

            file_path = os.path.join(tmp_dir, filename)
            try:
                print(current_time - os.path.getmtime(file_path))
                if current_time - os.path.getmtime(file_path) > ttl:
                    os.remove(file_path)
            except OSError:
                pass
    except OSError:
        pass


def warm_up_process(config):
    """
    Process level warm up
    This will not be called multiple when:
        - multi-threading
        - forked multi-processing
    This may be called multiple times when:
        - spawned multi-processing
        - ray remote actor

    ---

    Note: Skipping process level warm up will not cause significant issues, but may lead to
    slightly longer initialization times for certain components in each process.
    """

    if "PROCESS_LEVEL_WARMUP_INIT" in os.environ:
        return
    os.environ["PROCESS_LEVEL_WARMUP_INIT"] = "1"
    experiment_name = config.ajet.experiment_name
    init_parallel_rollout_logger(experiment_name)
    warm_up_task_judge_when_needed(config)
    clean_up_tmp_ajet_dir(config)
