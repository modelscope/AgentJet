"""
Process level warm up
"""


import asyncio
import logging
import os
from datetime import datetime
from ajet.utils.async_utils import (
    apply_httpx_aclose_patch,
    silence_hermes_tool_parser_loggers,
    suppress_httpx_aclose_exception,
)
apply_httpx_aclose_patch()
suppress_httpx_aclose_exception()

def init_parallel_rollout_logger(experiment_dir):
    """Initialize the logger with the given configuration."""
    if "PROCESS_LEVEL_WARMUP_INIT_LOGGER" in os.environ:
        return

    os.environ["PROCESS_LEVEL_WARMUP_INIT_LOGGER"] = "1"

    from beast_logger import register_logger

    final_log_path = os.path.join(
        experiment_dir,
        datetime.now().strftime("%Y_%m_%d_%H_%M"),
        os.uname().nodename, # machine host name
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

    silence_hermes_tool_parser_loggers()
    logging.getLogger("httpx").setLevel(logging.WARNING)



def warm_up_task_judge_when_needed(config):
    if config.ajet.task_judge.judge_type == "rubrics_auto_grader":
        from ajet.task_judge.rm_auto_grader_judge import AutoGraderJudge

        judge = AutoGraderJudge(config)
        asyncio.run(judge.generate_rubrics_from_samples())
        asyncio.run(judge.load_rubrics_from_cache())


def clean_up_tmp_ajet_dir(config):
    """Clean up old IPC socket files in the IPC socket directory."""
    import time
    if config.ajet.enable_interchange_server is False:
        return

    tmp_dir = os.getenv("AJET_IPC_DIR", "/tmp/agentjet")
    if not os.path.exists(tmp_dir):
        return
    current_time = time.time()
    ttl = 4 * 3600
    print(f"Clean up old IPC socket files in {tmp_dir}.")
    try:
        for filename in os.listdir(tmp_dir):
            if not filename.endswith(".sock"):
                continue

            file_path = os.path.join(tmp_dir, filename)
            try:
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
    experiment_dir = config.ajet.experiment_dir
    init_parallel_rollout_logger(experiment_dir)
    warm_up_task_judge_when_needed(config)
    clean_up_tmp_ajet_dir(config)
