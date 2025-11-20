import os


def init_parallel_rollout_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    from beast_logger import register_logger

    if "BEST_LOGGER_INIT" in os.environ:
        return  # prevent re-initialization in ray environment
    os.environ["BEST_LOGGER_INIT"] = "1"
    from datetime import datetime

    final_log_path = os.path.join(
        "launcher_record",
        experiment_name,
        datetime.now().strftime("%Y_%m_%d_%H_%M"),
    )
    os.environ["BEST_LOGGER_PATH"] = final_log_path
    non_console_mods = ["rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(
        mods=["evaluation", "exception"],
        non_console_mods=non_console_mods,
        auto_clean_mods=[],
        base_log_path=final_log_path,
        debug=False,
    )
