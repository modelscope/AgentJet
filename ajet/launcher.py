import argparse
import os
import subprocess
from types import SimpleNamespace

from dotenv import load_dotenv
from loguru import logger

from ajet.utils.cleaner import fast_kill_by_keyword_bash
from ajet.utils.config_utils import prepare_experiment_config
from ajet.utils.launch_utils import (
    execute_training_process,
    launch_logview,
    set_loguru_default_color,
    start_ray_service,
    check_debugpy_version,
    check_avail_gpu,
    dict_to_namespace,
)
from ajet.utils.pty import pty_launch

set_loguru_default_color()
load_dotenv(override=False)


def parse_args():
    parser = argparse.ArgumentParser(description="AgentJet Launcher")
    parser.add_argument(
        "--backbone",
        type=str,
        default="verl",
        required=False,
        help="verl or trinity or debug",
    )
    parser.add_argument(
        "--tinkerscript-server",
        action="store_true",
        default=False,
        help="Enable TinkerScript server mode",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="",
        required=False,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="saved_experiments",
        required=False,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--debug",
        "--db",
        type=str,
        default="",
        required=False,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--with-ray",
        action="store_true",
        default=False,
        help="Launch ray"
    )
    parser.add_argument(
        "--with-ray-cluster",
        action="store_true",
        default=False,
        help="Launch ray"
    )
    parser.add_argument(
        "--with-appworld",
        action="store_true",
        default=False,
        help="Launch appworld",
    )
    parser.add_argument(
        "--with-deepfinance",
        action="store_true",
        default=False,
        help="Launch deepfinance",
    )
    parser.add_argument(
        "--with-webshop",
        action="store_true",
        default=False,
        help="Launch webshop",
    )
    parser.add_argument(
        "--with-bfcl",
        action="store_true",
        default=False,
        help="Launch bfcl"
    )
    parser.add_argument(
        "--with-logview",
        action="store_true",
        default=False,
        help="Launch logview",
    )
    parser.add_argument(
        "--with-crafters",
        action="store_true",
        default=False,
        help="Launch Crafters Env Simulation",
    )
    parser.add_argument(
        "--skip-check-avail-gpu",
        action="store_true",
        default=False,
        help="Skip GPU availability check"
    )
    parser.add_argument(
        "--kill",
        type=str,
        default="",
        required=False,
        help="list of keywords for killing processes",
    )
    parser.add_argument(
        "--autokill",
        action="store_true",
        default=False,
        help="Kill system processes (ray + vllm + python) that may block the current experiment",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        required=False,
        help="Prefix for deepfinance service names"
    )
    return parser.parse_args()


def get_backbone_target(backbone):
    """
    Determine the appropriate backbone target module based on the backbone name.

    Args:
        backbone (str): The backbone name (e.g., "verl", "debug", "trinity")

    Returns:
        str: The full module path for the specified backbone
    """
    backbone_target = "ajet.backbone.main_verl"  # Default to trinity
    if backbone == "verl":
        backbone_target = "ajet.backbone.main_verl"
    if backbone == "debug":
        backbone_target = "ajet.backbone.main_vllm"
    if backbone == "trinity":
        backbone_target = "ajet.backbone.main_trinity"
    return backbone_target


def setup_environment_vars(args, exp_config, main_yaml_fp):
    """
    Configure environment variables based on command line arguments.

    Args:
        args: Command line arguments
        exp_config: Experiment configuration dictionary
        main_yaml_fp: Path to main YAML configuration file

    Returns:
        dict: Configured environment variables dictionary
    """
    env = os.environ.copy()
    if args.debug:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.debug
        env["RAY_record_task_actor_creation_sites"] = "true"
        # assert exp_config["ajet"]["rollout"]["max_env_worker"] <= 4, "parallel worker too many for debugging mode"  # type: ignore
        if exp_config["ajet"]["rollout"]["max_env_worker"] > 1:  # type: ignore
            exp_config["ajet"]["rollout"]["max_env_worker"] = 1
            logger.warning(
                "For debugging mode, max_env_worker is set to 1 to facilitate debugging."
            )
        logger.warning("Debug mode is ON")
    else:
        logger.warning("Debug mode is OFF")
        # if args.conf:
        #     assert exp_config["ajet"]["rollout"]["max_env_worker"] > 4, "parallel worker too few"  # type: ignore
    if args.backbone == "trinity":
        env["AJET_CONFIG_REDIRECT"] = main_yaml_fp  # type: ignore
    if args.backbone == "debug":
        env["AJET_DEBUG"] = "1"  # type: ignore
    return env, exp_config


def check_model_file_exists(exp_config):
    model_path = exp_config["ajet"]["model"]["path"]
    # if model_path has more than 2 '/', we consider it as a dir path
    if model_path.count("/") > 2:
        assert os.path.exists(model_path), f"Model path {model_path} does not exist. Please check your configuration."


def start_tinkerscript_server(env, config):
    config = dict_to_namespace(config)
    assert config.ajet.enable_tinkerscript_mode, \
        "Please enable_tinkerscript_mode in config to start tinkerscript server."
    assert config.ajet.enable_experimental_interchange_server, \
        "Please enable_experimental_interchange_server in config to start tinkerscript server."
    from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import start_interchange_server
    start_interchange_server(config, blocking=True)


def main():
    args = parse_args()

    # Enforce GPU availability and free memory threshold before proceeding
    if not args.skip_check_avail_gpu:
        if (args.backbone != "debug") and (not args.kill) and (not args.autokill):
            check_avail_gpu(min_free_ratio=0.95)

    if args.autokill:
        args.kill = "ray|vllm|VLLM|python"

    # Handle kill-keywords argument if provided
    if args.kill:
        logger.info(f"Killing processes matching keywords: {args.kill}")
        for keyword in args.kill.split("|"):
            logger.info(f"Killing processes matching keyword: {keyword}")
            killed_pids = fast_kill_by_keyword_bash(keyword)
            if killed_pids:
                logger.success(f"Successfully killed processes with PIDs: {killed_pids}")
            else:
                logger.warning(f"No processes found matching keyword: {keyword}")
        if not args.conf:
            return

    # Initialize variables with default values to avoid "possibly unbound" errors
    main_yaml_fp = None
    exe_exp_base = None
    exp_name = None

    # switch backbone target
    backbone_target = get_backbone_target(args.backbone)

    # read configuration from yaml
    exp_config = None
    exp_dir = args.exp_dir or "saved_experiments"
    if args.tinkerscript_server and (not args.conf):
        args.conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "default_config/ajet_ts_default.yaml"))
        assert os.path.exists(args.conf), "Please provide a valid config file for tinkerscript server mode."
    if args.conf:
        yaml_path = args.conf
        (
            main_yaml_fp,
            exe_exp_base,
            exp_name,
            exp_config,
        ) = prepare_experiment_config(yaml_path, exp_dir, args.backbone)

    # setup environment variables
    env, exp_config = setup_environment_vars(args, exp_config, main_yaml_fp)

    if args.tinkerscript_server:
        start_tinkerscript_server(env, exp_config)
        return

    if args.with_ray:
        assert (
            not args.with_ray_cluster
        ), "Cannot use both --with-ray and --with-ray-cluster simultaneously."
        start_ray_service(args, env)

    if args.with_appworld:
        pty_launch("appworld")

    if args.with_deepfinance:
        pty_launch("deepfinance", prefix=args.prefix)

    if args.with_crafters:
        pty_launch("crafters")

    if args.with_webshop:
        pty_launch("webshop")

    if args.with_bfcl:
        pty_launch("bfcl")

    if args.with_logview:
        launch_logview(exp_name)

    if args.with_ray_cluster:
        assert (
            not args.with_ray
        ), "Cannot use both --with-ray and --with-ray-cluster simultaneously."
        start_ray_service(args, env, cluster=True)

    if args.conf and main_yaml_fp and exe_exp_base and exp_config:
        check_model_file_exists(exp_config)
        execute_training_process(
            args,
            backbone_target,
            main_yaml_fp,
            exe_exp_base,
            main_yaml_fp,
            env,
            exp_config,
        )


if __name__ == "__main__":
    check_debugpy_version()
    main()
