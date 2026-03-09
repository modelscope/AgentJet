import argparse
import os

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
    get_backbone_target,
    setup_environment_vars,
)
from ajet.utils.pty import pty_launch

set_loguru_default_color()
load_dotenv(override=False)

DEFAULT_DIR = "saved_experiments"

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
        "--swarm-server",
        action="store_true",
        default=False,
        help="Enable Swarm server mode",
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
        default=DEFAULT_DIR,
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
        "--with-ray", action="store_true", default=False, help="Launch ray"
    )
    parser.add_argument(
        "--with-ray-cluster", action="store_true", default=False, help="Launch ray"
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
        "--with-bfcl", action="store_true", default=False, help="Launch bfcl"
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
        help="Skip GPU availability check",
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
        help="Prefix for deepfinance service names",
    )
    parser.add_argument(
        "--swarm-overwatch",
        type=str,
        default="",
        required=False,
        help="Swarm server URL for overwatch monitoring (e.g., http://localhost:10086)",
    )
    return parser.parse_args()


def check_model_file_exists(exp_config):
    model_path = exp_config["ajet"]["model"]["path"]
    # if model_path has more than 2 '/', we consider it as a dir path
    if model_path.count("/") > 2:
        assert os.path.exists(model_path), (
            f"Model path {model_path} does not exist. Please check your configuration."
        )


def start_swarm_server(env, config):
    config = dict_to_namespace(config)
    assert config.ajet.enable_swarm_mode, (
        "Please enable_swarm_mode in config to start swarm server."
    )
    assert config.ajet.enable_interchange_server, (
        "Please enable_interchange_server in config to start swarm server."
    )
    from ajet.tuner_lib.experimental.oai_model_server import (
        start_interchange_server,
    )

    start_interchange_server(config, blocking=True, env=env)


def main():
    args = parse_args()

    # Handle swarm overwatch mode
    if args.swarm_overwatch:
        from ajet.utils.swarm_overwatch import start_overwatch

        logger.info(f"Starting Swarm Overwatch for server: {args.swarm_overwatch}")
        start_overwatch(args.swarm_overwatch, refresh_interval=2.0)
        return

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
                logger.success(
                    f"Successfully killed processes with PIDs: {killed_pids}"
                )
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
    if args.swarm_server and (not args.conf):
        args.conf = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "default_config/ajet_ts_default.yaml"
            )
        )
        assert os.path.exists(args.conf), (
            "Please provide a valid config file for swarm server mode."
        )
    if args.conf:
        exp_base_dir = args.exp_dir or DEFAULT_DIR
        yaml_path = args.conf
        (
            main_yaml_fp,
            exe_exp_base,
            exp_name,
            exp_config,
        ) = prepare_experiment_config(
            yaml_path=yaml_path,
            exp_base_dir=exp_base_dir,
            backbone=args.backbone,
            storage=(not args.swarm_server)
        )

    # setup environment variables
    env, exp_config = setup_environment_vars(args, exp_config, main_yaml_fp)

    if args.swarm_server:
        start_swarm_server(env, exp_config)
        return

    if args.with_ray:
        assert not args.with_ray_cluster, (
            "Cannot use both --with-ray and --with-ray-cluster simultaneously."
        )
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
        assert not args.with_ray, (
            "Cannot use both --with-ray and --with-ray-cluster simultaneously."
        )
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
