"""Entry point for AgentJet swarm commands."""
import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from ajet.utils.config_utils import prepare_experiment_config
from ajet.utils.launch_utils import (
    dict_to_namespace,
    set_loguru_default_color,
    setup_environment_vars,
    check_debugpy_version,
)

set_loguru_default_color()
load_dotenv(override=False)

DEFAULT_DIR = "saved_experiments"


def start_swarm_server(env, config, port):
    config = dict_to_namespace(config)
    assert config.ajet.enable_swarm_mode, (
        "Please enable_swarm_mode in config to start swarm server."
    )
    assert config.ajet.enable_interchange_server, (
        "Please enable_interchange_server in config to start swarm server."
    )

    # Set the port in the config
    config.ajet.interchange_server.interchange_server_port = port

    from ajet.tuner_lib.experimental.oai_model_server import (
        start_interchange_server,
    )

    logger.info(f"Starting swarm server on port {port}")
    start_interchange_server(config, blocking=True, env=env)


def cmd_start(args):
    """Handle the 'start' subcommand."""
    # Use default config if not provided
    exp_base_dir = args.exp_dir or DEFAULT_DIR
    if not args.conf:
        args.conf = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "default_config/ajet_swarm_default.yaml"
            )
        )
        assert os.path.exists(args.conf), (
            "Default config file not found. Please provide a valid config file."
        )

    # Prepare experiment config
    yaml_path = args.conf
    (
        main_yaml_fp,
        exe_exp_base,
        exp_name,
        exp_config,
    ) = prepare_experiment_config(
        yaml_path=yaml_path,
        exp_base_dir=exp_base_dir,
        backbone="verl",
        storage=False
    )

    # Setup environment variables
    class SwarmArgs:
        def __init__(self, conf, backbone, exp_dir):
            self.conf = conf
            self.backbone = backbone
            self.exp_dir = exp_dir
            self.swarm_server = True
            self.swarm_overwatch = ""
            self.debug = ""
    swarm_args = SwarmArgs(args.conf, "verl", args.exp_dir)
    env, exp_config = setup_environment_vars(swarm_args, exp_config, main_yaml_fp)

    # Start swarm server
    start_swarm_server(env, exp_config, args.swarm_port)


def cmd_overwatch(args):
    """Handle the 'overwatch' subcommand."""
    from ajet.utils.swarm_overwatch import start_overwatch

    logger.info(f"Starting Swarm Overwatch for server: {args.swarm_url}")
    start_overwatch(args.swarm_url, refresh_interval=args.refresh_interval)


def main():
    parser = argparse.ArgumentParser(description="AgentJet Swarm Management")
    subparsers = parser.add_subparsers(dest="command", help="Swarm commands")

    # Subcommand: start
    parser_start = subparsers.add_parser("start", help="Start the swarm server")
    parser_start.add_argument(
        "--swarm-port",
        type=int,
        default=10086,
        required=False,
        help="Port for the swarm server (default: 10086)",
    )
    parser_start.add_argument(
        "--conf",
        type=str,
        default="",
        required=False,
        help="Path to configuration file",
    )
    parser_start.add_argument(
        "--exp-dir",
        type=str,
        default=DEFAULT_DIR,
        required=False,
        help="Path to experiment directory",
    )

    parser_start.set_defaults(func=cmd_start)

    # Subcommand: overwatch
    parser_overwatch = subparsers.add_parser("overwatch", help="Monitor the swarm server")
    parser_overwatch.add_argument(
        "--swarm-url",
        type=str,
        default="http://localhost:10086",
        required=False,
        help="Swarm server URL (default: http://localhost:10086)",
    )
    parser_overwatch.add_argument(
        "--refresh-interval",
        type=float,
        default=2.0,
        required=False,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser_overwatch.set_defaults(func=cmd_overwatch)

    # Subcommand: top (alias for overwatch)
    parser_top = subparsers.add_parser("top", help="Monitor the swarm server (alias for overwatch)")
    parser_top.add_argument(
        "--swarm-url",
        type=str,
        default="http://localhost:10086",
        required=False,
        help="Swarm server URL (default: http://localhost:10086)",
    )
    parser_top.add_argument(
        "--refresh-interval",
        type=float,
        default=2.0,
        required=False,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser_top.set_defaults(func=cmd_overwatch)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    check_debugpy_version()
    main()
