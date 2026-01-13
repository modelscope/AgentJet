import logging
import os
import shutil
import subprocess
import sys
import time

from beast_logger import print_dict
from loguru import logger

from ajet.utils.config_utils import align_parameters
from ajet.utils.smart_daemon import LaunchCommandWhenAbsent


def set_loguru_default_color():
    logger.remove()
    colorize = os.environ.get("LOGURU_COLORIZE", "YES").upper() not in ["NO", "0", "FALSE"]
    logger.add(sys.stderr, colorize=colorize, enqueue=False)
    if not colorize:
        os.environ["RAY_COLOR_PREFIX"] = "0"

    logging.getLogger("vllm.entrypoints.openai.tool_parsers.hermes_tool_parser").setLevel(
        logging.CRITICAL
    )
    return


def launch_logview(exp_name=None):
    """
    Launch the log viewer service and open the web browser to view logs.

    Args:
        exp_name: Optional experiment name. If not provided, "default_experiment" is used.
    """
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable,
            "-m",
            "web_display.start_web",
        ],
        dir="./",
        tag="logview",
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Uvicorn running on",
        env_dict={},
    )
    try:
        import webbrowser

        time.sleep(2.5)
        webbrowser.open("http://127.0.0.1:8181/")
    except Exception as e:
        logger.error(f"Error opening web browser: {e}")


def start_ray_service(args, env, cluster=False):
    """
    Start a Ray service with appropriate configuration.

    Args:
        args: Command line arguments containing debug settings
    """
    # Get the current Python interpreter directory
    python_dir = os.path.dirname(sys.executable)
    ray_path = os.path.join(python_dir, "ray")
    if not cluster:
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[f"{ray_path} start --head --block"],
            dir="./",
            tag="ray_service",
            use_pty=True,
        )
        launch_wait_time = 600
        success_std_string = "Ray runtime started"
    else:
        HOSTNAME = os.uname().nodename
        MASTER_ADDR = os.getenv("MASTER_ADDR")
        MASTER_PORT = os.getenv("MASTER_PORT")
        if HOSTNAME == MASTER_ADDR:
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    f"{ray_path} start --head --node-ip-address={MASTER_ADDR} --port={MASTER_PORT} --disable-usage-stats --block"
                ],
                dir="./",
                tag="ray_service_head",
                use_pty=True,
            )
            launch_wait_time = 600
            success_std_string = "Ray runtime started"
        else:
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    f"{ray_path} start --address={MASTER_ADDR}:{MASTER_PORT} --disable-usage-stats --block"
                ],
                dir="./",
                tag="ray_service_worker",
                use_pty=True,
            )
            launch_wait_time = 9999999999
            # success_std_string = "Connected to Ray cluster"
            success_std_string = "Just wait here forever"
    companion.launch(
        launch_wait_time=launch_wait_time,
        success_std_string=success_std_string,
        env_dict=env,
    )


def verify_python_env(args, exp_config):
    """
    Verify that the current Python environment matches the expected executable.

    Args:
        args: Command line arguments containing the expected python_executable
    """
    if exp_config["ajet"]["trainer_common"]["logger"] == "swanlab":
        if os.environ.get("SWANLAB_API_KEY", "") == "":
            cause = "SWANLAB_API_KEY is not set in the environment."
            solution = "To use the swanlab logger, please set `SWANLAB_API_KEY`. Otherwise, set `ajet.trainer_common.logger=tensorboard`"
            print_dict(
                {
                    "Python Environment Check": "FAILED",
                    "Cause": cause,
                    "Solution": solution,
                }
            )
            time.sleep(5)
            raise ImportError(cause + " " + solution)

    import verl
    if args.backbone == "trinity":
        if any([v in verl.__version__ for v in ["0.5.0.post", "0.7.0.post"]]):
            cause = "Python environment does not match current backbone 'trinity'."
            solution = "Please `cd /path/to/project/AgentJet` and run `(uv) pip install -e .[trinity]` to install the correct environment."
            print_dict(
                {
                    "Python Environment Check": "FAILED",
                    "Cause": cause,
                    "Solution": solution,
                }
            )
            time.sleep(5)
            raise ImportError(cause + " " + solution)
    elif args.backbone == "verl":
        if not any([v in verl.__version__ for v in ["0.5.0.post", "0.5.0.dev", "0.7.0.post"]]):  # you must install via `pip install -e .[verl]` to get every dependency right
            cause = "Python environment does not match current backbone 'verl'."
            solution = "Please `cd /path/to/project/AgentJet` and run `(uv) pip install -e .[verl]` to install the correct environment."
            print_dict(
                {
                    "Python Environment Check": "FAILED",
                    "Cause": cause,
                    "Solution": solution,
                }
            )
            time.sleep(5)
            raise ImportError(cause + " " + solution)


def execute_training_process(
    args,
    backbone_target,
    yaml_backup_dst,
    exe_exp_base,
    exe_yaml_path,
    env,
    exp_config,
):
    """
    Execute the training process based on the specified backbone and configuration.

    Args:
        args: Command line arguments
        backbone_target: The Python module to execute
        yaml_backup_dst: Path to the YAML configuration backup
        exe_exp_base: Base path for experiment execution
        exe_yaml_path: Path to the YAML configuration file
        env: Environment variables dictionary
    """

    # Fixed config asset locations
    TRINITY_BOOT_YAML = "ajet/default_config/trinity/trinity_launch.yaml"  # THIS FILE IS READ ONLY, and ALWAYS FIXED
    TRINITY_CONFIG_AUTO_CONVERSION = (
        "ajet/default_config/trinity/config_auto_convertion_trinity.jsonc"
    )
    VERL_CONFIG_AUTO_CONVERSION = (
        "ajet/default_config/verl/config_auto_convertion_verl.jsonc"
    )

    # let's begin the training process
    if args.backbone == "trinity":
        # replace boot yaml
        redirect_trinity_boot_yaml = os.path.dirname(yaml_backup_dst) + "/trinity_launch.yaml"
        shutil.copyfile(TRINITY_BOOT_YAML, redirect_trinity_boot_yaml)
        align_parameters(
            yaml_backup_dst,
            redirect_trinity_boot_yaml,
            TRINITY_CONFIG_AUTO_CONVERSION,
            args.backbone,
        )
        cmd = [
            sys.executable,
            "-m",
            backbone_target,
            "run",
            "--config",
            redirect_trinity_boot_yaml,
        ]
    else:
        align_parameters(
            yaml_backup_dst,
            yaml_backup_dst,
            VERL_CONFIG_AUTO_CONVERSION,
            args.backbone,
        )
        cmd = [
            sys.executable,
            "-m",
            backbone_target,
            "--config-path",
            os.path.abspath(exe_exp_base),
            "--config-name",
            os.path.basename(exe_yaml_path),
        ]

    if args.with_logview:
        env.update(
            {
                "BEST_LOGGER_WEB_SERVICE_URL": os.environ.get(
                    "BEST_LOGGER_WEB_SERVICE_URL", "http://127.0.0.1:8181/"
                )
            }
        )

    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        print_dict(
            {
                "Running Command": " ".join(cmd),
                "Experiment Base": exe_exp_base,
                "YAML Config": exe_yaml_path,
            },
            header="Final Training Command & Directory",
        )
        verify_python_env(args, exp_config)
        subprocess.run(cmd, check=True, cwd=os.path.abspath("./"), env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running subprocess: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
