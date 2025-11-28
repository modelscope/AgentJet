import os
import shutil
import subprocess
import sys
import time

from beast_logger import print_dict
from loguru import logger

from astuner.utils.config_utils import align_parameters
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent


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


def start_ray_service(args, env):
    """
    Start a Ray service with appropriate configuration.

    Args:
        args: Command line arguments containing debug settings
    """
    # Get the current Python interpreter directory
    python_dir = os.path.dirname(sys.executable)
    ray_path = os.path.join(python_dir, "ray")
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[f"{ray_path} start --head --block"],
        dir="./",
        tag="ray_service",
        use_pty=True,
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Ray runtime started",
        env_dict=env,
    )


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
    TRINITY_BOOT_YAML = "astuner/default_config/trinity/trinity_launch.yaml"  # THIS FILE IS READ ONLY, and ALWAYS FIXED
    TRINITY_CONFIG_AUTO_CONVERSION = (
        "astuner/default_config/trinity/config_auto_convertion_trinity.json"
    )
    VERL_CONFIG_AUTO_CONVERSION = "astuner/default_config/verl/config_auto_convertion_verl.json"

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
            }
        )
        subprocess.run(cmd, check=True, cwd=os.path.abspath("./"), env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running subprocess: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
