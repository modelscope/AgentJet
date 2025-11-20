import subprocess
import argparse
import shutil
import yaml
import time
import sys
import os
from loguru import logger
from astune.utils.pty import pty_launch
from astune.utils.cleaner import fast_kill_by_keyword_bash
from astune.utils.smart_daemon import LaunchCommandWhenAbsent
from astune.utils.config_utils import expand_astune_hierarchical_config, read_astune_hierarchical_config, align_parameters
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="BA Launcher")
    parser.add_argument(
        "--backbone",
        type=str,
        default="trinity",
        required=False,
        help="verl or trinity or debug",
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
        default="launcher_record",
        required=False,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="",
        required=False,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--with-exp-maker",
        action="store_true",
        default=False,
        help="Launch exp maker",
    )
    parser.add_argument("--with-ray", action="store_true", default=False, help="Launch ray")
    parser.add_argument(
        "--with-appworld",
        action="store_true",
        default=False,
        help="Launch appworld",
    )
    parser.add_argument(
        "--with-webshop",
        action="store_true",
        default=False,
        help="Launch webshop",
    )
    parser.add_argument("--with-bfcl", action="store_true", default=False, help="Launch bfcl")
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
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip launching services and training (test/smoke mode)",
    )
    parser.add_argument("--reboot", action="store_true", default=False, help="reboot flag")
    parser.add_argument(
        "--kill",
        type=str,
        default="",
        required=False,
        help="list of keywords for killing processes",
    )
    return parser.parse_args()



def check_debugpy_version():
    try:
        import debugpy
    except ImportError:
        raise RuntimeError(
            "Module 'debugpy>=1.8.0' cannot be loaded. "
            "Ray Debugpy Debugger will not work without 'debugpy>=1.8.0' installed. "
            "Install this module using 'pip install debugpy>=1.8.0'"
        )
    version = getattr(debugpy, "__version__", "0.0.0")
    from packaging import version as packaging_version
    if packaging_version.parse(version) < packaging_version.parse("1.8.0"):
        raise RuntimeError(
            f"debugpy version {version} is too old. "
            "Ray Debugpy Debugger requires 'debugpy>=1.8.0'. "
            "Upgrade using 'pip install debugpy>=1.8.0'"
        )
    logger.info(f"✓ debugpy version {version} meets requirement (>=1.8.0)")



def prepare_experiment_config(yaml_path, exp_dir, backbone):
    """
    Prepare experiment configuration by reading YAML, setting up backup directories,
    and copying necessary files for the experiment.

    Args:
        yaml_path: Path to the YAML configuration file
        exp_dir: Directory where experiment artifacts and backups should be stored
        backbone: Backbone identifier that controls config munging

    Returns:
        tuple: (yaml_backup_dst, exe_exp_base, exp_name, config_final)
    """
    assert yaml_path.endswith(".yaml"), "Configuration file must be a YAML file"
    exp_base = os.path.dirname(yaml_path)

    if not os.path.exists(exp_base):
        raise FileNotFoundError(f"Configuration file not found: {exp_base}")

    ## 0. read yaml & get experiment_name
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    exp_name = config.get("astune").get("experiment_name")
    if exp_name is None or exp_name == "read_yaml_name":
        if exp_name is not None:
            exp_name = exp_name.replace("|", "-")
        exp_name = os.path.basename(yaml_path).replace(".yaml", "")
    else:
        exp_name = exp_name.replace("|", "-")

    backup_dir = os.path.join(exp_dir, exp_name, "backup")
    yaml_backup_dst = os.path.join(exp_dir, exp_name, "yaml_backup.yaml")
    exe_exp_base = os.path.dirname(yaml_backup_dst)
    logger.info("----------------------------------------")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Experiment Backup Dir: {backup_dir}")
    logger.info(f"Experiment Yaml Dir: {yaml_backup_dst}")
    logger.info("----------------------------------------")

    ## 1. check exp_base/backup exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    else:
        total_seconds = 5
        for i in range(total_seconds):
            logger.warning(
                f"Warning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds..."
            )
            time.sleep(1)

    ## 2. copy files to backup
    BACK_TARGETS = os.environ.get("BACK_TARGETS", "").split(",")
    BACK_TARGETS = [p for p in BACK_TARGETS if os.path.exists(p)]

    for backup_target in BACK_TARGETS:
        logger.info(
            f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}"
        )
        shutil.copytree(
            backup_target,
            os.path.join(backup_dir, os.path.basename(backup_target)),
            dirs_exist_ok=True,
        )

    ## 3. copy yaml to backup
    yaml_backup_src = yaml_path
    shutil.copyfile(yaml_backup_src, yaml_backup_dst)

    ## 4. edit new yaml
    config = read_astune_hierarchical_config(yaml_backup_dst, exp_name, backbone, write_to=yaml_backup_dst)
    config_final = expand_astune_hierarchical_config(config, write_to=yaml_backup_dst)

    return yaml_backup_dst, exe_exp_base, exp_name, config_final


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
        pass


def start_ray_service(args, env):
    """
    Start a Ray service with appropriate configuration.

    Args:
        args: Command line arguments containing debug settings
    """
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[f"source ./.venv/bin/activate && ray start --head --block"],
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
    TRINITY_BOOT_YAML = \
        "astune/default_config/trinity/trinity_launch.yaml"  # THIS FILE IS READ ONLY, and ALWAYS FIXED
    TRINITY_CONFIG_AUTO_CONVERSION = \
        "astune/default_config/trinity/config_auto_convertion_trinity.json"
    VERL_CONFIG_AUTO_CONVERSION = \
        "astune/default_config/verl/config_auto_convertion_verl.json"


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
        subprocess.run(cmd, check=True, cwd=os.path.abspath("./"), env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running subprocess: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    args = parse_args()

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
        return  # Exit after killing processes

    # Initialize variables with default values to avoid "possibly unbound" errors
    backbone_target = "astune.main_trinity"  # Default to trinity
    main_yaml_fp = None
    exe_exp_base = None
    exp_name = None
    env = os.environ.copy()

    if args.backbone == "verl":
        backbone_target = "astune.main_verl"
    if args.backbone == "debug":
        backbone_target = "astune.main_vllm"
    if args.backbone == "trinity":
        backbone_target = "astune.main_trinity"

    exp_config = None
    exp_dir = args.exp_dir or "launcher_record"
    if args.conf:
        yaml_path = args.conf
        (
            main_yaml_fp,
            exe_exp_base,
            exp_name,
            exp_config,
        ) = prepare_experiment_config(yaml_path, exp_dir, args.backbone)

    if args.db:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.db
        env["RAY_record_task_actor_creation_sites"] = "true"
        logger.warning("Debug mode is ON")
    else:
        logger.warning("Debug mode is OFF")

    if args.backbone == "trinity":
        env["ASTUNE_CONFIG_REDIRECT"] = main_yaml_fp  # type: ignore
    if args.backbone == "debug":
        env["ASTUNE_DEBUG"] = "1"  # type: ignore

    if args.with_ray:
        start_ray_service(args, env)

    if args.with_exp_maker:
        pty_launch("exp_maker", success_std_string="Uvicorn running on")

    if args.with_appworld:
        pty_launch("appworld")

    if args.with_crafters:
        pty_launch("crafters")

    if args.with_webshop:
        pty_launch("webshop")

    if args.with_bfcl:
        pty_launch("bfcl")

    if args.with_logview:
        launch_logview(exp_name)

    if args.conf and main_yaml_fp and exe_exp_base and exp_config:
        if args.dry_run:
            logger.info("Dry-run enabled: skipping training process launch.")
            return {
                "yaml": main_yaml_fp,
                "exp_base": exe_exp_base,
                "exp_yaml_name": os.path.basename(main_yaml_fp),
                "exp_name": exp_config.get("astune", {}).get("experiment_name"),
            }
        else:
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
