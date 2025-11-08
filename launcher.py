import subprocess
import argparse
import shutil
import time
import sys
import os
from dotenv import load_dotenv; load_dotenv()
from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent




def parse_args():
    parser = argparse.ArgumentParser(description='BA Launcher')

    parser.add_argument('--backbone',
        type=str,
        default="trinity",
        required=False,
        help='verl or trinity'
    )
    parser.add_argument('--conf',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--db',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--with-ray',
        action='store_true',
        default=False,
        help='Launch ray'
    )
    parser.add_argument('--with-appworld',
        action='store_true',
        default=False,
        help='Launch appworld'
    )
    parser.add_argument('--with-webshop',
        action='store_true',
        default=False,
        help='Launch webshop'
    )
    parser.add_argument('--with-logview',
        action='store_true',
        default=False,
        help='Launch logview'
    )
    parser.add_argument('--with-crafters',
        action='store_true',
        default=False,
        help='Launch Crafters Env Simulation'
    )
    parser.add_argument('--reboot',
        action='store_true',
        default=False,
        help='reboot flag'
    )

    return parser.parse_args()

def check_debugpy_version():
    """
    检查 debugpy 模块版本是否 >= 1.8.0
    如果未安装或版本过低，抛出 RuntimeError
    """
    try:
        import debugpy
    except ImportError:
        raise RuntimeError(
            "Module 'debugpy>=1.8.0' cannot be loaded. "
            "Ray Debugpy Debugger will not work without 'debugpy>=1.8.0' installed. "
            "Install this module using 'pip install debugpy>=1.8.0'"
        )

    # 检查版本
    version = getattr(debugpy, '__version__', '0.0.0')
    from packaging import version as packaging_version

    if packaging_version.parse(version) < packaging_version.parse('1.8.0'):
        raise RuntimeError(
            f"debugpy version {version} is too old. "
            "Ray Debugpy Debugger requires 'debugpy>=1.8.0'. "
            "Upgrade using 'pip install debugpy>=1.8.0'"
        )

    print(f"✓ debugpy version {version} meets requirement (>=1.8.0)")

check_debugpy_version()

def pty_launch(service_name: str, success_std_string="Starting server on"):
    service_path = os.environ.get(f'{service_name.upper()}_PATH')
    service_script = os.environ.get(f'{service_name.upper()}_SCRIPT')
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[service_script],
        dir=service_path,
        tag="appworld_env_service",
        use_pty=True
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string=success_std_string,
    )

def prepare_experiment_config(yaml_path, args):
    """
    Prepare experiment configuration by reading YAML, setting up backup directories,
    and copying necessary files for the experiment.

    Args:
        yaml_path: Path to the YAML configuration file
        args: Command line arguments

    Returns:
        tuple: (yaml_backup_dst, exe_exp_base, exe_yaml_path, exp_name)
    """
    assert yaml_path.endswith('.yaml'), "Configuration file must be a YAML file"
    exp_base = os.path.dirname(yaml_path)

    if not os.path.exists(exp_base):
        raise FileNotFoundError(f"Configuration file not found: {exp_base}")

    ## 0. read yaml (get trainer.experiment_name)
    import yaml
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    exp_name = config.get('trainer').get('experiment_name')
    if exp_name is None or exp_name == 'read_yaml_name':
        if exp_name is not None: exp_name = exp_name.replace('|', '-')
        exp_name = os.path.basename(yaml_path).replace('.yaml', '')
    else:
        exp_name = exp_name.replace('|', '-')

    print('----------------------------------------')
    backup_dir = os.path.join('launcher_record', exp_name, 'backup')
    yaml_backup_dst = os.path.join('launcher_record', exp_name, 'yaml_backup.yaml')
    exe_yaml_path = yaml_backup_dst
    exe_exp_base = os.path.dirname(yaml_backup_dst)
    print('Experiment Name:', exp_name)
    print('Experiment Backup Dir:', backup_dir)
    print('Experiment Yaml Dir:', yaml_backup_dst)
    print('----------------------------------------')
    time.sleep(2)

    ## 1. check exp_base/backup exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    else:
        total_seconds = 5
        for i in range(total_seconds):
            print(f"\rWarning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...", end="", flush=True)
            time.sleep(1)

    ## 2. copy files to backup
    BACK_TARGETS = os.environ.get('BACK_TARGETS', '').split(',')

    for backup_target in BACK_TARGETS:
        print(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
        shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

    ## 3. copy yaml to backup
    yaml_backup_src = yaml_path
    shutil.copyfile(yaml_backup_src, yaml_backup_dst)

    ## 4. edit new yaml
    yaml_path = yaml_backup_dst
    # now, replace the trainer.experiment_name
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    if args.backbone != "trinity":
        config['trainer']['experiment_name'] = exp_name
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file)

    return yaml_backup_dst, exe_exp_base, exe_yaml_path, exp_name


def main():
    args = parse_args()

    # Initialize variables with default values to avoid "possibly unbound" errors
    backbone_target = "agentopia.main_trinity"  # Default to trinity
    yaml_backup_dst = None
    exe_exp_base = None
    exe_yaml_path = None
    exp_name = None
    env = os.environ.copy()

    if args.backbone == "verl":
        backbone_target = "agentopia.main_verl"
    if args.backbone == "debug":
        backbone_target = "agentopia.main_vllm"
    if args.backbone == "trinity":
        backbone_target = "agentopia.main_trinity"

    if args.conf:
        yaml_path = args.conf
        yaml_backup_dst, exe_exp_base, exe_yaml_path, exp_name = prepare_experiment_config(yaml_path, args)
    else:
        assert args.with_appworld or args.with_webshop or args.with_logview or args.with_crafters, "You must at least do something."

    if args.db:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.db
        env["RAY_record_task_actor_creation_sites"] = "true"
        print("Debug mode is ON")
    else:
        print("Debug mode is OFF")

    if args.with_ray:
        ray_env = {}
        if args.db:
            ray_env["RAY_DEBUG_POST_MORTEM"] = "1"
            ray_env["DEBUG_TAGS"] = args.db
            ray_env["RAY_record_task_actor_creation_sites"] =  "true"
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                f"source ./.venv/bin/activate && ray start --head && sleep infinity"
            ],
            dir='./',
            tag="ray_service",
            use_pty=True
        )
        companion.launch(
            launch_wait_time=1800,
            success_std_string="Ray runtime started",
            env_dict=ray_env,
        )

    if args.with_exp_maker:
        # test done
        pty_launch("exp_maker", success_std_string="Uvicorn running on")

    if args.with_appworld:
        # test done
        pty_launch("appworld")

    if args.with_crafters:
        # test done
        pty_launch("crafters")

    if args.with_webshop:
        # not tesed
        pty_launch("webshop")

    if args.with_bfcl:
        pty_launch("bfcl")

    if args.with_logview:
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                '-m',
                'web_display.start_web',
            ],
            dir='./',
            tag="logview"
        )
        companion.launch(launch_wait_time=1800,success_std_string="Uvicorn running on", env_dict={})
        time.sleep(2.5)
        try:
            import webbrowser
            from datetime import datetime
            # Use default experiment name if not set
            log_exp_name = exp_name if exp_name else "default_experiment"
            final_log_path = os.path.join("experiments", log_exp_name, "trace_rollout", datetime.now().strftime("%Y_%m_%d_%H_%M"))
            # make dir
            os.makedirs(final_log_path)
            webbrowser.open("http://127.0.0.1:8181/"+"?path="+os.path.abspath(final_log_path))
        except Exception as e:
            print(f"Error opening web browser: {e}")
            pass

    if args.conf and yaml_backup_dst and exe_exp_base and exe_yaml_path:
        execute_training_process(args, backbone_target, yaml_backup_dst, exe_exp_base, exe_yaml_path, env)

def execute_training_process(args, backbone_target, yaml_backup_dst, exe_exp_base, exe_yaml_path, env):
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
    # let's begin the training process
    if args.backbone == "trinity":
        cmd = [
            sys.executable,
            '-m', backbone_target,
            'run', '--config', yaml_backup_dst
        ]
    else:
        cmd = [
            sys.executable,
            '-m', backbone_target,
            '--config-path', os.path.abspath(exe_exp_base),
            '--config-name', os.path.basename(exe_yaml_path),
        ]

    if args.with_logview:
        env.update({
            'BEST_LOGGER_WEB_SERVICE_URL':
            os.environ.get('BEST_LOGGER_WEB_SERVICE_URL', 'http://127.0.0.1:8181/')
        })

    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running subprocess: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
