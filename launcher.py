import subprocess
import argparse
import shutil
import yaml
import time
import sys
import os
from loguru import logger
from dotenv import load_dotenv; load_dotenv()
from astune.utils.cleaner import _fast_kill_by_keyword_bash
from astune.utils.smart_daemon import LaunchCommandWhenAbsent
from astune.utils.config_utils import read_astune_config, dump_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description='BA Launcher')

    parser.add_argument('--backbone',
        type=str,
        default="trinity",
        required=False,
        help='verl or trinity or debug'
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
    parser.add_argument('--with-exp-maker',
        action='store_true',
        default=False,
        help='Launch exp maker'
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
    parser.add_argument('--with-bfcl',
        action='store_true',
        default=False,
        help='Launch bfcl'
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
    parser.add_argument('--kill',
        type=str,
        default="",
        required=False,
        help='list of keywords for killing processes'
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
    version = getattr(debugpy, '__version__', '0.0.0')
    from packaging import version as packaging_version
    if packaging_version.parse(version) < packaging_version.parse('1.8.0'):
        raise RuntimeError(
            f"debugpy version {version} is too old. "
            "Ray Debugpy Debugger requires 'debugpy>=1.8.0'. "
            "Upgrade using 'pip install debugpy>=1.8.0'"
        )
    logger.info(f"✓ debugpy version {version} meets requirement (>=1.8.0)")


def pty_launch(service_name: str, success_std_string="Starting server on"):
    service_path = os.environ.get(f'{service_name.upper()}_PATH')
    service_script = os.environ.get(f'{service_name.upper()}_SCRIPT')
    if service_path is None or service_script is None:
        raise ValueError(f"Environment variables for {service_name} not properly set.")
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

    ## 0. read yaml (get astune.experiment_name)
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    exp_name = config.get('astune').get('experiment_name')
    if exp_name is None or exp_name == 'read_yaml_name':
        if exp_name is not None: exp_name = exp_name.replace('|', '-')
        exp_name = os.path.basename(yaml_path).replace('.yaml', '')
    else:
        exp_name = exp_name.replace('|', '-')

    logger.info('----------------------------------------')
    backup_dir = os.path.join('launcher_record', exp_name, 'backup')
    yaml_backup_dst = os.path.join('launcher_record', exp_name, 'yaml_backup.yaml')
    exe_yaml_path = yaml_backup_dst
    exe_exp_base = os.path.dirname(yaml_backup_dst)
    logger.info(f'Experiment Name: {exp_name}')
    logger.info(f'Experiment Backup Dir: {backup_dir}')
    logger.info(f'Experiment Yaml Dir: {yaml_backup_dst}')
    logger.info('----------------------------------------')
    time.sleep(2)

    ## 1. check exp_base/backup exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    else:
        total_seconds = 5
        for i in range(total_seconds):
            logger.warning(f"Warning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...")
            time.sleep(1)

    ## 2. copy files to backup
    BACK_TARGETS = os.environ.get('BACK_TARGETS', '').split(',')
    BACK_TARGETS = [p for p in BACK_TARGETS if os.path.exists(p)]

    for backup_target in BACK_TARGETS:
        logger.info(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
        shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

    ## 3. copy yaml to backup
    yaml_backup_src = yaml_path
    shutil.copyfile(yaml_backup_src, yaml_backup_dst)

    ## 4. edit new yaml
    yaml_path = yaml_backup_dst
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    config['astune']['experiment_name'] = exp_name

    # remove extra config of verl for trinity
    if args.backbone == "debug":
        config['defaults'].remove('ppo_trainer')
        config['defaults'].remove('trinity_default')
        config['hydra']['searchpath'].remove('file://astune/default_config/trinity')
        config['hydra']['searchpath'].remove('file://external/verl/verl/trainer/config')
    # remove extra config of verl for trinity
    if args.backbone == "trinity":
        config['defaults'].remove('ppo_trainer')
        config['defaults'].remove('verl_default')
        config['hydra']['searchpath'].remove('file://external/verl/verl/trainer/config')
        config['hydra']['searchpath'].remove('file://astune/default_config/verl')
    # remove extra config of trinity for verl
    if args.backbone == "verl": #  or args.backbone == "debug"
        config['defaults'].remove('trinity_default')
        config['hydra']['searchpath'].remove('file://astune/default_config/trinity')
    # yaml dump
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file)

    # read everything
    full_config = read_astune_config(yaml_path)
    yaml_path = full_config_path = dump_yaml_config(full_config, yaml_fp=yaml_path)


    # put inherit info back
    with open(yaml_path, 'r') as file:
        config_final = yaml.safe_load(file)
    config_final['defaults'] = config['defaults']
    config_final['hydra'] = config['hydra']

    # write inherit info back
    with open(yaml_path, 'w') as file:
        yaml.dump(config_final, file)

    return yaml_path, exe_exp_base, exe_yaml_path, exp_name, config

def launch_logview(exp_name=None):
    """
    Launch the log viewer service and open the web browser to view logs.

    Args:
        exp_name: Optional experiment name. If not provided, "default_experiment" is used.
    """
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable,
            '-m',
            'web_display.start_web',
        ],
        dir='./',
        tag="logview"
    )
    companion.launch(launch_wait_time=1800, success_std_string="Uvicorn running on", env_dict={})
    time.sleep(2.5)
    try:
        import webbrowser
        from datetime import datetime
        # Use default experiment name if not set
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
        full_argument_list=[
            f"source ./.venv/bin/activate && ray start --head --block"
        ],
        dir='./',
        tag="ray_service",
        use_pty=True
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Ray runtime started",
        env_dict=env,
    )

import yaml

def align_parameters(from_config_fp, to_config_fp, convertion_json_fg):
    # read yaml files
    with open(from_config_fp, 'r') as file:
        from_config = yaml.safe_load(file)
    with open(to_config_fp, 'r') as file:
        to_config = yaml.safe_load(file)
    # read convertion json
    import json
    with open(convertion_json_fg, 'r') as file:
        convertion_json = json.load(file)

    logger.success("----------------------------------------------------")

    for from_key, to_keys in convertion_json.items():
        # get value from from_config
        keys = from_key.split('.')
        value = from_config
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        if value is None:
            logger.warning(f"[Warning]: Cannot find value for key: {from_key} in {from_config_fp}, skip aligning {to_keys}")
            continue

        to_keys = to_keys if isinstance(to_keys, list) else [to_keys]
        for to_key in to_keys:
            keys = to_key.split('.')
            sub_config = to_config
            for key in keys[:-1]:
                if key not in sub_config:
                    sub_config[key] = {}
                sub_config = sub_config[key]
            sub_config[keys[-1]] = value
            logger.success(f"[Note]: Aligned parameter from [{from_key}] to [{to_key}] with value: [{value}]")
            time.sleep(0.1)

    logger.success("----------------------------------------------------")

    if 'trinity' in from_config:
        trinity_config = from_config['trinity']
        def recursive_copy(src_dict, dst_dict, parent_key=""):
            for key, value in src_dict.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    if key not in dst_dict:
                        dst_dict[key] = {}
                    recursive_copy(value, dst_dict[key], full_key)
                else:
                    dst_dict[key] = value
                    logger.info(f"[Note]: Aligned parameter from [trinity.{full_key}] to [{full_key}] with value: [{value}]")
        recursive_copy(trinity_config, to_config)

    logger.success("----------------------------------------------------")


    # save to_config_fp
    with open(to_config_fp, 'w') as file:
        yaml.dump(to_config, file)
    logger.success(f"Saved aligned configuration to {to_config_fp}")




def execute_training_process(args, backbone_target, yaml_backup_dst, exe_exp_base, exe_yaml_path, env, exp_config):
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
        # replace boot yaml
        trinity_boot_yaml = "astune/default_config/trinity_launch.yaml" # THIS ONE IS READ ONLY, and ALWAYS FIXED
        align_parameters(yaml_backup_dst, trinity_boot_yaml, 'astune/default_config/trinity/config_auto_convertion_trinity.json')
        cmd = [
            sys.executable,
            '-m', backbone_target,
            'run', '--config', trinity_boot_yaml
        ]
    else:
        align_parameters(yaml_backup_dst, yaml_backup_dst, 'astune/default_config/verl/config_auto_convertion_verl.json')
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
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
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
        for keyword in args.kill.split('|'):
            logger.info(f"Killing processes matching keyword: {keyword}")
            killed_pids = _fast_kill_by_keyword_bash(keyword)
            if killed_pids:
                logger.success(f"Successfully killed processes with PIDs: {killed_pids}")
            else:
                logger.warning(f"No processes found matching keyword: {keyword}")
        return  # Exit after killing processes

    # Initialize variables with default values to avoid "possibly unbound" errors
    backbone_target = "astune.main_trinity"  # Default to trinity
    yaml_backup_dst = None
    exe_exp_base = None
    exe_yaml_path = None
    exp_name = None
    env = os.environ.copy()

    if args.backbone == "verl":
        backbone_target = "astune.main_verl"
    if args.backbone == "debug":
        backbone_target = "astune.main_vllm"
    if args.backbone == "trinity":
        backbone_target = "astune.main_trinity"

    exp_config = None
    if args.conf:
        yaml_path = args.conf
        yaml_backup_dst, exe_exp_base, exe_yaml_path, exp_name, exp_config = prepare_experiment_config(yaml_path, args)

    if args.db:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.db
        env["RAY_record_task_actor_creation_sites"] = "true"
        logger.warning("Debug mode is ON")
    else:
        logger.warning("Debug mode is OFF")

    if args.backbone == "trinity":
        env['ASTUNE_CONFIG_REDIRECT'] = yaml_backup_dst # type: ignore
    if args.backbone == "debug":
        env['ASTUNE_DEBUG'] = '1'  # type: ignore

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

    if args.conf and yaml_backup_dst and exe_exp_base and exe_yaml_path:
        execute_training_process(args, backbone_target, yaml_backup_dst, exe_exp_base, exe_yaml_path, env, exp_config)

if __name__ == "__main__":
    check_debugpy_version()
    main()
