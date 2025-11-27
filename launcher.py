import argparse
import os
import shutil
import subprocess
import sys
import time

import yaml
from dotenv import load_dotenv
from loguru import logger

from astuner.utils.cleaner import fast_kill_by_keyword_bash
from astuner.utils.config_utils import prepare_experiment_config
from astuner.utils.launch_utils import (
    execute_training_process,
    launch_logview,
    start_ray_service,
)
from astuner.utils.pty import pty_launch
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent

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
        "--debug",
        type=str,
        default="",
        required=False,
        help="Path to configuration file",
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


def check_avail_gpu(min_free_ratio: float = 0.95):
    """
    Ensure there is at least one GPU and all GPUs have >= min_free_ratio free memory.

    Uses `nvidia-smi` to query total and used memory for each GPU.
    Raises RuntimeError if no GPU is found or any GPU violates the free ratio threshold.
    """
    try:
        # Query GPU memory via nvidia-smi; output in MiB
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. NVIDIA drivers/GPU may be unavailable.")

    if result.returncode != 0:
        raise RuntimeError(f"Failed to query GPUs via nvidia-smi: {result.stderr.strip()}")

    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("No GPUs detected by nvidia-smi.")

    violations = []
    for idx, line in enumerate(lines):
        # Expected format: "<name>, <total>, <used>"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            violations.append((idx, "parse-error", line))
            continue
        name, total_str, used_str = parts[0], parts[1], parts[2]
        try:
            total = float(total_str)
            used = float(used_str)
        except ValueError:
            violations.append((idx, "parse-error", line))
            continue
        free = max(total - used, 0.0)
        free_ratio = free / total if total > 0 else 0.0
        logger.info(
            f"GPU {idx} ({name}): total={total:.0f} MiB, used={used:.0f} MiB, free_ratio={free_ratio:.3f}"
        )
        if free_ratio < min_free_ratio:
            violations.append((idx, name, f"free_ratio={free_ratio:.3f} < {min_free_ratio:.3f}"))

    if violations:
        details = "; ".join([f"GPU {i} ({n}): {msg}" for i, n, msg in violations])
        raise RuntimeError(
            "GPU memory check failed: all GPUs must have >= "
            f"{int(min_free_ratio*100)}% free. Violations: {details}"
        )
    logger.info(
        f"✓ GPU check passed: {len(lines)} GPUs, all >= {int(min_free_ratio*100)}% free memory"
    )


def main():
    args = parse_args()

    # Enforce GPU availability and free memory threshold before proceeding
    check_avail_gpu(min_free_ratio=0.95)

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
    backbone_target = "astuner.main_trinity"  # Default to trinity
    main_yaml_fp = None
    exe_exp_base = None
    exp_name = None
    env = os.environ.copy()

    if args.backbone == "verl":
        backbone_target = "astuner.main_verl"
    if args.backbone == "debug":
        backbone_target = "astuner.main_vllm"
    if args.backbone == "trinity":
        backbone_target = "astuner.main_trinity"

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

    if args.debug:
        env["RAY_DEBUG_POST_MORTEM"] = "1"
        env["DEBUG_TAGS"] = args.debug
        env["RAY_record_task_actor_creation_sites"] = "true"
        assert exp_config["astuner"]["rollout"]["max_env_worker"] <= 4, "parallel worker too many for debugging mode"  # type: ignore
        logger.warning("Debug mode is ON")
    else:
        logger.warning("Debug mode is OFF")
        if args.conf:
            assert exp_config["astuner"]["rollout"]["max_env_worker"] > 4, "parallel worker too few"  # type: ignore

    if args.backbone == "trinity":
        env["ASTUNER_CONFIG_REDIRECT"] = main_yaml_fp  # type: ignore
    if args.backbone == "debug":
        env["ASTUNER_DEBUG"] = "1"  # type: ignore

    if args.with_ray:
        start_ray_service(args, env)

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
                "exp_name": exp_config.get("astuner", {}).get("experiment_name"),
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
