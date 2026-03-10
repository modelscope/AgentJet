#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Episode runner for SkillsBench tasks with OpenCode agent.
Runs a single training episode and computes reward.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Tuple


# Special version of docker-compose-base.yaml that should be used
SPECIAL_DOCKER_COMPOSE_CONTENT = """services:
  main:
    environment:
      - OPENCODE_CONFIG_CONTENT=${OPENCODE_CONFIG_CONTENT}
    volumes:
      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
    deploy:
      resources:
        limits:
          cpus: ${CPUS}
          memory: ${MEMORY}
    network_mode: host
"""


def check_and_fix_docker_compose() -> bool:
    """
    Check if harbor's docker-compose-base.yaml is the special version.
    If not, update it to the special version.

    Returns:
        bool: True if file was modified, False if already correct
    """
    try:
        # Find harbor installation path
        result = subprocess.run(
            ["/root/.local/share/uv/tools/harbor/bin/python", "-c",
             "import harbor; import os; print(os.path.dirname(harbor.__file__))"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print("Warning: Could not locate harbor installation, skipping docker-compose check")
            return False

        harbor_path = result.stdout.strip()
        docker_compose_path = Path(harbor_path) / "environments" / "docker" / "docker-compose-base.yaml"

        if not docker_compose_path.exists():
            print(f"Warning: docker-compose-base.yaml not found at {docker_compose_path}")
            return False

        # Read current content
        current_content = docker_compose_path.read_text()

        # Compare with special version (strip whitespace for comparison)
        if current_content.strip() == SPECIAL_DOCKER_COMPOSE_CONTENT.strip():
            print(f"✓ docker-compose-base.yaml is already the special version")
            return False

        # Update to special version
        print(f"! docker-compose-base.yaml is NOT the special version")
        print(f"  Updating {docker_compose_path} to special version...")
        docker_compose_path.write_text(SPECIAL_DOCKER_COMPOSE_CONTENT)
        print(f"✓ Updated docker-compose-base.yaml to special version")
        return True

    except Exception as e:
        print(f"Warning: Error checking docker-compose-base.yaml: {e}")
        return False


def run_episode(task_id: str, task_path: str, api_key: str, base_url: str, model: str = "qwen/qwen3-max") -> Tuple[float, dict]:
    """
    Run a single episode for a SkillsBench task using OpenCode agent.

    Args:
        task_id: Unique identifier for the task
        task_path: Full path to the task directory
        api_key: API key for the model provider
        base_url: Base URL for the model provider
        model: Model identifier (ignored, hardcoded to use huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507)

    Returns:
        Tuple of (reward, metadata):
        - reward: float between 0 and 1 (1 = pass, 0 = fail)
        - metadata: dict with execution details
    """
    # Hardcoded model - do not change
    actual_model_with_provider = "huggingface/Qwen25-14B"
    actual_model_name = "Qwen25-14B"

    # Generate a unique job name to avoid concurrent conflicts
    # Format: {task_id}_{uuid} to make it both human-readable and unique
    job_name = f"{task_id}_{uuid.uuid4().hex[:12]}"

    print(f"\n{'='*60}")
    print(f"Running episode for task: {task_id}")
    print(f"Job name: {job_name}")
    print(f"Task path: {task_path}")
    print(f"Model: {actual_model_with_provider}")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}..." if len(api_key) > 10 else f"API Key: {api_key}")
    print(f"{'='*60}\n")

    # Check and fix docker-compose-base.yaml before running
    print("Checking harbor docker-compose-base.yaml configuration...")
    check_and_fix_docker_compose()
    print()

    # Set up environment variables for the agent
    env = os.environ.copy()

    # Create OPENCODE_CONFIG_CONTENT with the provided api_key and base_url
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": actual_model_with_provider,
        "provider": {
            "huggingface": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "vLLM Provider",
                "options": {
                    "baseURL": base_url,
                    "apiKey": api_key
                },
                "models": {
                    actual_model_name: {
                        "name": "vLLM Model",
                        "limit": {
                            "context": 20*1000,
                            "output": 10*1000
                        }
                    }
                }
            }
        }
    }

    print(f"export OPENCODE_CONFIG_CONTENT='{json.dumps(opencode_config)}'")
    import time
    time.sleep(10000)

    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_config)
    print(f"Set OPENCODE_CONFIG_CONTENT environment variable")
    print(f"Config: {json.dumps(opencode_config, indent=2)}\n")

    # Construct harbor run command
    # harbor run -p <task_path> -a opencode -m <model> --job-name <unique_job_name>
    cmd = [
        "harbor", "run",
        "-p", task_path,
        "-a", "opencode",
        "-m", actual_model_with_provider,
        "--job-name", job_name,
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"\nStarting execution...\n")

    metadata = {
        "task_id": task_id,
        "task_path": task_path,
        "job_name": job_name,
        "model": actual_model_with_provider,
        "base_url": base_url,
        "success": False,
        "reward": 0.0,
        "error": None,
    }

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        print(f"STDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        print(f"\nReturn code: {result.returncode}")

        # Parse the results
        # Harbor writes results to jobs/<job_name>/<task_id>/verifier/reward.txt
        # We use the job_name to locate the specific job directory
        reward = parse_harbor_results(result.stdout, result.stderr, result.returncode, job_name)

        metadata["success"] = (reward > 0)
        metadata["reward"] = reward
        metadata["stdout"] = result.stdout
        metadata["stderr"] = result.stderr
        metadata["returncode"] = result.returncode

        print(f"\n{'='*60}")
        print(f"Episode completed")
        print(f"Reward: {reward}")
        print(f"Success: {metadata['success']}")
        print(f"{'='*60}\n")

        return reward, metadata

    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Episode timed out after 3600 seconds")
        metadata["error"] = "timeout"
        metadata["timeout"] = True
        return 0.0, metadata

    except Exception as e:
        print(f"ERROR: Exception during episode execution: {e}")
        metadata["error"] = str(e)
        return 0.0, metadata


def parse_harbor_results(stdout: str, stderr: str, returncode: int, job_name: str) -> float:
    """
    Parse Harbor execution results to extract reward.

    Harbor typically outputs the results location and we can read reward.txt
    from the jobs directory.

    Args:
        stdout: Standard output from harbor run
        stderr: Standard error from harbor run
        returncode: Return code from harbor run
        job_name: The unique job name used for this run

    Returns:
        float: reward value (0.0 or 1.0)
    """
    # First priority: Use the job_name to directly locate the result directory
    # Harbor creates directories as: jobs/<job_name>/<task_name>/verifier/reward.txt
    jobs_dir = Path("jobs")
    print(f"Searching for results in job directory: {job_name}")

    if not jobs_dir.exists():
        print(f"WARNING: Jobs directory does not exist: {jobs_dir}")
    else:
        job_dir = jobs_dir / job_name
        print(f"Looking for job directory: {job_dir}")

        if not job_dir.exists():
            print(f"WARNING: Job directory does not exist: {job_dir}")
            print(f"Available job directories:")
            for d in sorted(jobs_dir.iterdir())[-5:]:  # Show last 5
                print(f"  - {d.name}")
        else:
            print(f"Found job directory: {job_dir}")
            # Look for verifier/reward.txt in this specific job directory
            reward_files = list(job_dir.rglob("reward.txt"))
            print(f"Found {len(reward_files)} reward.txt file(s) in job directory")

            for reward_file in reward_files:
                print(f"Checking reward file: {reward_file}")
                if "verifier" in str(reward_file):
                    try:
                        reward_value = float(reward_file.read_text().strip())
                        print(f"✓ Successfully read reward from {reward_file}: {reward_value}")
                        return reward_value
                    except Exception as e:
                        print(f"ERROR: Failed to read reward file {reward_file}: {e}")
                else:
                    print(f"Skipping non-verifier reward file: {reward_file}")

    # Second priority: Look for results directory in stdout
    # Harbor typically prints something like "Results saved to: jobs/..."
    for line in stdout.split('\n'):
        if 'Results saved to:' in line or 'results' in line.lower():
            # Try to extract path
            parts = line.split()
            for part in parts:
                if 'jobs/' in part:
                    results_path = Path(part.strip())
                    reward_file = results_path / "verifier" / "reward.txt"
                    if reward_file.exists():
                        try:
                            reward_value = float(reward_file.read_text().strip())
                            print(f"Found reward in {reward_file}: {reward_value}")
                            return reward_value
                        except Exception as e:
                            print(f"Error reading reward file: {e}")

    # Fallback: Look for any jobs directory and find the most recent one
    # WARNING: This is unreliable in concurrent scenarios and should only be used as last resort
    print("WARNING: Falling back to searching by modification time (unreliable in concurrent scenarios)")
    if jobs_dir.exists():
        # Get all subdirectories sorted by modification time (most recent first)
        job_dirs = sorted(
            [d for d in jobs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Only check the most recent directory to avoid picking up concurrent runs
        for job_dir in job_dirs[:1]:  # Only check the most recent one
            print(f"Checking most recent job directory: {job_dir.name}")
            # Look for verifier/reward.txt in any subdirectory
            for reward_file in job_dir.rglob("reward.txt"):
                try:
                    reward_value = float(reward_file.read_text().strip())
                    print(f"Found reward in {reward_file}: {reward_value}")
                    return reward_value
                except Exception as e:
                    print(f"Error reading reward file {reward_file}: {e}")

    # If we can't find the reward file, check return code
    # Harbor typically returns 0 on success
    if returncode == 0:
        # Check if there are any test failures mentioned in output
        if "FAILED" in stdout or "FAILED" in stderr:
            print("Tests FAILED according to output")
            return 0.0
        elif "PASSED" in stdout or "passed" in stdout.lower():
            print("Tests PASSED according to output")
            return 1.0

    # Default to 0 if we can't determine success
    print("Could not determine reward, defaulting to 0.0")
    return 0.0


# Example usage:
# python3 tutorial/opencode_build_skillsbench/run_episode.py \
#   --task-id adaptive-cruise-control \
#   --task-path /root/AgentJet/tmp/skillsbench_swarm_test/tasks/adaptive-cruise-control \
#   --api-key "sk-123467" \
#   --base-url "http://127.0.0.1:2888/v1" \
#   --model "huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507" \
#   --output results.json


def main():
    parser = argparse.ArgumentParser(description="Run a single SkillsBench training episode")
    parser.add_argument("--task-id", required=True, help="Task identifier")
    parser.add_argument("--task-path", required=True, help="Path to task directory")
    parser.add_argument("--api-key", required=True, help="API key for the model provider")
    parser.add_argument("--base-url", required=True, help="Base URL for the model provider")
    parser.add_argument("--model", default="qwen/qwen3-max", help="Model identifier (ignored, hardcoded model will be used)")
    parser.add_argument("--output", help="Output file for results (JSON)")

    args = parser.parse_args()

    reward, metadata = run_episode(
        task_id=args.task_id,
        task_path=args.task_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    # Save results if output file specified
    if args.output:
        output_data = {
            "reward": reward,
            "metadata": metadata,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Exit with code based on success
    sys.exit(0 if reward > 0 else 1)


if __name__ == "__main__":
    main()
