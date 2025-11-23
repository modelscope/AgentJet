import os
import subprocess
import sys

TEST_TARGET = "astune_tests/test_apply_chat_template/test.yaml"
cmd = [sys.executable, "launcher.py", "--conf", TEST_TARGET, "--backbone", "verl"]
subprocess.run(cmd, check=True, cwd=os.path.abspath("./"), env=os.environ)

# python launcher.py --kill="python|ray|vllm|VLLM" && ray stop
# python launcher.py --conf astune_tests/test_apply_chat_template/test.yaml --backbone verl

from astune.utils.smart_daemon import LaunchCommandWhenAbsent
service_path = os.path.dirname(__file__)
companion = LaunchCommandWhenAbsent(
    full_argument_list=cmd,
    dir=service_path,
    tag="appworld_env_service",
    use_pty=True,
)
try:
    companion.launch(
        launch_wait_time=1800,
        success_std_string="GoodbyeException",
    )
except Exception as e:
    print(f"Error launching companion: {e}")
    sys.exit(1)