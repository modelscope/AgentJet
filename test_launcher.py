import os
import subprocess
import sys

from beast_logger import print_dict

TEST_TARGET = "astune_tests/test_apply_chat_template/test.yaml"
cmd = [sys.executable, "launcher.py", "--conf", TEST_TARGET, "--backbone", "verl"]

from astune.utils.smart_daemon import LaunchCommandWhenAbsent

service_path = os.path.dirname(__file__)
companion = LaunchCommandWhenAbsent(
    full_argument_list=cmd,
    dir=service_path,
    tag="appworld_env_service",
)
test_successful = False
try:
    companion.launch(
        launch_wait_time=1800,
        success_std_string="GoodbyeException",
        env_dict=os.environ,
    )
    test_successful = True
except Exception as e:
    test_successful = False
    print(f"Error launching companion: {e}")
    sys.exit(1)
finally:
    companion.kill_self()

print_dict(
    {
        "TestTarget": TEST_TARGET,
        "TestSuccessful": test_successful,
    }
)
