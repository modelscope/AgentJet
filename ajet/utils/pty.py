import base64
import os
import pty


def run_command_with_pty(cmd, working_dir, env_dict):
    """
    Run a command in a pseudo-terminal (PTY) and stream output to stdout.

    Args:
        cmd (list): Command to run (e.g., ["ls", "-l"]).
        working_dir (str): Working directory.
        env_dict (dict): Environment variables dictionary.
    """
    # Save original environment and directory
    original_env = os.environ.copy()
    original_dir = os.getcwd()

    try:
        # Change to the target working directory
        os.chdir(working_dir)

        # Update environment variables
        for key, value in env_dict.items():
            os.environ[key] = value

        # # Open a log file in append mode (optional)
        # with open(log_file, 'a') as log_f:

        # Define master device read callback
        def master_read(fd):
            try:
                # Read data from PTY master
                data = os.read(fd, 1024)
            except OSError:
                return b""

            if data:
                # Write data to log file
                # log_f.write(data.decode())
                # log_f.flush()
                # Also print to stdout (optional)
                # Use errors='replace' to handle incomplete UTF-8 sequences
                print(data.decode(errors='replace'), end="")
            return data

        # Define stdin read callback
        def stdin_read(fd):
            # Return empty bytes if no stdin input is needed
            return b""

        # Spawn a PTY and run the command
        pty.spawn(cmd, master_read, stdin_read)

    finally:
        # Restore original working directory
        os.chdir(original_dir)

        # Restore original environment variables
        os.environ.clear()
        os.environ.update(original_env)


# Convert string to Base64
def string_to_base64(s):
    # First, encode the string to bytes
    s_bytes = s.encode("utf-8")
    # Convert bytes to base64
    base64_bytes = base64.b64encode(s_bytes)
    # Convert base64 bytes back to string
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


# Convert Base64 back to string
def base64_to_string(b):
    # Convert base64 string to bytes
    base64_bytes = b.encode("utf-8")
    # Decode base64 bytes
    message_bytes = base64.b64decode(base64_bytes)
    # Convert bytes back to string
    message = message_bytes.decode("utf-8")
    return message


def pty_wrapper(
    cmd: list[str],
    dir: str,
    env_dict: dict[str, str] = {},
):
    run_command_with_pty(cmd, working_dir=dir, env_dict=env_dict)


def pty_wrapper_final(human_cmd, dir, env_dict):
    print("[pty]: ", human_cmd)
    pty_wrapper(["/bin/bash", "-c", human_cmd], dir, env_dict)


def pty_launch(service_name: str, success_std_string="Starting server on", prefix: str=""):
    from ajet.utils.smart_daemon import LaunchCommandWhenAbsent

    service_path = os.environ.get(f"{service_name.upper()}_PATH")
    service_script = os.environ.get(f"{service_name.upper()}_SCRIPT")
    if service_path is None or service_script is None:
        raise ValueError(f"Environment variables for {service_name} not properly set.")
    if prefix != "":
        service_name = prefix + "_" + service_name  
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[service_script],
        dir=service_path,
        tag=f"{service_name}_service",
        use_pty=True,
    )
    companion.launch(
        launch_wait_time=3600,
        success_std_string=success_std_string,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run a shell command in a PTY with logging and custom env."
    )
    parser.add_argument("--human-cmd", type=str, help="Shell command to run (as a string)")
    parser.add_argument("--dir", type=str, default=".", help="Working directory")
    parser.add_argument(
        "--env",
        type=str,
        default="{}",
        help='Environment variables as JSON string, e.g. \'{"KEY":"VAL"}\'',
    )

    args = parser.parse_args()

    try:
        env_dict = json.loads(args.env)
        if not isinstance(env_dict, dict):
            raise ValueError
    except Exception:
        print(
            '--env must be a valid JSON object string, e.g. \'{"KEY":"VAL"}\'. But get:',
            args.env,
        )
        exit(1)

    pty_wrapper_final(base64_to_string(args.human_cmd), args.dir, env_dict)
