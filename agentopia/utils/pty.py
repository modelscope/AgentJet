import os
import pty

def run_command_with_pty(cmd, working_dir, env_dict):
    """
    使用伪终端运行命令，并将输出写入日志文件。

    参数：
        cmd (list): 要运行的命令（如 ["ls", "-l"]）。
        working_dir (str): 工作目录。
        env_dict (dict): 环境变量字典。
    """
    # 保存原始环境变量
    original_env = os.environ.copy()
    original_dir = os.getcwd()

    try:
        # 切换到指定工作目录
        os.chdir(working_dir)

        # 更新环境变量
        for key, value in env_dict.items():
            os.environ[key] = value

        # # 打开日志文件以追加模式写入
        # with open(log_file, 'a') as log_f:

        # 定义主设备读取回调函数
        def master_read(fd):
            try:
                # 从主设备读取数据
                data = os.read(fd, 1024)
            except OSError:
                return b""

            if data:
                # 将数据写入日志文件
                # log_f.write(data.decode())
                # log_f.flush()
                # 同时打印到标准输出（可选）
                print(data.decode(), end="")
            return data

        # 定义标准输入读取回调函数
        def stdin_read(fd):
            # 如果不需要从标准输入读取数据，直接返回空字节
            return b""

        # 使用 pty.spawn 分配伪终端并运行命令
        pty.spawn(cmd, master_read, stdin_read)

    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)

        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(original_env)

import base64

# 将字符串转换为 Base64
def string_to_base64(s):
    # 首先将字符串编码为字节
    s_bytes = s.encode('utf-8')
    # 将字节转换为 base64
    base64_bytes = base64.b64encode(s_bytes)
    # 将 base64 字节转换回字符串
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

# 将 Base64 转换回字符串
def base64_to_string(b):
    # 将 base64 字符串转换为字节
    base64_bytes = b.encode('utf-8')
    # 解码 base64 字节
    message_bytes = base64.b64decode(base64_bytes)
    # 将字节转换回字符串
    message = message_bytes.decode('utf-8')
    return message

def pty_wrapper(
    cmd: list[str],
    dir: str,
    env_dict: dict = {},
):
    run_command_with_pty(cmd, working_dir=dir, env_dict=env_dict)

def pty_wrapper_final(human_cmd, dir, env_dict):
    print("[pty]: ", human_cmd)
    pty_wrapper(["/bin/bash", "-c", human_cmd], dir, env_dict)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a shell command in a PTY with logging and custom env.")
    parser.add_argument("--human-cmd", type=str, help="Shell command to run (as a string)")
    parser.add_argument("--dir", type=str, default=".", help="Working directory")
    parser.add_argument("--env", type=str, default="{}", help="Environment variables as JSON string, e.g. '{\"KEY\":\"VAL\"}'")

    args = parser.parse_args()

    try:
        env_dict = json.loads(args.env)
        if not isinstance(env_dict, dict):
            raise ValueError
    except Exception:
        print("--env must be a valid JSON object string, e.g. '{\"KEY\":\"VAL\"}'. But get:", args.env)
        exit(1)

    pty_wrapper_final(base64_to_string(args.human_cmd), args.dir, env_dict)

