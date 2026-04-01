## 你的任务

1. 根据实验蓝图，运行实验
2. 等待实验结束或者超时
3. 如果实验失败，尝试进行修正，把试错过程放置到指定位置（exp_result_dir中创建一个文档），如果无法修复，则跳到第 5 步
4. 将尽可能全面的实验结果放置到指定位置（exp_result_dir中创建一个文档）
5. 在 exp_result_dir 中创建一个 `finish.flag` 文件，标志任务结束
6. 结束

## 实验蓝图：

设计实验蓝图是为了执行实验以验证猜想或者获取必要数据。
一个实验蓝图是一个markdown文件（blueprint.md）。实验蓝图包含的内容主要包括 7 部分，你需要根据这些信息，执行任务：

1. [exp_purpose] 实验目的（一段文本）：
    简要说明本次实验的主要目的，以及本实验蓝图与之前其他蓝图的主要区别（例如哪个超参数不同，哪个环境变量不同）。
2. [exp_codebase_dir] 主实验代码路径（绝对路径）：
    包含运行实验所需要的所有代码的**绝对路径**。体积较小。不包含python虚拟环境。记得在实验开始前，cd到这个路径。
3. [exp_venv_exe] python虚拟环境路径（python的绝对路径）:
    python可执行文件的路径。比如: /mnt/data_cpfs/agentjet/project/.venv/bin/python
4. [exp_yaml_path] 实验配置文件路径（相对路径，相对于主实验代码路径）：
    实验配置的yaml文件的路径。注意该文件必须包含在主实验代码路径内。比如: /mnt/data_cpfs/agentjet/project/tests/bench/benchmark_math/benchmark_math.yaml
5. [exp_launch_command] 执行训练的命令（字符串）：
    例如 python -m ajet.launcher --conf tests/bench/benchmark_math/benchmark_math.yaml --autokill
6. [exp_result_dir] 结果数据存储路径（绝对路径）：
    输出数据的存储路径
7. [exp_max_time] 运行时间不超过 ${MaxTime}，每个实验强制在  ${MaxTime} 后终止


## YAML 配置内容提示：

`ajet.execute_test` 应该为 False，因为启用后如果训练奖励分数低于预定义阈值，训练将被中断。
`ajet.trainer_common.test_freq` 意思是每间隔多少个step测试一次。
`ajet.trainer_common.n_gpus_per_node` 为每个节点显卡的数量，一般为 `8`。
`ajet.trainer_common.val_print_to_markdown_file_path` 应该是存储评估结果的位置。虽然你可以参考 tmux 控制台日志获取数据，但你应该始终在此路径中找到评估结果。选择一个放置日志的路径，例如 `saved_val_result/qwen2-7b-task-math-exp-01.md`。val 属性列表：
    pass_n: 对于每个任务，重复运行多少次。
    total_tasks: 验证数据集中的任务数量。
    num_all_success_tasks: 达到100%成功率的 task数量。
    num_pass_n_tasks: 至少成功一次的 task数量。
    task_pass_rate@1: 平均成功率
    task_pass_rate@2: 在前2次试验中至少成功一次的 task数量（占所有task的比例）
    task_pass_rate@4: 在前4次试验中至少成功一次的 task数量（占所有task的比例）（可选）
    task_pass_rate@8: 在前8次试验中至少成功一次的 task数量（占所有task的比例）（可选）
    mean_reward: 所有数据点的平均验证奖励。
    std_reward: 所有数据点的奖励标准差。
`ajet.trainer_common.val_before_train` 应该为 train，因为我们希望获得训练模型的初始性能。
`ajet.trainer_common.total_epochs` 应该足够大，但每个实验你只有 `${MaxTime}` 小时来运行，



## 使用tmux运行实验

详细见“监控实验的技能”，注意，当你创建session时，session名字中必须包含关键字 `ajet` 并且体现 `exp_purpose`，例如 `ajet_math_top_k_ablation`。


## 不要轻易中止进行中的实验

你必须保证在 [exp_max_time] 时间段内，维持实验继续进行下去。除非：

- 除非实验的错误过于严重，无法修复

- 除非实验已经提前成功，程序主动地运行结束，取得了完整的数据

- 除非实验已经进入中后期，且 `val_print_to_markdown_file_path` 中的 mean_reward 或者 task_pass_rate 已经开始长时间不发生变化


## 监控实验的技能

```
    ---
    name: monitor-with-tmux
    description: 通过指数退避间隔（30秒、1分钟、2分钟、4分钟、8分钟、16分钟）读取tmux内容来监控训练进度，在出现异常时分析日志，并提供修复建议
    license: 完整条款见 LICENSE.txt
    ---

    # 使用 Tmux 监控

    在 tmux 中监控，检测异常，分析错误，提供修复建议。

    ## 步骤零

    创建用于 tmux 监控的睡眠脚本：

    1. 创建 `./tmp/wait_tmux.py`

    ```python
    import argparse
    import subprocess
    import time

    SHELLS = {"bash", "zsh", "sh", "fish", "csh", "tcsh", "ksh", "dash", "ash"}

    def smart_sleep(session: str, seconds: float, check_every: float = 2.0) -> bool:
        """
        替代 time.sleep()，但在命令结束时提前返回。

        Returns:
            True  - 正常超时（命令还在跑）
            False - 提前返回（命令结束了或session没了）
        """
        end_time = time.time() + seconds
        while time.time() < end_time:
            try:
                r = subprocess.run(
                    ["tmux", "list-panes", "-F", "#{pane_current_command}", "-t", session],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode != 0:
                    return False  # session没了
                cmds = [l.strip().lower() for l in r.stdout.splitlines() if l.strip()]
                if not any(c not in SHELLS for c in cmds):
                    return False  # 命令结束了，回到shell
            except Exception:
                return False

            time.sleep(min(check_every, end_time - time.time()))

        return True


    def main():
        parser = argparse.ArgumentParser(description="Wait for a tmux session with smart early-exit.")
        parser.add_argument("session", help="tmux session name")
        parser.add_argument("seconds", type=float, help="total seconds to wait")
        args = parser.parse_args()

        timed_out = smart_sleep(args.session, args.seconds, 2)
        raise SystemExit(0 if timed_out else 1)


    if __name__ == "__main__":
        main()
    ```

    ## 开始监控

    当你需要监控一个 tmux 窗口时，运行：

    ```bash
    python ./tmp/tmux_wait.py my_ajet_session_name 30
    ```

    意思是：
    1. 监控名字为 my_ajet_session_name 的 tmux session
    2. 等待30秒

    - 退出码 0：正常超时（命令仍在运行）
    - 退出码 1：命令提前结束或 session 消失

    ## 使用 SSH

    使用 SSH 时，始终使用本地 tmux 窗口来建立 SSH 连接。

    ## 当你想在再次读取 tmux 窗口前延迟一段时间

    必须要具备出错即时返回功能，即不能使用 `sleep xxx`，而是使用 `python ./tmux_wait.py my_ajet_session_name xxx`

    不要使用：`sleep 60 && tmux capture-pane -t my_ajet_session_name -p | tail -80`

    你应该使用：`python ./tmux_wait.py my_ajet_session_name 30 && tmux capture-pane -t my_ajet_session_name -p | tail -80`

    - 60秒太长了
    - 始终使用 `python ./tmux_wait.py` 来等待



    ## 示例：


    ### 不使用 SSH 的示例

    0 个示例可用


    ### 使用 SSH 的示例

    1 个示例可用

    ```agent

    --- example user instruction begin ---
    login
    ssh root@localhost -p 8824
    goto
    /mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet
    run with venv and monitor
    source .venv/bin/activate && python -m ajet.launcher --conf tests/bench/benchmark_math/benchmark_math.yaml --autokill
    --- example user instruction end ---

    --- example agent reaction begin ---
    用户希望我监控一个 tmux session 同时运行训练任务。让我分解他们要求的内容：
    1. SSH 到远程机器（root@localhost -p 8824）
    2. 转到特定目录
    3. 激活虚拟环境并运行 Python 命令
    4. 监控 tmux session

    # 编写 ./tmp/tmux_wait.py
    ....

    # 创建用于训练的 tmux session (注意，无论如何，session名字中必须包含关键字 `ajet` )
    $ tmux new-session -d -s ajet_session -c "/mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet"
    ...

    # 发送 SSH 命令到 tmux session
    $ tmux send-keys -t ajet_session "ssh root@localhost -p 8824" Enter
    ...

    # 等待并检查 SSH 连接
    $ python3 /tmp/tmux_wait.py ajet_session 10 && tmux capture-pane -t ajet_session -p | tail -20
    SSH 连接成功。现在我需要导航到目录并运行训练命令。让我发送命令到 tmux session。
    ...

    # 导航到工作目录
    $ tmux send-keys -t ajet_session "cd /mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet" Enter
    ...

    # 使用 venv 和 launcher 启动训练
    $ tmux send-keys -t ajet_session "source .venv/bin/activate && python -m ajet.launcher --conf tests/bench/benchmark_math/benchmark_math.yaml --autokill" Enter
    ...

    # 捕获初始训练输出
    $ tmux capture-pane -t ajet_session -p | tail -40
    ...

    # 等待30秒并检查训练进度
    $ python3 /tmp/tmux_wait.py ajet_session 30 && tmux capture-pane -t ajet_session -p | tail -50
    ...
    训练正在运行。命令正在正确执行。让我继续以1分钟间隔监控。

    # 等待1分钟并检查训练进度
    $ python3 /tmp/tmux_wait.py ajet_session 60 && tmux capture-pane -t ajet_session -p | tail -50
    ...

    # 等待2分钟并检查训练进度
    $ python3 /tmp/tmux_wait.py ajet_session 120 && tmux capture-pane -t ajet_session -p | tail -60
    ...
    训练进展良好。检查点已加载，vLLM 服务器已启动，swanlab 正在跟踪运行。目前没有错误，只有警告。让我继续监控。

    # 等待4分钟并检查训练进度
    $ python3 /tmp/tmux_wait.py ajet_session 240 && tmux capture-pane -t ajet_session -p | tail -60
    ...

    ... 多轮监控 ...

    --- example agent reaction end ---

    # 销毁 tmux session
    tmux kill-session -t ajet_session

    ```
```
