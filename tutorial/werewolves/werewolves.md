## 运行 WereWolves AgentScope Agent


### 1. 准备 AgentScope Workflow

详细请见 `tutorial/werewolves/start.py`。可以在项目任意地方新建新的 AgentScope Workflow 代码

- 定义 AgentScope 的工作流 （把agent的model修改为`astune_proxy`）
    ```python
    class ExampleWerewolves(AgentScopeLearnProtocol):
        trainer: str = Field(default="astune-trinity")
        async def agentscope_execute(self, init_messages, astune_proxy: ModelTuner, config) -> WorkflowOutput:

            train_which_role = "witch"
            roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]

            # Set random seed for reproducibility
            task_core_arg = astune_proxy.get_agentscope_input_dictionary()[task_core_arg]
            task_id = task_core_arg.task.task_id

            np.random.seed(int(task_id))
            np.random.shuffle(roles)

            players = [get_official_agents(f"Player{x + 1}", roles[x], train_which_role, astune_proxy) for x in range(9)]

            good_guy_win = await werewolves_game(players, roles)
            raw_reward = 1 if (good_guy_win and train_which_role != "werewolf") or (not good_guy_win and train_which_role == "werewolf") else 0
            astune_proxy.update_judge_input_dictionary(raw_reward = raw_reward)
            astune_proxy.update_judge_input_dictionary(is_success = (raw_reward == 1))
            return astune_proxy

    ```


### 2. 测试


- 复制并修改 [launcher/werewolves_agent/git-rpg-agentscope.yaml](../launcher/werewolves_agent/git-rpg-agentscope.yaml) 中的关键参数，yaml中与本文档最相关的部分已经用✨✨✨✨符号标记
    ```yaml
    astune:
    task_reader:
        type: random_dummy # `env_service` or `dataset_file` or `huggingface_dat_repo` or `random_dummy`
    task_judge:
        # 编写并选择评价函数
        judge_protocol: null
    model:
        # ✨✨✨✨ 设置待训练的模型
        path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
    rollout:
        agentscope_learn_protocol: tutorial.werewolves.start->ExampleWerewolvesLearn # ✨✨✨✨ 编写并选择Agent
    ```


- 全链路调试（脱离ray快速调试:--backbone='debug'）
    ```bash
    # （训练math agent demo）建议开始前杀死所有ray、env_service进程 ( python launcher.py --kill="python|ray" )
    python launcher.py --kill="python|ray|vllm|VLLM" && ray stop && clear && python launcher.py --conf launcher/werewolves_agent/git-rpg-agentscope.yaml --backbone='debug' --with-logview

    ```
    备注：当--backbone=debug时，程序不再使用ray，可以编写vscode的launch.json进行便捷的断点调试，launch.json的配置:
    ```json
    {

        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python Debugger: Launch rollout",
                "type": "debugpy",
                "request": "launch",
                "program": "launcher.py",
                "console": "integratedTerminal",
                "args": [
                    "--backbone",  "debug",
                    "--conf", "launcher/werewolves_agent/git-rpg-agentscope.yaml"
                ],
                "env": {
                }
            },
        ]
    }
    ```


- 当调试完成后，开始训练(只需要把backbone切换一下即可：--backbone='trinity')
    ```bash
    # 建议开始前杀死所有ray、vllm、env_service进程 ( python launcher.py --kill="python|ray|vllm" )
    python launcher.py --kill="python|ray|vllm|VLLM" && ray stop && clear && \
    python launcher.py --conf launcher/werewolves_agent/git-rpg-agentscope.yaml --backbone='trinity' --with-ray
    ```


### 3. 读取Rollout日志

<div align="center">
  <img src="tutorial/figure/best-logger.png" alt="日志界面">
</div>

- 找到日志文件夹，默认在 `./launcher_record/exp_yaml_file_name/*` 下面
- 运行 `beast_logger_go` 启动日志浏览器，vscode端口映射8181端口
```bash
root@xxxx:/xxx/xxx/xxx# beast_logger_go
INFO:     Started server process [74493]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8181 (Press CTRL+C to quit)
```
- 打开 http://127.0.0.1:8181，提示输入日志文件路径，填写日志文件夹的**绝对路径**，以下形式皆可
    - /mnt/data/qingxu.fu/astune/astune/launcher_record
    - /mnt/data/qingxu.fu/astune/astune/launcher_record/exp_yaml_file_name
    - /mnt/data/qingxu.fu/astune/astune/launcher_record/exp_yaml_file_name/2025_11_10_02_52/rollout

- 依次打开界面 **左侧** 的日志文件目标，**中间** 的日志条目，**右侧** 的交互记录，即可显示完整的轨迹

- 蓝色 Token 代表参与loss计算的 Token，黄色反之

- 鼠标悬浮在 Token 上面可以查看 Token 的 **logprob** (暂时仅限trinity backbone)


#### 4. 训练参考曲线

<div align="center">
  <img src="tutorial/figure/werewolves_train_witch.png" alt="日志界面">
</div>
