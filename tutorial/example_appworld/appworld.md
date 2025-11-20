## 运行 Appworld AgentScope Agent

### 1. 准备 dataset

请下载 `env_service` 以及 `appworld`。具体步骤请参考 [EnvService文档](https://code.alibaba-inc.com/EconML/EnvService)


### 2. 准备 AgentScope Workflow

详细请见 `tutorial/math_agent.py`。可以在项目任意地方新建新的 AgentScope Workflow 代码

- 定义 AgentScope 的工作流 （把agent的model修改为`astune_proxy`）

```python

agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg['content'],
    model=astune_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

for _ in range(config.astune.rollout.multi_turn.max_steps):
    # agentscope deal with interaction message
    reply_message = await agent(interaction_message)
    # env service protocol
    obs, _, terminate, _ = astune_proxy.gym_step(action={"content": reply_message.content, "role": "assistant"})
    # generate new message from env output
    interaction_message = Msg(name="env", content=obs, role="user")
    # is terminated?
    if terminate: break
    if astune_proxy.context_overflow: break

```

- 其中，使用了 astune_proxy 与 agentscope runtime 环境交互的一些接口如下：
    - `astune_proxy.gym_step` 模拟gym接口，输入动作，输出 observation, reward, terminate_flag, info 四元组
    - `astune_proxy.context_overflow` 查询当前的context窗口是否token溢出

### 3. 准备Judge (奖励模块)

在 `astune/task_judge/env_service_as_judge.py` 中，我们直接向 env_service 发送http请求，读取奖励。

Judge的返回值： raw_reward, is_success


### 4. 测试


4.1 复制并修改 [tutorial/example_appworld/appworld.yaml](../tutorial/example_appworld/appworld.yaml) 中的关键参数，yaml中与本文档最相关的部分已经用✨✨✨✨符号标记

1. 读取task（对应配置字段 astune.task_reader）
2. 定义 Workflow（对应配置字段 astune.rollout.agentscope_learn_protocol ）
    - 举例如果 agentscope workflow 定义在 `tutorial/appworld.py` 的`ExampleAgentScopeLearnProtocol` 类
    - 则填写 astune.rollout.agentscope_learn_protocol=`tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol`
3. 定义评分函数（对应配置字段 astune.task_judge.judge_protocol ）
    - 填写 astune.task_judge.judge_protocol=`astune.task_judge.env_service_as_judge->EnvServiceJudge`
4. 指定模型（对应配置字段 astune.model.path ）

```yaml
astune:
  project_name: appworld_astune
  experiment_name: "read_yaml_name"
  task_judge:
    # ✨✨✨✨ 编写并选择评价函数
    judge_protocol: astune.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # ✨✨✨✨ 设置待训练的模型
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ 编写并选择Agent
    use_agentscope_protocol: True
    agentscope_learn_protocol: tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol
    agentscope_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```


4.2 全链路调试（脱离ray快速调试:--backbone='debug'）
```bash
# （训练math agent demo）建议开始前杀死所有ray、env_service进程 ( python launcher.py --kill="python|ray" )
clear && python launcher.py --conf tutorial/example_appworld/appworld.yaml --backbone='debug' --with-logview

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
                "--with-appworld",
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        },
    ]
}
```


4.3 当调试完成后，开始训练(只需要把backbone切换一下即可：--backbone='trinity')
```bash
# 建议开始前杀死所有ray、vllm、env_service进程 ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity'
```


### 5. 读取Rollout日志

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


### 6. 参考训练曲线


<div align="center">
  <img src="tutorial/figure/appworld.png" alt="训练曲线">
</div>
