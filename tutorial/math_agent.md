## 运行math agent

### 1. 准备dataset

请下载 `openai/gsm8k` 数据集
```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset

# ╭───────────────────────────────────────────────────────────────── downloaded files ─────────────────────────────────────────────────────────────────╮
# │ ┌──────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
# │ │ 0    │ /the/path/to/store/dataset/openai/gsm8k/main/test-00000-of-00001.parquet                                                               │ │
# │ ├──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
# │ │ 1    │ /the/path/to/store/dataset/openai/gsm8k/main/train-00000-of-00001.parquet                                                              │ │
# │ ├──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
# │ │ 2    │ /the/path/to/store/dataset/openai/gsm8k/socratic/test-00000-of-00001.parquet                                                           │ │
# │ ├──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
# │ │ 3    │ /the/path/to/store/dataset/openai/gsm8k/socratic/train-00000-of-00001.parquet                                                          │ │
# │ └──────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
# ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
# Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████| 7473/7473 [00:00<00:00, 380098.15 examples/s]
# Generating test split: 100%|███████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:00<00:00, 358866.57 examples/s]
# 2025-11-10 11:14:30.220 | INFO     | best_logger.print_basic:print_listofdict:155 -
# ╭────────────────────────────────────────────────────────────────────── train ───────────────────────────────────────────────────────────────────────╮
# │ ┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
# │ ┃     ┃ question                                                           ┃ answer                                                              ┃ │
# │ ┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
# │ │ [0] │ Natalia sold clips to 48 of her friends in  April,  and  then  she │ Natalia sold 48/2 = <<48/2=24>>24 clips in May.                     │ │
# │ │     │ sold half as many clips in May. How many clips  did  Natalia  sell │ Natalia sold 48+24 = <<48+24=72>>72 clips altogether in  April  and │ │
# │ │     │ altogether in April and May?                                       │ May.                                                                │ │
# │ │     │                                                                    │ #### 72                                                             │ │
# │ ├─────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤ │
# │ │ [1] │ Weng earns $12 an hour for babysitting. Yesterday, she just did 50 │ Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.                    │ │
# │ │     │ minutes of babysitting. How much did she earn?                     │ Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.         │ │
# │ │     │                                                                    │ #### 10                                                             │ │
```

### 2. 准备 AgentScope Workflow

详细请见 `tutorial/math_agent.py`。可以在项目任意地方新建新的 AgentScope Workflow 代码

- 定义 AgentScope 的工作流
```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)
self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=beyondagent_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)
msg = Msg("user", init_messages[0]['content'], role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)
```


- 在 AgentScope Workflow 中，注册评价函数需要的任意关键数据
```python
beyondagent_proxy.update_judge_input_dictionary(final_answer=final_answer)
```


### 3. 准备Judge (奖励模块)

在 astune/task_judge/math_answer_as_judge.py 中，提供了两个简单的Judge。可以在项目任意地方新建新的Judge代码

Judge的输入参数包含：

```python
judge_input_dictionary['env']: env_service 外部环境 （如果使用了env_service）
judge_input_dictionary['task_core_arg']: 任务信息（如果里面包含了参考答案，可以从中取出）
judge_input_dictionary['grouped_steps']: LLM的每一次历史对话记录（如果中间过程比较重要，可以从中取出）
judge_input_dictionary['final_answer']: 默认没有final_answer，需要在agentscope workflow中手动调用 beyondagent_proxy.update_judge_input_dictionary(final_answer=final_answer) 注册
```

Judge的返回值： raw_reward, is_success


### 4. 测试


4.1 复制并修改 [launcher/math_agent/git-math-agentscope.yaml](../launcher/math_agent/git-math-agentscope.yaml) 中的关键参数，yaml中与本文档最相关的部分已经用✨✨✨✨符号标记

1. 读取task（对应配置字段 astune.task_reader）
2. 定义 AgentScopeWorkflow（对应配置字段 astune.rollout.agentscope_learn_protocol ）
    - 举例如果 agentscope workflow 定义在 `tutorial/math_agent.py` 的`ExampleMathLear` 类
    - 则填写 astune.rollout.agentscope_learn_protocol=`tutorial.math_agent->ExampleMathLearn`
3. 定义评分函数（对应配置字段 astune.task_judge.judge_protocol ）
    - 举例如果 agentscope workflow 定义在 `astune/task_judge/math_answer_as_judge.py` 的`MathAnswerAndLlmAsJudge` 类
    - 则填写 astune.task_judge.judge_protocol=`astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge`
4. 指定模型（对应配置字段 astune.model.path ）

```yaml
astune:
    task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` or `dataset_file` or `huggingface_dat_repo`
    rollout:
        use_agentscope_protocol: True
        agentscope_learn_protocol: tutorial.math_agent->ExampleMathLearn # ✨✨✨✨ 编写并选择Agent
    task_judge:
        # ✨✨✨✨ 编写并选择评价函数
        judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge
    model:
        # ✨✨✨✨ 设置待训练的模型
        path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
```


4.2 全链路调试（脱离ray快速调试:--backbone='debug'）
```bash
# （训练math agent demo）建议开始前杀死所有ray、env_service进程 ( python launcher.py --kill="python|ray" )
clear && \
python launcher.py --conf launcher/math_agent/git-math-agentscope.yaml --backbone='debug' --with-logview
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
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        },
    ]
}
```


4.3 当调试完成后，开始训练(只需要把backbone切换一下即可：--backbone='verl')
```bash
# 建议开始前杀死所有ray、vllm、env_service进程 ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf launcher/math_agent/git-math-agentscope.yaml --backbone='verl'
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