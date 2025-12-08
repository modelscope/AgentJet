# 数学
本页演示如何从零开始准备数据、构建 Agent 与 Workflow、配置 Reward，并最终训练一个数学 Agent。

## 1. 准备数据集
下载 `openai/gsm8k` 数据集：

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

## 2. 准备 AgentScope Workflow
详细示例见 `tutorial/example_math_agent/math_agent.py`。你可以在项目中的任意位置编写新的 AgentScope Workflow 代码。

- **定义 AgentScope Workflow**

```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)
self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=astune_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)
msg = Msg("user", init_messages[0]['content'], role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)
```

在 AgentScope Workflow 中，你需要将评估函数所需的关键信息写入（注册）进去，例如：

```python
astune_proxy.update_judge_input_dictionary(final_answer=final_answer)
```



## 3. 准备 Reward
在 `astune/task_judge/math_answer_as_judge.py` 中提供了两个简单的 Judge 示例。你也可以在项目任意位置实现自己的 Judge 逻辑。

Judge 的输入参数包括：

```
judge_input_dictionary['env']: env_service 外部环境（如果使用 env_service）
judge_input_dictionary['workflow_task']: 任务信息（如果包含参考答案，可以从这里获取）
judge_input_dictionary['grouped_steps']: 每轮 LLM 对话的完整历史（如果需要利用中间推理过程，可以从这里获取）
judge_input_dictionary['final_answer']: 默认不存在，需要你在 AgentScope Workflow 中手动调用 astune_proxy.update_judge_input_dictionary(final_answer=final_answer) 进行注册。
```

Judge 的返回值包括：

- raw_reward
- is_success

## 4. 启动训练

### 4.1 配置
拷贝并修改 `tutorial/example_math_agent/math_agent.yaml` 中的关键配置参数。yaml 中与本示例最相关的部分已经用 ✨✨✨✨ 标出。

1. 读取任务（对应配置字段 `astune.task_reader`）
2. 定义 Workflow（对应配置字段 `astune.rollout.agentscope_learn_protocol`）
   - 示例：如果 AgentScope Workflow 定义在 `tutorial/math_agent.py` 的 `ExampleMathLear` 类中
   - 则配置 `astune.rollout.agentscope_learn_protocol`=`tutorial.math_agent->ExampleMathLearn`
3. 定义评分函数（对应配置字段 `astune.task_judge.judge_protocol`）
   - 示例：如果评分逻辑定义在 `astune/task_judge/math_answer_as_judge.py` 的 `MathAnswerAndLlmAsJudge` 类中
   - 则配置 `astune.task_judge.judge_protocol`=`astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge`
4. 指定模型（对应配置字段 `astune.model.path`）

```yaml
astune:
    task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` 或 `dataset_file` 或 `huggingface_dat_repo`
    rollout:
        use_agentscope_protocol: True
        agentscope_learn_protocol: tutorial.math_agent->ExampleMathLearn # ✨✨✨✨ 编写并选择 Agent
    task_judge:
        # ✨✨✨✨ 编写并选择评估函数
        judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge
    model:
        # ✨✨✨✨ 设置需要训练的模型
        path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
```



### 4.2 调试

```bash
# 建议在启动前先杀掉所有 ray 和 env_service 相关进程（ python launcher.py --kill="python|ray" ）
clear && \
python launcher.py --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
```

当 `--backbone=debug` 时，程序不再使用 Ray。你可以在 vscode 中配置 `launch.json`，方便地进行断点调试。例如：

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



### 4.3 正式训练

```bash
# 建议在启动前先杀掉所有 ray、vllm 和 env_service 相关进程（ python launcher.py --kill="python|ray|vllm" ）
python launcher.py --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

## 5 参考结果

![Tracing curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

## 调参与现象观察

在训练一开始，Agent 就已经能够解决大部分问题；但由于模型规模有限，仍然会产生许多不符合标准 `<tool_call>` 格式的回答。在部分题目中，Agent 会编写 Python 代码来计算答案，但又太急于给出结果，在真正执行工具调用之前，就直接“猜”了一个答案并跳过了 tool_call 执行阶段。

```
# bad case 1: 在 python 代码中忘记打印结果。
<|im_start|>assistant
To find the height difference between Bella and Anne's sister, let's first determine their heights based on the information provided.
- Anne's height = 80 cm
- Anne is 2 times as tall as her sister, so Anne's sister's height = \( \frac{Anne's\ height}{2} \)
- Bella is 3 times as tall as Anne, so Bella's height = \( 3 \times Anne's\ height \)
We can calculate these values and then determine the height difference between Bella and Anne's sister.
<tool_call>
{"name": "execute_python_code", "arguments": {"code": "anne_height = 80\nsister_height = anne_height / 2\nbella_height = 3 * anne_height\nheight_difference = bella_height - sister_height\nheight_difference"}}
</tool_call><|im_end|>

# bad case 2: 太着急给出答案，没有等待工具执行结束。
<|im_start|>assistant
To find the number of board members who attended the meeting, let's break down the information given: 1. There are 40 chairs, each with a capacity of 2 people. 2. The total capacity of the chairs is \( 40 \times 2 = 80 \) people. 3. Since the number of board members is the same as the chairs' capacity, there are 80 board members in total. 4. \( \frac{2}{5} \) of the chairs were not occupied, so \( \frac{2}{5} \times 40 = 16 \) chairs were empty. 5. Therefore, \( 40 - 16 = 24 \) chairs were occupied. 6. Each of these 24 chairs was fully occupied by 2 people, meaning \( 24 \times 2 = 48 \) people attended the meeting.  Let's confirm this with a Python calculation.
<tool_call> {"name": "execute_python_code", "arguments": {"code": "total_chairs = 40\nchair_capacity = 2\nunoccupied_fraction = 2/5\nunoccupied_chairs = total_chairs * unoccupied_fraction\noccupied_chairs = total_chairs - unoccupied_chairs\nattended_board_members = occupied_chairs * chair_capacity\nattended_board_members"}} </tool_call>
 <tool_call> {"name": "generate_response", "arguments": {"response": "The number of board members who did attend the meeting is \(\boxed{48}\)."}}
</tool_call>
<|im_end|>
```

经过调优后，这些问题会得到明显改善，例如下图所示：

![image](https://img.alicdn.com/imgextra/i4/O1CN01v1gGQZ1ftMiil5Cxg_!!6000000004064-2-tps-1367-684.png)
![image](https://img.alicdn.com/imgextra/i4/O1CN01WarPpf1yNk4awZOIO_!!6000000006567-2-tps-1363-422.png)