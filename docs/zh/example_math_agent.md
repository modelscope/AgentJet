# 数学智能体

训练一个**会用工具的数学智能体**（ReAct + Python 执行器），用于解决 GSM8K 风格的小学应用题。
奖励来自一个**评审器 (judge)**：它检查最终答案是否正确（并且可选地惩罚不良的工具调用行为）。

---

### 1. 概览

在 **Math Agent** 中，每条训练样本是一道数学文字题（如 GSM8K）。智能体将学习：

* **分步推理**（ReAct 风格），
* 在需要计算时**调用 Python 工具**，
* 产出与参考答案一致的最终答案。

本教程分两步组织：

1. **先跑起来**：下载数据集，并用默认 YAML 配置启动训练。
2. **理解与自定义**：理解 workflow 代码（`ExampleMathLearn`）与 judge/reward（`MathAnswerAndLlmAsJudge`）。

---

### 2. 快速开始

#### 2.1 准备数据集

下载 `openai/gsm8k` 数据集：

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

#### 2.2 启动训练

```bash
# （可选）训练前建议清理残留进程
# astuner --kill="python|ray|vllm"

astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

<details>
<summary>快速调试（可选）</summary>

如果你想在本地对 workflow / judge 打断点调试：

```bash
# （可选）调试前建议清理残留进程
# astuner --kill="python|ray"

clear && \
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
```

当 `--backbone=debug` 时，Ray 会被禁用。你可以使用类似下面的 VSCode `launch.json`：

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
        "--backbone", "debug",
        "--conf", "./path/to/yaml.yaml"
      ],
      "env": {}
    }
  ]
}

```

</details>

---

### 3. 理解实现

#### 3.1 每一步训练都发生了什么

每个 training step 会做：

1. 从数据集加载**一道题**（`task_reader`）。
2. 运行 **AgentScope workflow**：
* 用题目文本构造 prompt，
* 让 ReAct 智能体在需要时调用 Python 工具进行计算，
* 抽取**最终答案**。


3. **注册用于评估的关键信息（很重要！）**：
* Workflow 应该返回一个 `WorkflowOutput` 对象，其 `metadata` 携带最终答案，例如：`WorkflowOutput(reward=None, metadata={"final_answer": final_answer})`。评审器（Judge）会直接读取此 metadata，无需额外的 API 调用。


4. 运行 **judge** 计算 reward：
* 将 `final_answer` 与任务中的参考答案对比，
* 输出 `raw_reward` 与 `is_success`，
* Trainer 使用这些结果来更新策略。


#### 3.2 YAML 配置说明

大部分“连线”都在 `tutorial/example_math_agent/math_agent.yaml` 中完成。关键字段包括：

* `astune.task_reader`：任务来源
* `astune.rollout.agentscope_workflow`：每条样本运行哪个 workflow
* `astune.task_judge.judge_protocol`：由哪个 judge 计算 reward
* `astune.model.path`：要微调的预训练模型路径

最小示例：

```yaml
astune:
  task_reader:
    type: huggingface_dat_repo   # 也支持: dataset_file / env_service（如果启用）

  rollout:
    agentscope_workflow: tutorial.example_math_agent.math_agent->ExampleMathLearn

  task_judge:
    judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge

  model:
    path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct

```

#### 3.3 代码解读

**Workflow (AgentScope)：** `tutorial/example_math_agent/math_agent.py`

Workflow 通常会做：

* 注册工具（例如 `execute_python_code`）
* 构造一个 ReAct agent
* 从用户题目运行一轮对话
* 解析最终答案
* 通过 `WorkflowOutput(..., metadata={"final_answer": final_answer})` 返回答案，以便评审器评分。

Workflow 代码梗概：

```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)

self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=model_tuner,  # trainer 管理的模型封装
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)

msg = Msg("user", init_messages[0]["content"], role="user")
result = await self.agent.reply(msg)
final_answer = extract_final_answer(result)

# 重要：通过 WorkflowOutput 的 metadata 将最终答案提供给评审器
return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
```

**Judge / Reward：** `astune/task_judge/math_answer_as_judge.py`

该文件内提供了两个简单的评审器；你也可以在项目的任何位置添加自己的评审器。

#### 3.4 奖励 (Reward)

评审器接收两个对象：

* `workflow_task`：任务信息，可从中检索参考答案。
* `workflow_output`：由 workflow 返回；通过 `workflow_output.metadata["final_answer"]` 获取最终答案。

评审器返回：

* `raw_reward`
* `is_success`

**实用建议：**
如果你观察到模型“几乎做对了，但搞错了工具调用格式 / 没等工具执行就跳过了”，你可以扩展评审器来：

* 增加格式惩罚（无效的 `<tool_call>`）
* 增加行为惩罚（调用了工具但没 `print` / 没用到执行结果）
* 同时保持“答案正确性”作为主要信号。

---

### 4. 结果

#### 4.1 训练曲线

> **可视化说明：** 训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md)。

解读：随着训练进行，reward 随之上升。这通常意味着智能体在**两件事**上变得更稳定：

* **该用工具时会用**：能正确发起 `<tool_call>`，并在需要计算时调用 `execute_python_code`。
* **产出更可靠的答案**：能基于工具返回的结果（例如 `<tool_response>`）输出与参考答案一致的最终答案。

> 在实践中，这里的提升往往不在于“数学能力变强”，而在于“更好的工具调用纪律 + 对执行结果更一致的使用”。

---

#### 4.2 案例展示：从“会算”到“会用工具算”

训练前，智能体可能已经能解出不少题。然而，小模型经常在**工具调用规范**上翻车，例如：

* 忘记在 Python 代码里 `print` 计算结果（工具运行了，但没有产出可用的输出）。
* 在工具执行结束前就急着输出最终答案（抢答）。
* `<tool_call>` 块格式错误（导致工具不触发或解析失败）。

##### Bad case：典型失败表现

```text
# bad case 1: 忘记在 Python 代码里 print 结果
<tool_call>
{"name": "execute_python_code", "arguments": {"code": "... height_difference"}}
</tool_call>

# bad case 2: 太心急 —— 没等工具返回结果就输出了最终答案
<tool_call> {"name": "execute_python_code", ...} </tool_call>
<tool_call> {"name": "generate_response", "arguments": {"response": "... \\boxed{48} ..."}} </tool_call>

```

这些失败本质上不是因为模型“不会算”，而是因为它**没有形成决策闭环**，未能将工具执行结果纳入考虑：

* bad case 1：工具可能执行成功，但没有 `print`，`stdout` 就是空的，模型无法可靠地读取数值。
* bad case 2：模型在同一轮中连续生成工具调用和最终回答，实际上**跳过了“等待 `<tool_response>`”的步骤**。

---

##### Good case：调优后，工具调用链路变得闭环

调优后，智能体通常会遵循规范的三段式结构（对应截图中的 Message 3/4/5）：

1. **Message 3 (assistant)**：拆解问题 + 发起 `<tool_call>`，并在代码里使用 `print(...)` 输出关键数值。
2. **Message 4 (tool_response)**：工具返回执行结果（如 `returncode=0`, `stdout=...`）。
3. **Message 5 (assistant)**：读取 `stdout`，然后产出最终答案（如 `\\boxed{18}`）。

图中右侧彩色块是 **token 级别的序列可视化**：

> **Token级可视化：** 这些详细日志由 Beast-Logger 生成。详见 [Beast-Logger 使用说明](./beast_logger.md)。

* **每个小块代表一个 token**（块内数字是 token id）。
* 块的顺序就是模型**消耗/生成** token 的顺序。
* 重点不在于 token id 本身，而在于你能否看到清晰的边界标记，例如：
* `<im_start> assistant ... <tool_call> ... <im_end>`
* `<im_start> user <tool_response> ... <stdout>18.0</stdout> ... <im_end>`
* `<im_start> assistant ... \\boxed{18} ... <im_end>`


* **黄色 token**：排除在损失函数（loss）计算之外的 token。**蓝色 token**：参与损失计算的 token（从浅蓝到深蓝表示 `logprob` 从高到低）。

一个“好”的 tool-call 行为在日志中通常体现为：

* `<tool_call>` 和 `<tool_response>` **分轮出现**（发起调用 -> 得到响应 -> 最终回答）。
* `<tool_response>` 包含 **非空的 stdout**。
* 最终答案出现在工具返回**之后**，而不是提前抢答。