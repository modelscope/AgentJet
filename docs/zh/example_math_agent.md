# Math Agent

训练一个**会用工具的数学智能体**（ReAct + Python 执行器），用于解决 GSM8K 风格的小学应用题。
奖励来自一个**评审器（judge）**：它检查最终答案是否正确（并且可选地惩罚不良的工具调用行为）。

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

#### 2.3 本地调试（不启用 Ray）

如果你想在本地对 workflow / judge 打断点调试：

```bash
# （可选）调试前建议清理残留进程
# astuner --kill="python|ray"

clear && \
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
```

当 `--backbone=debug` 时，会禁用 Ray。你可以使用类似下面的 VSCode `launch.json`：

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
        "--conf", "xxxx/xxxx/xxxx.yaml"
      ],
      "env": {}
    }
  ]
}
```

---

### 3. 理解实现

#### 3.1 每一步训练都发生了什么

每个 training step 会做：

1. 从数据集加载**一道题**（`task_reader`）。
2. 运行 **AgentScope workflow**：

   * 用题目文本构造 prompt，
   * 让 ReAct 智能体在需要时调用 Python 工具进行计算，
   * 抽取**最终答案**。
3. 注册用于评估的关键信息（很重要！）：

   * workflow 必须调用：

     * `astune_proxy.update_judge_input_dictionary(final_answer=final_answer)`
4. 运行 **judge** 计算 reward：

   * 将 `final_answer` 与 task 中的参考答案对比，
   * 输出 `raw_reward` 与 `is_success`，
   * trainer 用它们来更新策略。

#### 3.2 YAML 配置说明

大部分“连线”都在 `tutorial/example_math_agent/math_agent.yaml` 中完成，关键字段包括：

* `astune.task_reader`：任务来源
* `astune.rollout.agentscope_workflow`：每条样本跑哪个 workflow
* `astune.task_judge.judge_protocol`：由哪个 judge 计算 reward
* `astune.model.path`：要微调的预训练模型路径

最小示例：

```yaml
astune:
  task_reader:
    type: huggingface_dat_repo   # 也支持: dataset_file / env_service（如果启用）

  rollout:
    agentscope_workflow: tutorial.math_agent->ExampleMathLearn

  task_judge:
    judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge

  model:
    path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
```

#### 3.3 代码解读

**Workflow（AgentScope）：** `tutorial/example_math_agent/math_agent.py`

workflow 通常会做：

* 注册工具（例如 `execute_python_code`）
* 构造一个 ReAct agent
* 从用户题目跑一轮对话
* 解析最终答案
* 通过 `update_judge_input_dictionary` 把答案提供给 judge

workflow 示意：

```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)

self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=astune_proxy,  # trainer 管理的模型封装
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)

msg = Msg("user", init_messages[0]["content"], role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)

# IMPORTANT: 将最终答案提供给 judge
astune_proxy.update_judge_input_dictionary(final_answer=final_answer)
```

**Judge / Reward：** `astune/task_judge/math_answer_as_judge.py`

该文件内提供了两个简单 judge；你也可以在项目任意位置添加自己的 judge。

#### 3.4 奖励

judge 会从 `judge_input_dictionary` 中读取信息，常见字段包括：

* `env`：env_service 外部环境（若启用）
* `workflow_task`：任务信息；参考答案可从这里获取
* `grouped_steps`：所有 LLM 对话轮次（如果你要做过程型打分会很有用）
* `final_answer`：默认**不会**自动提供——你必须在 workflow 中通过下面方式设置：

  * `astune_proxy.update_judge_input_dictionary(final_answer=final_answer)`

judge 输出：

* `raw_reward`
* `is_success`

**实用建议：**
如果你观察到模型“几乎做对了，但工具调用格式搞错 / 太着急跳过了工具执行”，你可以扩展 judge 来：

* 增加格式惩罚（无效的 `<tool_call>`）
* 增加行为惩罚（调用了工具但没 print / 没用到执行结果）
* 同时仍然保持“答案正确性”为主信号


---

### 4. 结果

#### 4.1 训练曲线

![Tracing curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

解读：reward 随训练上升，通常意味着智能体在**两件事**上更稳定：

* **该用工具时会用**：能正确发起 `<tool_call>`，并在需要计算时调用 `execute_python_code`。
* **答案更可靠**：能基于工具返回的结果（`<tool_response>`）输出与参考答案一致的最终答案。

---

#### 4.2 案例展示：从“会算”到“会用工具算”

训练前，模型可能已经能解出不少题，但小模型经常在工具调用上翻车，常见表现包括：

* 忘记在 python 代码里 `print` 计算结果（工具执行了，但没有可用输出）
* 没等工具执行结束就急着输出最终答案（抢答）
* `<tool_call>` 块格式不正确（导致工具不触发或解析失败）

##### Bad case：典型失败长什么样

```text
# bad case 1: python 代码里忘记 print 结果
<tool_call>
{"name": "execute_python_code", "arguments": {"code": "... height_difference"}}
</tool_call>

# bad case 2: 太着急，没等工具执行结束就输出最终答案
<tool_call> {"name": "execute_python_code", ...} </tool_call>
<tool_call> {"name": "generate_response", "arguments": {"response": "... \\boxed{48} ..."}} </tool_call>
```

这些失败本质上不是“不会算”，而是**没有把工具执行结果纳入决策闭环**：

* bad case 1：工具可能执行成功，但由于没有 `print`，`stdout` 为空，后续无法可靠地读取数值。
* bad case 2：模型在同一轮里连续生成工具调用和最终回答，相当于“**跳过了等待 `<tool_response>` 的步骤**”。

---

##### Good case：调优后，工具调用链路变得闭环

调优后，模型通常会形成更规范的三段式结构（对应截图中的 Message 3/4/5）：

1. **Message 3（assistant）**：拆解问题 + 发起 `<tool_call>`，并在代码里 `print(...)` 输出关键数值
2. **Message 4（tool_response）**：工具返回执行结果（`returncode=0`、`stdout=...`）
3. **Message 5（assistant）**：读取 `stdout`，再输出最终答案（如 `\\boxed{18}`）

![image](https://img.alicdn.com/imgextra/i4/O1CN01v1gGQZ1ftMiil5Cxg_!!6000000004064-2-tps-1367-684.png)

![image](https://img.alicdn.com/imgextra/i4/O1CN01WarPpf1yNk4awZOIO_!!6000000006567-2-tps-1363-422.png)


图中右侧彩色块是 **token 级别的序列可视化**：

* **每个小块代表一个 token**（块内数字是 token id）
* 小块的顺序就是模型“看到/生成”的顺序
* 重点不是 token id 本身，而是你能否在序列里看到清晰的边界标记，例如：

  * `<im_start> assistant ... <tool_call> ... <im_end>`
  * `<im_start> user <tool_response> ... <stdout>18.0</stdout> ... <im_end>`
  * `<im_start> assistant ... \\boxed{18} ... <im_end>`

一个“好”的 tool-call 行为，在 log 上通常体现为：

* `<tool_call>` 和 `<tool_response>` **分轮出现**（先 call，再 response，最后才 answer）
* `<tool_response>` 中有 **可用的 stdout**
* 最终答案出现在工具返回之后，而不是抢答
