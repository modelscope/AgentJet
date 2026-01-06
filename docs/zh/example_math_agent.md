# 数学智能体

训练一个**会用工具的数学智能体**（ReAct + Python 执行器），用于解决 GSM8K 风格的小学应用题。奖励来自一个**评审器 (judge)**：它检查最终答案是否正确（并且可选地惩罚不良的工具调用行为）。

---

## 概览

<div class="callout-tip">
<p>
在 <strong>Math 智能体</strong> 中，每条训练样本是一道数学文字题（如 GSM8K）。智能体将学习分步推理（ReAct 风格），在需要计算时调用 Python 工具，并产出与参考答案一致的最终答案。
</p>
</div>

本教程分两步组织：

1. **先跑起来**：下载数据集，并用默认 YAML 配置启动训练
2. **理解与自定义**：理解 workflow 代码与 judge/reward

---

## 快速开始

### 准备数据集

下载 `openai/gsm8k` 数据集：

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

### 启动训练

```bash
# （可选）训练前建议清理残留进程
# astuner --kill="python|ray|vllm"

astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

??? tip "快速调试（可选）"
    如果您想在本地对 workflow / judge 打断点调试：

    ```bash
    # （可选）调试前建议清理残留进程
    # astuner --kill="python|ray"

    clear && \
    astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
    ```

    当 `--backbone=debug` 时，Ray 会被禁用。您可以使用类似下面的 VSCode 配置：

    ```json title=".vscode/launch.json"
    {
      "version": "0.2.0",
      "configurations": [
        {
          "name": "Python Debugger: Launch rollout",
          "type": "debugpy",
          "request": "launch",
          "module": "agentscope_tuner.cli.launcher",
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

---

## 理解训练流程

### 每一步训练都发生了什么

<div class="workflow-single">
<div class="workflow-header">训练步骤流程</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>加载一道题</strong>

从数据集通过 `task_reader` 加载一道数学题。</li>
<li><strong>运行 AgentScope workflow</strong>

用题目文本构造 prompt，让 ReAct 智能体调用 Python 工具，并抽取最终答案。</li>
<li><strong>注册用于评估的关键信息</strong>

返回 `WorkflowOutput(reward=None, metadata={"final_answer": final_answer})`。</li>
<li><strong>运行 judge</strong>

将 `final_answer` 与参考答案对比，计算 `raw_reward` 和 `is_success`。</li>
</ol>
</div>
</div>

### YAML 配置说明

大部分"连线"都在 `tutorial/example_math_agent/math_agent.yaml` 中完成：

```yaml title="math_agent.yaml"
astuner:
  task_reader:
    type: huggingface_dat_repo   # 也支持: dataset_file / env_service

  rollout:
    agentscope_workflow: tutorial.example_math_agent.math_agent->ExampleMathLearn

  task_judge:
    judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAndLlmAsJudge

  model:
    path: YOUR_MODEL_PATH
```

| 字段 | 说明 |
|------|------|
| `task_reader` | 任务来源 |
| `agentscope_workflow` | 每条样本运行哪个 workflow |
| `judge_protocol` | 由哪个 judge 计算 reward |
| `model.path` | 要微调的预训练模型路径 |

### 代码解读

**Workflow：** `tutorial/example_math_agent/math_agent.py`

```python title="Workflow 代码梗概"
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

!!! warning "重要提示"
    始终通过 `WorkflowOutput.metadata` 提供最终答案，以便 judge 进行评分。

### 奖励计算

评审器接收两个对象：

| 对象 | 包含内容 |
|------|----------|
| `workflow_task` | 任务信息；从 `metadata` 获取参考答案 |
| `workflow_output` | Workflow 结果；从 `metadata["final_answer"]` 获取最终答案 |

!!! tip "扩展 Judge"
    如果您观察到模型"几乎做对了，但搞错了工具调用格式"，可以扩展评审器来：
    
    - 增加格式惩罚（无效的 `<tool_call>`）
    - 增加行为惩罚（调用了工具但没 `print`）
    - 同时保持"答案正确性"作为主要信号

---

## 结果

### 训练曲线

![训练曲线](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

!!! info "可视化说明"
    训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md)。

**解读：** 随着训练进行，reward 随之上升。这通常意味着智能体在两件事上变得更稳定：

- **该用工具时会用**：能正确发起 `<tool_call>`，并在需要计算时调用 `execute_python_code`
- **产出更可靠的答案**：能基于工具返回的结果输出与参考答案一致的最终答案

### 案例展示：工具调用纪律的改进

训练前，智能体可能已经能解出不少题。然而，小模型经常在**工具调用规范**上翻车。

=== "Bad Cases"

    ```text
    # bad case 1: 忘记在 Python 代码里 print 结果
    <tool_call>
    {"name": "execute_python_code", "arguments": {"code": "... height_difference"}}
    </tool_call>

    # bad case 2: 太心急 —— 没等工具返回结果就输出了最终答案
    <tool_call> {"name": "execute_python_code", ...} </tool_call>
    <tool_call> {"name": "generate_response", "arguments": {"response": "... \\boxed{48} ..."}} </tool_call>
    ```

    这些失败本质上不是因为模型"不会算"，而是因为它**没有形成决策闭环**，未能将工具执行结果纳入考虑。

=== "Good Case（调优后）"

    调优后，智能体通常会遵循规范的三段式结构：

    1. **Message 3 (assistant)**：拆解问题 + 发起 `<tool_call>`，并在代码里使用 `print(...)`
    2. **Message 4 (tool_response)**：工具返回执行结果
    3. **Message 5 (assistant)**：读取 `stdout`，然后产出最终答案

    ![Good case](https://img.alicdn.com/imgextra/i4/O1CN01v1gGQZ1ftMiil5Cxg_!!6000000004064-2-tps-1367-684.png)

!!! note "Token 级可视化"
    图中彩色块是 [Beast-Logger](./beast_logger.md) 生成的 token 级别序列可视化：
    
    - **黄色 token**：排除在损失函数（loss）计算之外的 token
    - **蓝色 token**：参与损失计算的 token（从浅蓝到深蓝表示 logprob 从高到低）

---

## 下一步

<div class="card-grid">
<a href="../example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>狼人杀游戏</h3></div><p class="card-desc">探索多智能体协作训练。</p></a>
<a href="../example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld</h3></div><p class="card-desc">训练用于真实应用交互的智能体。</p></a>
<a href="../visualization/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-line.svg" class="card-icon card-icon-general" alt=""><h3>训练可视化</h3></div><p class="card-desc">监控和分析您的训练进度。</p></a>
</div>
