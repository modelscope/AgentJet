# 训练你的第一个智能体

在本文档中，我们会从零开始，带你实现并训练一个能够使用 Python 进行计算、解决复杂数学问题的智能体。

---

## 学习路径

<div class="workflow-single">
<div class="workflow-header">完整训练流程</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>准备目录</strong>

创建项目结构和配置文件。</li>
<li><strong>配置参数</strong>

设置项目名称和基础配置。</li>
<li><strong>准备训练数据</strong>

配置数据源和加载方式。</li>
<li><strong>准备 Workflow</strong>

定义智能体行为和奖励计算。</li>
<li><strong>调试与训练</strong>

验证代码并启动正式训练。</li>
</ol>
</div>
</div>

完成整个过程后，您将得到一个可在数学任务环境中使用的 Math 智能体，理解 AgentJet 中的核心概念，并学会如何设计您自己的训练流程。

---

## 准备目录

首先，我们为训练准备一个目录：

```bash
mkdir math_agent
cd math_agent
touch math_agent.yaml
touch workflow.py
```

执行上述命令后，当前目录应当包含以下文件：

```
/math_agent
    /math_agent.yaml  # 配置文件
    /workflow.py      # 训练 workflow，包含智能体和环境交互的定义
```

---

## 配置参数

我们在配置文件中为项目起一个名字：

```yaml title="math_agent.yaml"
ajet:
  project_name: math_agent

# ------------------ 不需要修改 ------------------
hydra:
  searchpath:
    - file://ajet/default_config
    - file://ajet/default_config/verl         # verl only
    - file://ajet/default_config/trinity      # trinity only

# ------------------ 不需要修改 ------------------
defaults:
  - verl_default # verl inherit 1/1
  - trinity_default # trinity inherit 1/1
  - ajet_default
  - _self_
```

---

## 准备训练数据

智能体需要在指定的任务环境下，使用训练数据驱动进行训练。

!!! info "支持的数据源"
    AgentJet 提供了多种读取数据的方式：

    - 从本地硬盘中的文件读取
    - 从 HuggingFace Repo 中读取
    - 从 EnvService 中读取

    所有数据在读取后，都会被转换为 AgentJet 中统一的数据格式。

在本示例中，我们将直接从 HuggingFace Repo 获取 `openai/gsm8k` 作为训练数据。

!!! tip "关于 GSM8K"
    `openai/gsm8k` 是一个经典的数学问题数据集，非常适合用于训练数学推理智能体。

在 `math_agent.yaml` 中添加数据配置：

```yaml title="math_agent.yaml"
ajet:
  project_name: math_agent
  task_reader:
    type: huggingface_dat_repo
    huggingface_dat_repo:
      dataset_path: 'gsm8k/main'
      training_split: "train"
      validation_split: "test"
  data:
    train_batch_size: 264
    max_prompt_length: 3000
    max_response_length: 10000
```

| 配置项 | 说明 |
|--------|------|
| `type` | 指定使用 `huggingface_dat_repo` 读取器 |
| `dataset_path` | HuggingFace 上的数据集路径 |
| `training_split` | 训练集划分名称 |
| `validation_split` | 验证集划分名称 |
| `train_batch_size` | 每批次的任务数量 |
| `max_prompt_length` | 输入的最大 token 长度 |
| `max_response_length` | 回复的最大 token 长度 |

至此，我们就完成了数据相关的全部配置，剩余的工作将由 AgentJet 自动完成。

---

## 准备 Workflow

在 AgentJet 中，workflow 是进行训练的基本单元。它定义了智能体的行为、工具、上下文等，智能体与环境交互的具体流程，以及 Reward 的计算方法。

我们将在 `workflow.py` 中实现我们的 workflow。

### 定义 Agent

首先，引入 agentscope 的必要依赖，并设计一个 Agent：

```python title="workflow.py"
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code

system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""

toolkit = Toolkit()
toolkit.register_tool_function(execute_python_code)

ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model= # 暂时留空,
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
    max_iters=2,
)
```

!!! note "ReActAgent 说明"
    通过这段代码，我们定义了一个完整的 ReAct 智能体：

    - 使用 ReAct 范式与环境/工具交互
    - 设置了 system prompt
    - 注册了一个工具：执行 Python 代码
    - 实现了 in-memory 的记忆机制

    更多具体的配置可参考 AgentScope 官方文档。

### 封装为 Workflow

接下来，将智能体封装进一个 Workflow 类中：

```python title="workflow.py"
from ajet import ModelTuner, Workflow, WorkflowTask, WorkflowOutput
from agentscope.message import Msg
from loguru import logger

system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""

def extract_final_answer(result) -> str:
    """Extract the final answer from the agent's response."""
    try:
        if (
            hasattr(result, "metadata")
            and isinstance(result.metadata, dict)
            and "result" in result.metadata
        ):
            return result.metadata["result"]
        if hasattr(result, "content"):
            if isinstance(result.content, dict) and "result" in result.content:
                return result.content["result"]
            return str(result.content)
        return str(result)
    except Exception as e:
        logger.warning(f"Extract final answer error: {e}. Raw: {result}")
        return str(result)

class MathAgentWorkflow(Workflow):
    name: str = "math_agent_workflow"

    async def execute(
        self, workflow_task: WorkflowTask, model_tuner: ModelTuner
    ) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit, execute_python_code

        query = workflow_task.task.main_query
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=model_tuner,  # 使用 model_tuner 作为模型
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
            max_iters=2,
        )
        # 禁用控制台输出
        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        # 调用智能体执行任务
        result = await self.agent.reply(msg)
        # 提取最终答案
        final_answer = extract_final_answer(result)
        # 将答案封装为输出
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
```

!!! warning "关键点：model_tuner"
    在构造智能体时，**必须将 model 参数设置为 model_tuner**，这是使智能体可被训练的关键！

### 配置 Workflow 和 Reward

在 `math_agent.yaml` 中添加 workflow 和奖励配置：

```yaml title="math_agent.yaml"
ajet:
  # ...
  rollout:
    user_workflow: workflow.py->MathAgentWorkflow

  task_judge:
    judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge
```

!!! info "关于 Judge"
    这里我们使用 AgentJet 内部提供的 math judge。Judge 会读取 metadata 中的 `final_answer`，并与数据集中的 ground_truth 对比来得到评分。

---

## 设置必要的参数

### 预训练模型

在 `math_agent.yaml` 中指定智能体使用的 LLM，也就是我们将要训练的模型：

```yaml title="math_agent.yaml"
ajet:
  model:
    path: Qwen/Qwen2.5-14B-Instruct
```

!!! tip "模型来源"
    `path` 既支持 HuggingFace Repo 中的远程地址，也可以是本地目录路径。

### 训练参数

添加必要的训练参数配置：

```yaml title="math_agent.yaml"
ajet:
  # ...
  rollout:
    user_workflow: workflow.py->MathAgentWorkflow
    temperature: 0.7                    # 模型温度
    max_env_worker: 64                  # 最大的 rollout 多线程数量
    num_repeat: 4                       # 每个任务重复次数
    agent_madness_reward: 0.0           # 模型乱码输出时的强制 reward
    tensor_model_parallel_size: 1       # vllm 的并行参数
    max_num_seqs: 40                    # vllm 最大的并行 sequence 数量
    multi_turn:
      max_sample_per_task: 4
    compute_madness_checklist:          # 检查的模型乱码输出类型
      - "nonsense"
      - "wrong_toolcall"
    max_response_length_in_one_turn: 1024   # 一轮交互中的最长回复长度
    max_model_len: 13000                    # 模型的最长上下文长度

  trainer_common:
    save_freq: 99999        # 保存周期（每 n 步保存一次）
    test_freq: 99999        # 验证周期（每 n 步验证一次）
    total_epochs: 99999     # 训练 epoch 总量
    trinity_only__n_vllm_engine: 2  # vllm engine 数量

trinity:
  trainer:
    max_token_len_per_gpu: 13000
```

---

## 调试

到目前为止，我们已经完成了训练一个智能体所需的全部代码与配置。

!!! tip "推荐调试模式"
    接下来，我们将在 debug 模式下启动训练流程，检查代码是否有误。您也可以跳过这一步，直接开始训练。

使用 debug 模式启动训练：

```bash
ajet --conf math_agent/math_agent.yaml --backbone='debug' --with-logview
```

!!! info "Debug 模式特点"
    debug 模式不会启动 ray 集群，非常适合单机代码调试。

### VS Code 断点调试

编写 VS Code 的 `.vscode/launch.json` 进行便捷的断点调试：

```json title=".vscode/launch.json"
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Launch rollout",
      "type": "debugpy",
      "request": "launch",
      "module": "ajet.cli.launcher",
      "console": "integratedTerminal",
      "args": [
        "--backbone", "debug",
        "--conf", "math_agent/math_agent.yaml"
      ],
      "env": {}
    }
  ]
}
```

---

## 训练

调试完成后，即可开始正式训练：

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

!!! success "训练输出"
    您可以在 `./launcher_record/{exp_yaml_file_name}` 目录下找到训练日志和 checkpoint。

---

## 了解更多

<div class="card-grid">
<a href="./workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>可训练工作流</h3></div><p class="card-desc">深入了解 Workflow 的定义和高级用法。</p></a>
<a href="./data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>数据管道</h3></div><p class="card-desc">了解更多数据加载和处理方式。</p></a>
<a href="./task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>任务评判器</h3></div><p class="card-desc">学习如何自定义奖励计算。</p></a>
<a href="./configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-tool" alt=""><h3>配置指南</h3></div><p class="card-desc">完整的配置选项参考。</p></a>
</div>
