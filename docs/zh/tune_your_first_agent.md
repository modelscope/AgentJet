# 训练你的第一个 Agent


在本文档中，我们会从零开始，带你实现并训练这样一个能够使用 Python 进行计算、解决复杂数学问题的 Agent。

我们将依次完成以下步骤：

1. 训练数据与环境准备
2. Agent 与可训练 workflow 的定义
3. Reward 计算
4. 训练参数设置
5. 调试
6. 开始训练 & 监测指标

完成整个过程后，你将得到一个可在数学任务环境中使用的 Math Agent，理解 AgentScope Tuner 中的核心概念，并学会如何设计你自己的训练流程。

## 准备目录
首先，我们为训练准备一个目录：

```bash
mkdir math_agent
cd math_agent
touch math_agent.yaml
touch workflow.py
```

在执行上述命令后，当前目录应当包含以下文件：

```
/math_agent
    /math_agent.yaml # 配置文件
    /workflow.py      # 训练 workflow，将包含 agent 和环境交互的定义
```

## 配置参数

我们在配置文件中为项目起一个闪亮的名字：

```yaml
astuner:
  project_name: math_agent

# ------------------ 不需要修改 ------------------
hydra:
  searchpath:
    - file://astuner/default_config
    - file://astuner/default_config/verl         # verl only
    - file://astuner/default_config/trinity      # trinity only

# ------------------ 不需要修改 ------------------
defaults:
  - verl_default # verl inherit 1/1
  - trinity_default # trinity inherit 1/1
  - astuner_default
  - _self_
```

## 准备训练数据
Agent 需要在指定的任务环境下，使用训练数据驱动进行训练。本节中我们首先解决数据和环境的问题。

ASTuner 提供了多种读取数据的方式：

- 从本地硬盘中的文件读取
- 从 Hugginface Repo 中读取
- 从 EnvService 中读取

所有数据在读取后，都会被转换为 ASTuner 中统一的数据格式。

在本示例中，我们将直接从 Huggingface Repo 获取 `openai/gsm8k` 作为训练数据。

> `openai/gsm8k` 是一个经典的数学问题数据集。

我们在 `math_agent.yaml` 中写入：

```yaml
astuner:
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

在这段配置中，我们：
- 指定使用 huggingface_dat_repo 读取器，并设置了 dataset 的 path，以及训练集与验证集划分的名称
- 设置了数据的 batch size（训练参数），并指定了数据的最长输入长度、最长回复长度（答案长度）

至此，我们就完成了数据相关的全部配置，剩余的工作将由 ASTuner 自动完成。

## 准备 Workflow
在 ASTuner 中，workflow 是进行训练的基本单元。它定义了 Agent 的行为、工具、上下文等，Agent 与环境交互的具体流程，以及 Reward 的计算方法。

我们将在 `workflow.py` 中实现我们的 workflow。

首先，我们引入 agentscope 的必要依赖，并设计一个 Agent：

```python
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
    model= #暂时留空,
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
    max_iters=2,
)
```

通过这段代码，我们快速地定义了一个完整的 ReAct Agent：
- 使用 ReAct 范式与环境/工具交互
- 设置了 system prompt
- 注册了一个工具：执行 Python 代码
- 实现了 in-memory 的记忆机制

> 更多具体的配置可参考 AgentScope 官方文档。

接下来，我们实现训练该 Agent 的其余代码：

```python
from astuner import ModelTuner, Workflow, WorkflowTask, WorkflowOutput
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
            model=model_tuner # use model_tuner as model,
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
            max_iters=2,
        )
        # disable console output
        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        # call agent to do task
        result = await self.agent.reply(msg)
        # extract the final answer
        final_answer = extract_final_answer(result)
        # bring final answer to the output
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
```

在这段代码中，我们将 Agent 封装进一个 Workflow 中，并实现了其中的 `execute` 函数：
1. 从 `WorkflowTask` 中取出输入
2. 用与前面相同的方式构造 Agent，但 **指定 model 为 model_tuner** —— 这是使 Agent 可被训练的关键
3. 执行 Agent
4. 解析 Agent 输出，并将答案封装为 `WorkflowOutput` 返回

在定义好 workflow 后，我们还需要告诉 ASTuner 使用这个 class 作为训练协议。在 `math_agent.yaml` 中写入：

```yaml
astuner:
  # ...
  rollout:
    agentscope_workflow: math_agent.math_agent->MathAgentWorkflow
```

我们还有一件事情尚未解决：Reward。幸运的是，ASTuner 同样支持通过配置自定义 Reward 计算方法。继续在配置中写入：

```yaml
astuner:
  #...
  task_judge:
    judge_protocol: astuner.task_judge.math_answer_as_judge->MathAnswerAsJudge
```

这里我们直接使用 ASTuner 内部提供的 math judge。Judge 会读取 metadata 中的 final_answer，并与数据集中的 ground_truth 对比来得到评分。

## 设置必要的参数
接下来，为了正式开始训练，还需要补充一些必要的参数配置。

### 预训练模型
在 `math_agent.yaml` 中指定 Agent 使用的 LLM，也就是我们将要训练的模型：

```yaml
astuner:
  model:
    path: Qwen/Qwen2.5-14B-Instruct
```

这里的 path 既支持 Huggingface Repo 中的远程地址，也可以是本地目录路径。

### 训练参数
我们还需要设置一些必要的训练参数：

```yaml
astuner:
  # ...
  rollout:
    agentscope_workflow: math_agent.math_agent->MathAgentWorkflow
    # 模型温度
    temperature: 0.7
    # 最大的 rollout 多线程数量
    max_env_worker: 64
    num_repeat: 4
    # 模型乱码输出时的强制 reward
    agent_madness_reward: 0.0
    # vllm 的并行参数
    tensor_model_parallel_size: 1
    # vllm 最大的并行 sequence 数量
    max_num_seqs: 40
    multi_turn:
      max_sample_per_task: 4
    # 检查的模型乱码输出类型：无意义内容、错误的 tollcall
    compute_madness_checklist:
      - "nonsense"
      - "wrong_toolcall"
    # 一轮交互中的最长回复长度
    max_response_length_in_one_turn: 1024
    # 模型的最长上下文长度
    max_model_len: 13000

  trainer_common:
    # 保存周期（每 n 步保存一次）
    save_freq: 99999
    # 验证周期（每 n 步验证一次）
    test_freq: 99999
    # 训练 epoch 总量
    total_epochs: 99999
    # vllm engine 数量，必须能够整除 tensor_model_parallel_size
    trinity_only__n_vllm_engine: 2

trinity:
  trainer:
    max_token_len_per_gpu: 13000
```

## 调试
到目前为止，我们已经完成了训练一个 Agent 所需的全部代码与配置。

接下来，我们将在 debug 模式下启动训练流程，检查代码是否有误。你也可以跳过这一步，直接开始训练。

使用 debug 模式启动训练：

```bash
astuner --conf math_agent/math_agent.yaml --backbone='debug' --with-logview
```

debug 模式不会启动 ray 集群，非常适合单机代码调试。另外，我们也可以编写 VS Code 的 `launch.json` 进行便捷的断点调试：

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

## 训练
当调试完成后，即可开始训练：
```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

你可以在 `./launcher_record/{exp_yaml_file_name}` 目录下找到训练日志和 checkpoint。


## 阅读更多

我们还在左侧边栏提供了更多详细的使用案例和教程，包括数据处理、模型训练、多智能体协作等高级主题。
你可以阅读并深入了解使用方法。