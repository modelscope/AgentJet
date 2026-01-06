# Tune Your First Agent

In this document, we demonstrate how to implement and train, from scratch, an agent that can use Python to perform calculations and solve complex math problems.

<div class="workflow-single">
<div class="workflow-header">Training Pipeline Overview</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Prepare training data and environment</strong>

Set up the dataset and configure the task reader.</li>
<li><strong>Define the agent and trainable workflow</strong>

Create your agent using AgentScope and wrap it in a Workflow class.</li>
<li><strong>Define reward function</strong>

Configure how the agent's outputs are evaluated and scored.</li>
<li><strong>Configure training hyperparameters</strong>

Set model path, batch size, and other training parameters.</li>
<li><strong>Debug</strong>

Test your workflow in debug mode before full training.</li>
<li><strong>Start training & monitor metrics</strong>

Launch the training process and track progress.</li>
</ol>
</div>
</div>

!!! success "What You'll Learn"
    After completing this guide, you will:

    - Obtain a Math Agent that can solve math problems using Python
    - Understand the core concepts in AgentScope Tuner
    - Learn how to design your own training pipeline

---

## Step 1: Prepare the Working Directory

First, create a directory for this training project:

```bash
mkdir math_agent
cd math_agent
touch math_agent.yaml
touch workflow.py
```

After running the commands above, the directory should contain:

```
/math_agent
    /math_agent.yaml  # Configuration file
    /workflow.py      # Training workflow definition
```

---

## Step 2: Configure Project Name

Give the project a name in the config file:

```yaml title="math_agent.yaml"
ajet:
   project_name: math_agent

# ------------------ No need to modify ------------------
hydra:
  searchpath:
    - file://ajet/default_config
    - file://ajet/default_config/verl         # verl only
    - file://ajet/default_config/trinity      # trinity only

# ------------------ No need to modify ------------------
defaults:
  - verl_default # verl inherit 1/1
  - trinity_default # trinity inherit 1/1
  - astuner_default
  - _self_
```

---

## Step 3: Prepare Training Data

The agent needs to be trained in a specific task environment, driven by training data.

!!! info "Data Sources"
    ASTuner provides multiple ways to read data:

    - Read from local files on disk
    - Read from a Hugging Face repo
    - Read from an EnvService

All data will be converted into a unified ASTuner data format after loading.

In this example, we will use the `openai/gsm8k` dataset from Hugging Face:

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

| Configuration | Description |
|--------------|-------------|
| `type` | Use the `huggingface_dat_repo` reader |
| `dataset_path` | Path to the HuggingFace dataset |
| `train_batch_size` | Training batch size (hyperparameter) |
| `max_prompt_length` | Maximum input length |
| `max_response_length` | Maximum response/answer length |

---

## Step 4: Prepare the Workflow

In ASTuner, a workflow is the basic unit for training. It defines:

- Agent's behavior and tools
- Interaction procedure with the environment
- How to calculate rewards

### Define the Agent

First, import dependencies and design an agent in `workflow.py`:

```python title="workflow.py"
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code

system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{}.
"""

toolkit = Toolkit()
toolkit.register_tool_function(execute_python_code)

# Agent definition (model will be set later)
ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=None,  # leave empty for now
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
    max_iters=2,
)
```

!!! note "Agent Features"
    This agent:

    - Uses the ReAct paradigm to interact with tools
    - Has a custom system prompt
    - Registers `execute_python_code` as a tool
    - Implements in-memory memory

### Wrap in Workflow Class

Next, wrap the agent into a trainable workflow:

```python title="workflow.py"
from ajet import ModelTuner, Workflow, WorkflowTask, WorkflowOutput
from agentscope.message import Msg
from loguru import logger

system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{}.
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
            model=model_tuner,  # use model_tuner as the model
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
            max_iters=2,
        )
        # disable console output
        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        # call agent to do the task
        result = await self.agent.reply(msg)
        # extract the final answer
        final_answer = extract_final_answer(result)
        # pass the final answer to the output
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
```

!!! warning "Key Change"
    The critical step is setting `model=model_tuner` — this is what makes the agent trainable!

### Configure Workflow in YAML

Add the workflow configuration to `math_agent.yaml`:

```yaml title="math_agent.yaml"
ajet:
  # ...
  rollout:
    agentscope_workflow: workflow.py->MathAgentWorkflow
  task_judge:
    judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge
```

The judge reads `final_answer` from `metadata` and compares it with ground-truth answers to produce a score.

---

## Step 5: Configure Required Parameters

### Pretrained Model

Specify the LLM to train:

```yaml title="math_agent.yaml"
ajet:
  model:
    path: Qwen/Qwen2.5-14B-Instruct
```

!!! tip "Model Path"
    The `path` can be a remote Hugging Face repo, or a local directory path.

### Training Hyperparameters

Configure important training hyperparameters:

??? example "Full Configuration Example"
    ```yaml title="math_agent.yaml"
    ajet:
      # ...
      rollout:
        agentscope_workflow: workflow.py->MathAgentWorkflow
        temperature: 0.7
        max_env_worker: 64
        num_repeat: 4
        agent_madness_reward: 0.0
        tensor_model_parallel_size: 1
        max_num_seqs: 40
        multi_turn:
          max_sample_per_task: 4
        compute_madness_checklist:
          - "nonsense"
          - "wrong_toolcall"
        max_response_length_in_one_turn: 1024
        max_model_len: 13000

      trainer_common:
        save_freq: 99999
        test_freq: 99999
        total_epochs: 99999
        trinity_only__n_vllm_engine: 2

    trinity:
      trainer:
        max_token_len_per_gpu: 13000
    ```

| Parameter | Description |
|-----------|-------------|
| `temperature` | Model sampling temperature |
| `max_env_worker` | Maximum parallel rollout workers |
| `num_repeat` | Number of repetitions per sample |
| `save_freq` | Checkpoint save interval |
| `test_freq` | Evaluation interval |

---

## Step 6: Debug

Before full training, test in debug mode:

```bash
ajet --conf math_agent/math_agent.yaml --backbone='debug' --with-logview
```

!!! tip "VS Code Debugging"
    You can configure `.vscode/launch.json` for breakpoint debugging:

    ```json
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

## Step 7: Start Training

After debugging, launch the full training:

```bash
ajet --conf math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

!!! success "Output Location"
    Training logs and checkpoints will be saved to:
    ```
    ./launcher_record/{exp_yaml_file_name}/
    ```

---

## Next Steps

<div class="card-grid">
<a href="../workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Workflow</h3></div><p class="card-desc">Learn to define trainable workflows and multi-agent setups.</p></a>
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Configure data loading from various sources.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions for your training.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent Example</h3></div><p class="card-desc">See the complete Math Agent implementation.</p></a>
</div>
