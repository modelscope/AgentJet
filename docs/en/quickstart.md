# Quick Start

AgentScope Tuner provides a complete feature set for tuning agents. You can try starting training an agent right away:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

In this document, we demonstrate how to implement and train, from scratch, this agent that can use Python to perform calculations and solve complex math problems.

We will go through the following steps in order:

1. Prepare training data and environment
2. Define the agent and a trainable workflow
3. Define reward
4. Configure training hyperparameters
5. Debug
6. Start training & monitor metrics

After completing the whole process, you will obtain a Math Agent that can be used in a math task environment, understand the core concepts in AgentScope Tuner, and learn how to design your own training pipeline.

## Prepare the working directory

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
    /workflow.py      # Training workflow; will contain the definition of the agent and its interaction with the environment
```

Next, give the project a shiny name in the config file:

```yaml
astuner:
   project_name: math_agent

# ------------------ No need to modify ------------------
hydra:
  searchpath:
    - file://astuner/default_config
    - file://astuner/default_config/verl         # verl only
    - file://astuner/default_config/trinity      # trinity only

# ------------------ No need to modify ------------------
defaults:
  - verl_default # verl inherit 1/1
  - trinity_default # trinity inherit 1/1
  - astuner_default
  - _self_
 ```

## Prepare training data

The agent needs to be trained in a specific task environment, driven by training data. In this section we first prepare the data and environment.

ASTuner provides multiple ways to read data:

- Read from local files on disk
- Read from a Hugging Face repo
- Read from an EnvService

All data will be converted into a unified ASTuner data format after loading.

In this example, we will directly use the `openai/gsm8k` dataset from a Hugging Face repo as our training data.

> `openai/gsm8k` is a classic math problem dataset.

Add the following to `math_agent.yaml`:

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

In this configuration we:

- Specify that we use the `huggingface_dat_repo` reader, and configure the dataset path and the names of the training and validation splits
- Configure the training batch size (a hyperparameter), and set the maximum input length and maximum response length (answer length)

At this point, all data-related configuration is finished, and the remaining work will be handled automatically by ASTuner.

## Prepare the workflow

In ASTuner, a workflow is the basic unit for training. It defines the agent's behavior, tools, context, the detailed interaction procedure between the agent and the environment, and how to calculate rewards.

We will implement our workflow in `workflow.py`.

First, import the necessary dependencies from `agentscope` and design an agent:

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
    model= # leave empty for now,
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
    max_iters=2,
)
```

With this code, we quickly define a complete ReAct agent:

- Uses the ReAct paradigm to interact with the environment/tools
- Sets a system prompt
- Registers a tool: execute_python_code
- Implements in-memory memory

> For more detailed configuration options, please refer to the official AgentScope documentation.

Next, we implement the remaining code for training this agent:

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
            model=model_tuner # use model_tuner as the model,
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

In this code, we wrap the agent into a workflow and implement the `execute` function:

1. Read the input from `WorkflowTask`
2. Construct the agent in the same way as before, but **set `model` to `model_tuner`** — this is the key to making the agent trainable
3. Run the agent
4. Parse the agent's output and pack the answer into a `WorkflowOutput` object

After defining the workflow, we also need to tell ASTuner to use this class as the training protocol. Add the following to `math_agent.yaml`:

```yaml
astuner:
  # ...
  rollout:
    agentscope_workflow: math_agent.math_agent->MathAgentWorkflow
```

There is one more thing we have not addressed yet: the reward. Fortunately, ASTuner also supports customizing how rewards are calculated through configuration. Continue editing the config:

```yaml
astuner:
  #...
  task_judge:
    judge_protocol: astuner.task_judge.math_answer_as_judge->MathAnswerAsJudge
```

Here we directly use the built-in math judge provided by ASTuner. The judge reads `final_answer` from `metadata` and compares it with the ground-truth answers in the dataset to produce a score.

## Configure required parameters

Next, to actually start training, we need to add several required configuration parameters.

### Pretrained model

In `math_agent.yaml`, specify the LLM used by the agent, i.e., the model we are going to train:

```yaml
astuner:
  model:
    path: Qwen/Qwen2.5-14B-Instruct
```

The `path` can be a remote Hugging Face repo, or a local directory path.

### Training hyperparameters

We also need to configure several important training hyperparameters:

```yaml
astuner:
  # ...
  rollout:
    agentscope_workflow: math_agent.math_agent->MathAgentWorkflow
    # Model temperature
    temperature: 0.7
    # Maximum number of parallel rollout workers
    max_env_worker: 64
    num_repeat: 4
    # Forced reward when the model outputs gibberish
    agent_madness_reward: 0.0
    # vLLM tensor parallelism
    tensor_model_parallel_size: 1
    # Maximum number of parallel sequences in vLLM
    max_num_seqs: 40
    multi_turn:
      max_sample_per_task: 4
    # Types of gibberish to check for: meaningless content, wrong tool calls
    compute_madness_checklist:
      - "nonsense"
      - "wrong_toolcall"
    # Maximum response length in a single interaction
    max_response_length_in_one_turn: 1024
    # Maximum context length of the model
    max_model_len: 13000

  trainer_common:
    # Save interval (save every n steps)
    save_freq: 99999
    # Evaluation interval (evaluate every n steps)
    test_freq: 99999
    # Total number of training epochs
    total_epochs: 99999
    # Number of vLLM engines, must be divisible by tensor_model_parallel_size
    trinity_only__n_vllm_engine: 2

trinity:
  trainer:
    max_token_len_per_gpu: 13000
```

## Debug

At this point, all the code and configuration required to train an agent are ready.

Next, we will start the training process in debug mode to check whether there are any issues. You can also skip this step and directly start training.

Start training in debug mode:

```bash
astuner --conf math_agent/math_agent.yaml --backbone='debug' --with-logview
```

In debug mode, the Ray cluster will not be started, which is very suitable for single-machine debugging. In addition, you can configure `launch.json` in VS Code to conveniently debug with breakpoints:

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

## Training

After debugging, you can start training:

```bash
astuner --conf math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

You can find training logs and checkpoints in the `./launcher_record/{exp_yaml_file_name}` directory.


## Read more

We provide more detailed explanations for ASTuner's core concepts and advanced usage:

- [Data](data_pipeline.md): reading training & test data
- [Data Generation](data_generation.md): synthesizing training data from scratch or from small seed sets
- [Tracing-Feedback Loop](./example_tracing_feedback_loop.md): iteratively training from logs of deployed agents

In addition, we provide several other use cases:

- [Math Agent](example_math_agent.md): the math agent training process described in this document
- [Appworld Agent](example_app_world.md): training agents that operate complex apps to complete complex tasks
- [Werewolves](example_werewolves.md): training multi-agent cooperation/competition agents for the Werewolves game
