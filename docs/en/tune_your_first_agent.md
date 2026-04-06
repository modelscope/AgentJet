# Tune Your First Agent

In this document, we demonstrate how to implement and train, from scratch, an agent that can use Python to perform calculations and solve 'gsm8k' math problems.

AgentJet provides **two training modes**. Choose the one that fits your needs:

!!! tip "Which Mode Should I Choose?"
    - **Classic Mode**: Simple, all-in-one solution. Start here if you're new to AgentJet.
    - **Swarm Mode**: Advanced distributed training. Run agent code on your laptop while training happens on remote GPUs.

<div class="card-grid">
<a href="#classic-mode-tutorial" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-agent" alt=""><h3>Classic Mode Tutorial</h3></div><p class="card-desc">Centralized training - everything runs in one process on GPU machine.</p></a>
<a href="#swarm-mode-tutorial" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cloud-sync.svg" class="card-icon card-icon-math" alt=""><h3>Swarm Mode Tutorial</h3></div><p class="card-desc">Distributed training - develop on laptop, train on remote GPU cluster.</p></a>
</div>

---

# Classic Mode Tutorial

Classic Mode is the simplest way to train an agent. Everything runs in a single process on a GPU machine.

## Classic Mode Pipeline

<div class="workflow-single">
<div class="workflow-header">Classic Mode Training Pipeline</div>

<div class="workflow">
<ol class="workflow-steps">

<li><strong>Define agent workflow</strong>

Create your agent using AgentScope/Langchain/OpenaiSDK or only http requests, wrap it in a Workflow class.</li>

<li><strong>Define reward</strong>

Configure how the agent's outputs are evaluated and scored.</li>

<li><strong>Prepare dataset</strong>

Set up the dataset and configure the task reader.</li>


<li><strong>Debug (Optional)</strong>

Test your workflow in debug mode before full training.</li>
<li><strong>Start training</strong>

Launch the training process and track progress.</li>
</ol>
</div>
</div>


!!! info ""
    Checkout the full code of this example by [clicking here](#classic-mode-full-code)


## Step 1: Define agent Workflow + Reward


First of all, create a directory for this training project:

```bash
tutorial/example_math_agent
├── math_agent.py
└── math_agent.yaml
```

Next, define your workflow (or convert an existing workflow). Here we use AgentScope to implement this agent. You can toggle two code before and after convertion to see the difference. If you prefer langchain or openai sdk, [please refer to this article](agent_framework_support.md).

=== "`math_agent.py` - AgentJet Workflow (After Convertion)"

    ```python title="math_agent.py"
    class MathToolWorkflow(Workflow): # ✨✨ inherit `Workflow` class
        name: str = "math_agent_workflow"

        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
            # run agentscope
            query = workflow_task.task.main_query
            self.toolkit = Toolkit()
            self.toolkit.register_tool_function(execute_python_code)
            self.agent = ReActAgent(
                name="math_react_agent", sys_prompt=system_prompt,
                model=tuner.as_agentscope_model(),  # ✨✨ compared with a normal agentscope agent, here is the difference!
                formatter=DashScopeChatFormatter(),
                toolkit=self.toolkit,
                memory=InMemoryMemory(), max_iters=2,
            )
            self.agent.set_console_output_enabled(False)
            msg = Msg("user", query, role="user")
            result = await self.agent.reply(msg)
            final_answer = extract_final_answer(result)

            # compute reward
            reference_answer = workflow_task.task.metadata["answer"].split("####")[-1].strip()
            match = re.search(r"\\boxed\{([^}]*)\}", final_answer)
            if match: is_success = (match.group(1) == reference_answer)
            else:     is_success = False
            return WorkflowOutput(reward=(1.0 if is_success else 0.0), metadata={"final_answer": final_answer})

    ```


=== "Original Workflow (Before Convertion)"

    ```python title="math_agent.py"
    class MathToolWorkflow(object):
        name: str = "math_agent_workflow"

        async def execute(self, workflow_task: WorkflowTask) -> WorkflowOutput:
            # run agentscope
            query = workflow_task.task.main_query
            self.toolkit = Toolkit()
            self.toolkit.register_tool_function(execute_python_code)
            self.agent = ReActAgent(
                name="math_react_agent", sys_prompt=system_prompt,
                model=DashScopeChatModel(model='qwen-max'),
                formatter=DashScopeChatFormatter(),
                toolkit=self.toolkit,
                memory=InMemoryMemory(), max_iters=2,
            )
            self.agent.set_console_output_enabled(False)
            msg = Msg("user", query, role="user")
            result = await self.agent.reply(msg)
            final_answer = extract_final_answer(result)

            # compute reward
            reference_answer = workflow_task.task.metadata["answer"].split("####")[-1].strip()
            match = re.search(r"\\boxed\{([^}]*)\}", final_answer)
            if match: is_success = (match.group(1) == reference_answer)
            else:     is_success = False
            return WorkflowOutput(reward=(1.0 if is_success else 0.0), metadata={"final_answer": final_answer})

    ```



## Step 2: Prepare dataset

!!! info "Data Sources"
    AgentJet provides multiple ways to read data:

    - Read from local files on disk
    - Read from a Hugging Face repo
    - Read from an EnvService


Download the `openai/gsm8k` dataset:

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

Now, we have obtained all materials required to train the agent.


=== "`math_agent.yaml` - Configuration Yaml"

    ```yaml
    # ------------------ main configuration ------------------
    ajet:
      project_name: example_math_agent
      task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` or `dataset_file` or `huggingface_dat_repo`
        # effective when `type: huggingface_dat_repo`
        huggingface_dat_repo:
          dataset_path: 'openai/gsm8k'
          training_split: "train"
          validation_split: "test"

      task_judge:
        # ✨✨✨✨ null, because in this certain case, we write reward function together with workflow
        judge_protocol: null

      model:
        # ✨✨✨✨ set the model to be trained
        path: Qwen/Qwen2.5-7B

      rollout:
        user_workflow: "tutorial.example_math_agent.math_agent->ExampleMathLearn" # ✨✨✨✨ write and select workflow
        num_repeat: 6 # grpo `n`
        tensor_model_parallel_size: 1 # vllm tp
        max_response_length_in_one_turn: 1024
        max_model_len: 10000

      data:
        train_batch_size:    100
        max_prompt_length:   3000
        max_response_length: 7000

      debug:
        debug_max_parallel: 1
        debug_first_n_tasks: 1

      trainer_common:
        save_freq: 100
        test_freq: 100
        total_epochs: 100
        logger: swanlab

    # ------------------ do not modify ------------------
    hydra:
      searchpath:
        - file://ajet/default_config
        - file://ajet/default_config/verl
        - file://ajet/default_config/trinity

    # ------------------ do not modify ------------------
    defaults:
      - - trinity_default verl_default

      - ajet_default
      - _self_

    ```

### Configuration Parameters

| Category | Parameter | Description | Example Value |
|----------|-----------|-------------|---------------|
| **Project** | `project_name` | Name of the training project | `example_math_agent` |
| **Task Reader** | `type` | Type of data source to read tasks from | `huggingface_dat_repo` (options: `env_service`, `dataset_file`, `huggingface_dat_repo`) |
| | `dataset_path` | Path or identifier of the dataset | `openai/gsm8k` |
| | `training_split` | Dataset split used for training | `train` |
| | `validation_split` | Dataset split used for validation/testing | `test` |
| **Model** | `path` | Path or identifier of the model to be trained | `Qwen/Qwen2.5-7B` |
| **Rollout** | `user_workflow` | Python module path to the workflow class | `tutorial.example_math_agent.math_agent->ExampleMathLearn` |
| | `num_repeat` | Number of rollout repeats per task (GRPO `n` parameter) | `6` |
| | `tensor_model_parallel_size` | vLLM tensor parallelism size | `1` |
| | `max_response_length_in_one_turn` | Maximum token length for a single agent response | `1024` |
| | `max_model_len` | Maximum total context length for the model | `10000` |
| **Data** | `train_batch_size` | Number of tasks per training batch | `100` |
| | `max_prompt_length` | Maximum token length for input prompts | `3000` |
| | `max_response_length` | Maximum token length for model responses | `7000` |
| **Debug** | `debug_max_parallel` | Maximum parallel workers in debug mode | `1` |
| | `debug_first_n_tasks` | Number of tasks to process in debug mode | `1` |
| **Trainer** | `save_freq` | Frequency (in steps) to save model checkpoints | `100` |
| | `test_freq` | Frequency (in steps) to run validation | `100` |
| | `total_epochs` | Total number of training epochs | `100` |
| | `logger` | Logging backend for experiment tracking | `swanlab` |
| **Task Judge** | `judge_protocol` | Protocol for judging task completion | `null` (reward is computed in workflow) |


## Step 3: Debug (Optional)

Before full training, you can run some test in debug mode, using raw base model to test whether bug exists.
We choose VSCode to debug because it is open-source and fast.


!!! tip "VS Code Debugging"
    - You can create `.vscode/launch.json` for breakpoint debugging:

    ```json
    {
      "version": "0.2.0",
      "configurations": [
        {
          "name": "Python Debugger: Launch rollout",
          "type": "debugpy",
          "request": "launch",
          "module": "ajet.launcher",
          "console": "integratedTerminal",
          "args": [
            "--backbone", "debug",
            "--conf", "tutorial/example_math_agent/math_agent.yaml"
          ],
          "env": {}
        }
      ]
    }
    ```

After `.vscode/launch.json` is created, press `F5` to start debugging. (Do not forget to configure python venv path in VSCode.)

For more debugging techniques, please refer to [debugging guidelines](debugging_guide.md).


## Step 4: Start Training

After debugging, launch the full training:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml
```

!!! success "Output Location"
    Training logs and checkpoints will be saved default to:
    ```
    ./saved_experiments/{exp_yaml_file_name}/
    ```


## Classic Mode Full Code {#classic-mode-full-code}

=== "`tutorial/example_math_agent/math_agent.py` - AgentJet Workflow (After Convertion)"

    ```python
    import re
    from loguru import logger
    from agentscope.message import Msg
    from agentscope.agent import ReActAgent
    from agentscope.formatter import DashScopeChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit, execute_python_code
    from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask


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


    system_prompt = """
    You are an agent specialized in solving math problems with tools.
    Please solve the math problem given to you.
    You can write and execute Python code to perform calculation or verify your answer.
    You should return your final answer within \\boxed{{}}.
    """


    class MathToolWorkflow(Workflow): # ✨✨ inherit `Workflow` class
        name: str = "math_agent_workflow"

        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
            # run agentscope
            query = workflow_task.task.main_query
            self.toolkit = Toolkit()
            self.toolkit.register_tool_function(execute_python_code)
            self.agent = ReActAgent(
                name="math_react_agent", sys_prompt=system_prompt,
                model=tuner.as_agentscope_model(),  # ✨✨ compared with a normal agentscope agent, here is the difference!
                formatter=DashScopeChatFormatter(),
                toolkit=self.toolkit,
                memory=InMemoryMemory(), max_iters=2,
            )
            self.agent.set_console_output_enabled(False)
            msg = Msg("user", query, role="user")
            result = await self.agent.reply(msg)
            final_answer = extract_final_answer(result)

            # compute reward
            reference_answer = workflow_task.task.metadata["answer"].split("####")[-1].strip()
            match = re.search(r"\\boxed\{([^}]*)\}", final_answer)
            if match: is_success = (match.group(1) == reference_answer)
            else:     is_success = False
            return WorkflowOutput(reward=(1.0 if is_success else 0.0), metadata={"final_answer": final_answer})

    ```

=== "`tutorial/example_math_agent/math_agent.yaml` - Configuration Yaml"

    ```yaml
    # ------------------ main configuration ------------------
    ajet:
      project_name: example_math_agent
      task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` or `dataset_file` or `huggingface_dat_repo`
        # effective when `type: huggingface_dat_repo`
        huggingface_dat_repo:
          dataset_path: 'openai/gsm8k'        # '/mnt/data_cpfs/dataset_cache/openai/gsm8k/main'
          training_split: "train"
          validation_split: "test"

      model:
        # ✨✨✨✨ set the model to be trained
        path: Qwen/Qwen2___5-7B-Instruct      # /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct

      rollout:
        user_workflow: "tutorial/example_math_agent/math_agent.py->MathToolWorkflow" # ✨✨✨✨ write and select workflow
        num_repeat: 6 # grpo `n`
        tensor_model_parallel_size: 1 # vllm tp
        max_response_length_in_one_turn: 1024
        max_model_len: 10000

      task_judge:
        # ✨✨✨✨ null, because in this certain case, we write reward function together with workflow
        judge_protocol: null

      data:
        train_batch_size:    100
        max_prompt_length:   3000
        max_response_length: 7000

      debug:
        debug_max_parallel: 1
        debug_first_n_tasks: 1

      trainer_common:
        save_freq: 100
        test_freq: 100
        total_epochs: 100
        logger: swanlab

    # ------------------ do not modify ------------------
    hydra:
      searchpath:
        - file://ajet/default_config
        - file://ajet/default_config/verl
        - file://ajet/default_config/trinity

    # ------------------ do not modify ------------------
    defaults:
      - verl_default
      - trinity_default
      - ajet_default
      - _self_

    ```

---

# Swarm Mode Tutorial

Swarm Mode enables distributed training. Run your agent code on a laptop while training happens on a remote GPU cluster. This completely decouples training from sampling.

## Swarm Mode Pipeline

<div class="workflow-single">
<div class="workflow-header">Swarm Mode Training Pipeline</div>

<div class="workflow">
<ol class="workflow-steps">

<li><strong>Start Swarm Server</strong>

Launch the training server on a GPU machine.</li>

<li><strong>Create Swarm Client</strong>

Write a client script that connects to the server and runs your agent workflow.</li>

<li><strong>Run Training</strong>

Launch the client to start distributed training.</li>
</ol>
</div>
</div>

!!! info ""
    This tutorial uses the GSM8K math dataset as an example. The client code is available at `tutorial/example_math_swarm/math.py`.


## Step 1: Start Swarm Server

On your GPU machine (or cluster), start the Swarm Server:

```bash
# Start the swarm server
ajet-swarm start

# Open the monitoring dashboard in another terminal
ajet-swarm overwatch --swarm-url=http://localhost:10086
```

!!! tip "Custom Port"
    Use `--swarm-port` to change the default port (10086):
    ```bash
    ajet-swarm start --swarm-port=10086
    ```

The Swarm Server will:

- Load the model specified by the client
- Provide vLLM API endpoints for inference
- Compute gradients and update model parameters
- Track training progress


## Step 2: Create Swarm Client

Create your client script. The client reads the dataset, runs the agent workflow, computes rewards, and sends results back to the server.

=== "`tutorial/example_math_swarm/math.py` - Swarm Client"

    ```python
    import os
    import re
    import requests
    from textwrap import dedent
    from ajet.schema.task import Task, WorkflowOutput
    from ajet.copilot.job import AgentJetJob
    from ajet.task_reader import RouterTaskReader
    from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
    from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
    from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
    from ajet.tuner_lib.experimental.swarm_client import SwarmClient

    # Configuration
    GRPO_N = 4  # grpo group size
    NUM_EPOCH = 10000
    AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
    REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/path/to/your/model")
    REMOTE_BATCH_SIZE = 32
    REMOTE_ALLOCATE_GPU_PER_NODE = 8

    def main():
        # Initialize dataset reader
        dataset = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(
                    dataset_path="openai/gsm8k",  # Or use local path: "/root/agentjet/benchmark_datasets/dataset/gsm8k/socratic"
                )
            )
        )

        # Connect to swarm server and configure training
        swarm_worker = SwarmClient(AJET_SWARM_URL)
        swarm_worker.auto_sync_train_config_and_start_engine(
            AgentJetJob(
                experiment_name="math_gsm8k_grpo",
                algorithm="grpo",
                n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
                model=REMOTE_MODEL_PATH,
                batch_size=REMOTE_BATCH_SIZE,
                num_repeat=GRPO_N,
            ),
            force_restart=True,
        )

        # Define rollout function
        def rollout(task):
            try:
                # Begin episode - get API endpoint from server
                episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=60)
                # Execute agent workflow
                workflow_output = execute_agent(task, api_baseurl_key)
                # Report result back to server
                swarm_worker.end_episode(task, episode_uuid, workflow_output)
            except:
                pass

        # Run training loop
        executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True)
        for _ in range(NUM_EPOCH):
            for _, task in enumerate(dataset.generate_training_tasks()):
                for _ in range(GRPO_N):
                    executor.submit_with_periodic_drain(fn=rollout, task=task)


    def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
        # Get API endpoint from server
        base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
        query = task.main_query
        reference_answer = task.metadata["answer"]

        # Prepare messages
        messages = [
            {"role": "system", "content": dedent("""You are an agent specialized in solving math problems.
               Please solve the math problem given to you. You can write and execute Python code.
               Return your final answer within \\boxed{}.""")},
            {"role": "user", "content": query}
        ]

        # Call remote model API
        response = requests.post(
            f"{base_url}/chat/completions",
            json={"model": "fill_whatever_model", "messages": messages, "stream": False},
            headers={"Authorization": f"Bearer {api_key}", "Connection": "close"},
            timeout=300,
        )
        response.raise_for_status()
        final_answer = response.json()['choices'][0]['message']['content']

        # Compute reward
        reference_answer = reference_answer.split("####")[-1].strip()
        pattern = r"\\boxed\{([^}]*)\}"
        match = re.search(pattern, final_answer)
        is_success = match.group(1) == reference_answer if match else False
        raw_reward = 1.0 if is_success else 0.0

        return WorkflowOutput(reward=raw_reward, metadata={"final_answer": final_answer})


    if __name__ == "__main__":
        main()
    ```

### Key Components

| Component | Description |
|-----------|-------------|
| `SwarmClient` | Connects to the Swarm Server |
| `auto_sync_train_config_and_start_engine` | Sends training config (model, algorithm, batch size) to server |
| `begin_episode()` | Requests an API endpoint from the server for inference |
| `end_episode()` | Sends the reward back to the server |
| `execute_agent()` | Your agent logic - runs on the client, calls remote API for inference |


## Step 3: Run Training

Run the client on any machine (laptop, workstation, etc.):

```bash
# Set the swarm server URL
export AJET_SWARM_URL="http://<server-ip>:10086"

# Optionally set the model path
export REMOTE_MODEL_PATH="/path/to/your/model"

# Run the client
python tutorial/example_math_swarm/math.py
```

!!! tip "Run Anywhere"
    The client can run on:
    - Your laptop (no GPU needed!)
    - A workstation
    - An ECS instance
    - Any machine with Python and network access

The client will continuously:
1. Read tasks from the dataset
2. Call the remote server for model inference
3. Execute the agent workflow
4. Compute rewards
5. Send results back to the server

The server handles gradient computation and model updates automatically.


## Swarm Mode Full Code {#swarm-mode-full-code}

=== "`tutorial/example_math_swarm/math.py` - Complete Swarm Client"

    ```python
    # -*- coding: utf-8 -*-

    import os
    import re
    import requests
    from textwrap import dedent
    from ajet.schema.task import Task, WorkflowOutput
    from ajet.copilot.job import AgentJetJob
    from ajet.task_reader import RouterTaskReader
    from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
    from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
    from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
    from ajet.tuner_lib.experimental.swarm_client import SwarmClient

    GRPO_N = 4  # grpo group size
    NUM_EPOCH = 10000
    AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
    REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct")
    REMOTE_BATCH_SIZE = 32
    REMOTE_ALLOCATE_GPU_PER_NODE = 8

    def main():

        # Initialize dataset reader
        dataset = RouterTaskReader(
            reader_type = "huggingface_dat_repo",
            reader_config = AjetTaskReader(
                huggingface_dat_repo = HuggingfaceDatRepo(
                    dataset_path = "openai/gsm8k",  # Or use local path: "/root/agentjet/benchmark_datasets/dataset/gsm8k/socratic"
                )
            )
        )

        # Connect to swarm server and configure training
        swarm_worker = SwarmClient(AJET_SWARM_URL)
        swarm_worker.auto_sync_train_config_and_start_engine(
            AgentJetJob(
                experiment_name="math_gsm8k_grpo",
                algorithm="grpo",
                n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
                model=REMOTE_MODEL_PATH,
                batch_size=REMOTE_BATCH_SIZE,
                num_repeat=GRPO_N,
            ),
            force_restart=True,
        )

        def rollout(task):
            try:
                # Begin episode - get API endpoint from server
                episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=60)
                # Execute agent workflow
                workflow_output = execute_agent(task, api_baseurl_key)
                # Report result back to server
                swarm_worker.end_episode(task, episode_uuid, workflow_output)
                return
            except:
                pass

        # Run training loop
        executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True)
        for _ in range(NUM_EPOCH):
            for _, task in enumerate(dataset.generate_training_tasks()):
                for _ in range(GRPO_N):
                    executor.submit_with_periodic_drain(fn=rollout, task=task)

        return None



    def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
        # Get API endpoint from server
        base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
        query, reference_answer = (task.main_query, task.metadata["answer"])

        # Prepare messages
        messages = [
            { "role": "system", "content": dedent("""You are an agent specialized in solving math problems. Please solve the math problem given to you.
               You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.""") },
            { "role": "user", "content": query }
        ]

        # Call remote model API
        response = requests.post(
            f"{base_url}/chat/completions",
            json    = { "model": "fill_whatever_model", "messages": messages, "stream": False },
            headers = { "Authorization": f"Bearer {api_key}", "Connection": "close" },
            timeout = 300,
        )
        response.raise_for_status()
        final_answer = response.json()['choices'][0]['message']['content']

        # Compute reward
        reference_answer = reference_answer.split("####")[-1].strip()
        pattern = r"\\boxed\{([^}]*)\}"
        match = re.search(pattern, final_answer)
        if match: is_success = match.group(1) == reference_answer
        else: is_success = False
        raw_reward = 1.0 if is_success else 0.0

        return WorkflowOutput(reward=raw_reward, metadata={"final_answer": final_answer})



    if __name__ == "__main__":
        main()
    ```



## Next Steps

### Continue Learning

<div class="card-grid">
<a href="../classic_workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Classic Workflow</h3></div><p class="card-desc">Learn to define trainable workflows and multi-agent setups.</p></a>
<a href="../swarm_workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:network.svg" class="card-icon card-icon-general" alt=""><h3>Swarm Workflow</h3></div><p class="card-desc">Distributed training with rollout on separate machines.</p></a>
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Configure data loading from various sources.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions for your training.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent Example</h3></div><p class="card-desc">See the complete Math Agent implementation.</p></a>
</div>

### Explore Swarm Mode

Ready to unlock the full power of distributed training? Explore Swarm mode:

<div class="card-grid">
<a href="../swarm/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cloud-sync.svg" class="card-icon card-icon-agent" alt=""><h3>Swarm Training</h3></div><p class="card-desc">Complete guide to distributed swarm training with server and client setup.</p></a>
<a href="../swarm_best_practice/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:star-circle.svg" class="card-icon card-icon-general" alt=""><h3>Swarm Best Practices</h3></div><p class="card-desc">4 demo scenarios: multi-model, distributed, and multi-task training.</p></a>
<a href="../swarm_deepdive/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:dive.svg" class="card-icon card-icon-data" alt=""><h3>Swarm Deep Dive</h3></div><p class="card-desc">Technical deep dive into swarm architecture and advanced features.</p></a>
<a href="../swarm_intro_blog_en/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:file-document.svg" class="card-icon card-icon-math" alt=""><h3>Swarm Introduction</h3></div><p class="card-desc">Comprehensive introduction comparing classic vs swarm modes.</p></a>
</div>
