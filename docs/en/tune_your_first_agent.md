# Tune Your First Agent

In this document, we demonstrate how to implement and train, from scratch, an agent that can use Python to perform calculations and solve 'gsm8k' math problems.



<div class="workflow-single">
<div class="workflow-header">Training Pipeline Overview</div>

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



## Step 1: ✨Define agent Workflow + Reward


First of all, create a directory for this training project:

```bash
tutorial/example_math_agent
├── math_agent.py
└── math_agent.yaml
```

Next, define your workflow (or convert an existing workflow). Here we use AgentScope to implement this agent. You can toggle two code before and after convertion to see the difference. If you prefer langchain or openai sdk, [please refer to this article](../agent_framework_support).

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



## Step 2: ✨Prepare dataset

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
      - verl_default
      - trinity_default
      - ajet_default
      - _self_

    ```



## Step 6: Debug (Optional)

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

For more debugging techniques, please refer to [debugging guidelines](../debugging_guide).


## Step 7: Start Training

After debugging, launch the full training:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml
```

!!! success "Output Location"
    Training logs and checkpoints will be saved default to:
    ```
    ./saved_experiments/{exp_yaml_file_name}/
    ```


## Next Steps

<div class="card-grid">
<a href="../workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Workflow</h3></div><p class="card-desc">Learn to define trainable workflows and multi-agent setups.</p></a>
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Configure data loading from various sources.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions for your training.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent Example</h3></div><p class="card-desc">See the complete Math Agent implementation.</p></a>
</div>
