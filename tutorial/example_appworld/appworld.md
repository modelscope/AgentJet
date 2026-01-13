## Run Appworld AgentScope Agent

### 1. Prepare dataset

Please download `env_service` and `appworld`. For specific steps, please refer to [EnvService Documentation](https://code.alibaba-inc.com/EconML/EnvService)


### 2. Prepare AgentScope Workflow

See `tutorial/math_agent.py` for details. You can create new AgentScope Workflow code anywhere in the project

- Define AgentScope Workflow (Change the agent's model to `ajet_proxy`)

```python

agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg['content'],
    model=ajet_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

for _ in range(config.ajet.rollout.multi_turn.max_steps):
    # agentscope deal with interaction message
    reply_message = await agent(interaction_message)
    # env service protocol
    obs, _, terminate, _ = ajet_proxy.gym_step(action={"content": reply_message.content, "role": "assistant"})
    # generate new message from env output
    interaction_message = Msg(name="env", content=obs, role="user")
    # is terminated?
    if terminate: break
    if ajet_proxy.context_overflow: break

```

- Among them, some interfaces used by `ajet_proxy` to interact with the agentscope runtime environment are as follows:
    - `ajet_proxy.gym_step`: Simulates the gym interface, inputs action, outputs (observation, reward, terminate_flag, info) tuple
    - `ajet_proxy.context_overflow`: Queries whether the current context window has token overflow

### 3. Prepare Judge (Reward Module)

In `ajet/task_judge/env_service_as_judge.py`, we directly send an http request to env_service to read the reward.

Judge returns: raw_reward, is_success


### 4. Testing


4.1 Copy and modify key parameters in [tutorial/example_appworld/appworld.yaml](../tutorial/example_appworld/appworld.yaml). The parts most relevant to this document in the yaml have been marked with ✨✨✨✨ symbols

1. Read task (corresponding configuration field `ajet.task_reader`)
2. Define Workflow (corresponding configuration field `ajet.rollout.user_workflow`)
    - For example, if the agentscope workflow is defined in the `ExampleAgentScopeWorkflow` class of `tutorial/appworld.py`
    - Then fill in `ajet.rollout.user_workflow=tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow`
3. Define scoring function (corresponding configuration field `ajet.task_judge.judge_protocol`)
    - Fill in `ajet.task_judge.judge_protocol=ajet.task_judge.env_service_as_judge->EnvServiceJudge`
4. Specify model (corresponding configuration field `ajet.model.path`)

```yaml
ajet
  project_name: appworld_ajet
  experiment_name: "read_yaml_name"
  task_judge:
    # ✨✨✨✨ Write and select evaluation function
    judge_protocol: ajet.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # ✨✨✨✨ Set model to be trained
    path: YOUR_MODEL_PATH
  rollout:
    # ✨✨✨✨ Write and select Agent
    user_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
    force_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```


4.2 Full-link debugging (Quick debugging without ray: --backbone='debug')
```bash
# (Training math agent demo) It is recommended to kill all ray and env_service processes before starting ( ajet --kill="python|ray" )
clear && ajet --conf tutorial/example_appworld/appworld.yaml --backbone='debug' --with-logview

```
Note: When --backbone=debug, the program no longer uses ray. You can write vscode's launch.json for convenient breakpoint debugging. launch.json configuration:
```json
{

    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Launch rollout",
            "type": "debugpy",
            "request": "launch",
            "program": "ajet/cli/launcher.py",
            "console": "integratedTerminal",
            "args": [
                "--backbone",  "debug",
                "--with-appworld",
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        },
    ]
}
```


4.3 When debugging is complete, start training (just switch backbone: --backbone='verl')
```bash
# It is recommended to kill all ray, vllm, and env_service processes before starting ( ajet --kill="python|ray|vllm" )
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='verl'
```


### 5. Read Rollout Log

<div align="center">
  <img src="tutorial/figure/best-logger.png" alt="Log Interface">
</div>

- Find the log folder, default is under `./saved_experiments/exp_yaml_file_name/*`
- Run `beast_logger_go` to start the log browser, vscode port mapping 8181 port
```bash
root@xxxx:/xxx/xxx/xxx# beast_logger_go
INFO:     Started server process [74493]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8181 (Press CTRL+C to quit)
```
- Open http://127.0.0.1:8181, prompt to enter the log file path, fill in the **absolute path** of the log folder, the following forms are acceptable
    - /ajet/ajet/saved_experiments
    - /ajet/ajet/saved_experiments/exp_yaml_file_name
    - /ajet/ajet/saved_experiments/exp_yaml_file_name/2025_11_10_02_52/rollout

- Open the log file target on the **left**, the log entry in the **middle**, and the interaction record on the **right** of the interface to display the complete trajectory

- Blue Token represents Token involved in loss calculation, yellow is the opposite

- Hover over the Token to view the Token's **logprob** (currently limited to trinity backbone)


### 6. Reference Training Curve


<div align="center">
  <img src="tutorial/figure/appworld.png" alt="Training Curve">
</div>
