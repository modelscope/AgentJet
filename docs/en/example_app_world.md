# AppWorld

This page shows how to prepare the environment and data, build the AgentScope workflow, configure the reward module (Judge), and finally complete the full process from debugging to training in the Appworld scenario.

## Scenario Overview
+ **Scenario**: A high-fidelity execution environment of 9 day-to-day apps, operable via 457 APIs, populated with digital activities of 106 people living in a simulated world.
+ **Goal**: Train an agent to correctly use APPs and solve the specific tasks.

## 1. Prepare Dataset and Environment
First, prepare the environment services required by Appworld:

+ Download and deploy `env_service`
+ Download and deploy `appworld`

For detailed installation and startup steps, refer to the [EnvService documentation](https://modelscope.github.io/AgentEvolver/tutorial/install/#step-2-setup-env-service-appworld-as-example).

## 2. Prepare the AgentScope Workflow
The AgentScope workflow code for the Appworld example is located at `tutorial/example_appworld/appworld.py`.

The code first defines the AgentScope workflow (set the agent's `model` to `astune_proxy`):

```python
agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg['content'],
    model=astune_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

for _ in range(config.astune.rollout.multi_turn.max_steps):
    # AgentScope handles the interaction message with the environment
    reply_message = await agent(interaction_message)
    # env_service protocol: send the model output to the environment as an action
    obs, _, terminate, _ = astune_proxy.gym_step(
        action={"content": reply_message.content, "role": "assistant"}
    )
    # Generate a new interaction message from the environment output
    interaction_message = Msg(name="env", content=obs, role="user")
    # Check whether to terminate this rollout
    if terminate:
        break
    if astune_proxy.context_overflow:
        break
```

In this code, `astune_proxy` provides interfaces to interact with the AgentScope runtime environment:

+ `astune_proxy.gym_step`: simulates the gym interface. It takes an action as input and returns a four-tuple `(observation, reward, terminate_flag, info)`.
+ `astune_proxy.context_overflow`: checks whether the current context window has exceeded the token limit.



## 3. Prepare the Judge (Reward Module)
In `astune/task_judge/env_service_as_judge.py`, we directly send HTTP requests to `env_service` and read the reward signal from the environment.

You can also refer to this file to implement your own Judge for your specific task.



## 4. Testing and Training
### 4.1 Configure YAML
Copy and modify the key parameters in `tutorial/example_appworld/appworld.yaml`. The parts most relevant to this document are marked with ✨✨✨✨ in the yaml file:

1. **Read tasks** (corresponding config field: `astune.task_reader`)
2. **Define the workflow** (corresponding config field: `astune.rollout.agentscope_learn_protocol`)
    - Example: if the AgentScope workflow is defined in the `ExampleAgentScopeLearnProtocol` class in `tutorial/example_appworld/appworld.py`
    - Then set
`astune.rollout.agentscope_learn_protocol = "tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol"`
3. **Define the scoring function** (corresponding config field: `astune.task_judge.judge_protocol`)
    - Example:
`astune.task_judge.judge_protocol = "astune.task_judge.env_service_as_judge->EnvServiceJudge"`
4. **Specify the model** (corresponding config field: `astune.model.path`)

```yaml
astune:
  project_name: appworld_astune
  experiment_name: "read_yaml_name"
  task_judge:
    # ✨✨✨✨ Implement and select the evaluation function
    judge_protocol: astune.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # ✨✨✨✨ Set the model to be trained
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ Implement and select the Agent
    use_agentscope_protocol: True
    agentscope_learn_protocol: tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol
    agentscope_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```

### 4.2 Debugging
```bash
# It is recommended to kill all ray and env_service processes before starting
# ( python launcher.py --kill="python|ray" )
python launcher.py --conf tutorial/example_appworld/appworld.yaml --backbone='debug' --with-logview
```

When `--backbone=debug`, the program no longer uses Ray. You can configure VSCode's `launch.json` for convenient breakpoint debugging. Example configuration:

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
                "--with-appworld",
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        }
    ]
}
```

### 4.3 Start Training
After debugging is complete, simply switch the `backbone` to `trinity` to start formal training:

```bash
# It is recommended to kill all ray, vllm, and env_service processes before starting
# ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf tutorial/example_appworld/appworld.yaml --backbone='trinity'
```

## 5 Reference Result

![Training curve (small batch)](https://img.alicdn.com/imgextra/i2/O1CN01toRt2c1Nj8nKDqoTd_!!6000000001605-2-tps-1410-506.png)

## 6 Case Observation

### Before tuning:

1. Keep using api that does not exist

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN015FgjqI20Ip3AJybr0_!!6000000006827-2-tps-1259-683.png)

2. Do not learn to follow instruction to get access token

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN01bGZ1s01VyjCSrTJte_!!6000000002722-2-tps-1181-954.png)

### After tuning:

1. Look up to api document first, and learn to use effective api

![After tuning](https://img.alicdn.com/imgextra/i4/O1CN01VRIDy922PoKD1bETl_!!6000000007113-2-tps-1180-944.png)

2. Learn to get access token

![After tuning](https://img.alicdn.com/imgextra/i2/O1CN01xiF9UU20h62dyrZ4x_!!6000000006880-2-tps-1182-793.png)
