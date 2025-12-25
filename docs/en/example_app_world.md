# AppWorld

This tutorial demonstrates how to train an agent to interact with AppWorld and solve complex tasks through tool usage.

## 1. Overview

AppWorld is a high-fidelity execution environment of 9 day-to-day apps, operable via 457 APIs, populated with digital activities of 106 people living in a simulated world. The goal is to tune an agent that can effectively navigate and utilize these apps to complete complex tasks.

This document is organized as follows:

- Quick Start: run the example with minimal setup
- Understand: workflow loop, configuration, code locations, and reward
- Results: training curve and qualitative cases

## 2. Quick Start

### 2.1 Preparation

First, download and unpack the Appworld services. The script below is idempotent: it clears any existing folder and re-downloads the archive.

```bash
base_path="/tmp"
export APPWORLD_PATH="${base_path}/pack_all_in_one"
export APPWORLD_SCRIPT="bash EnvService/env_sandbox/appworld.sh"

rm -rf "${APPWORLD_PATH}"
rm -f ./appworld_pack_v2.tar.gz

wget -q "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v2.tar.gz" -O appworld_pack_v2.tar.gz
tar -xzf ./appworld_pack_v2.tar.gz -C "${base_path}"
```

Then export the environment variables (re-run in every new shell):

```bash
export BASE_PATH=/tmp
export APPWORLD_PATH="${BASE_PATH}/pack_all_in_one"
export APPWORLD_SCRIPT="bash EnvService/env_sandbox/appworld.sh"
```

### 2.2 Start Training

Run the training script:

```bash
astuner --conf tutorial/example_appworld/appworld.yaml --backbone='trinity' --with-ray
```

<details>
<summary>Quick Debugging (Optional)</summary>

If you want to breakpoint-debug the workflow/judge locally:

```bash
# (optional) recommended cleanup before debug
# astuner --kill="python|ray"

clear && \
astuner --conf tutorial/example_appworld/math_agent.yaml --backbone='debug' --with-logview
```

When `--backbone=debug`, Ray is disabled. You can use a VSCode `launch.json` like below:

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
        "--conf", "./path/to/yaml.yaml"
      ],
      "env": {}
    }
  ]
}
```
</details>

## 3. Understand

This section explains how the AppWorld example is assembled: workflow, reward, configuration, and code locations.

### 3.1 Core Process

The AgentScope workflow code for the AppWorld example is located at `tutorial/example_appworld/appworld.py`.

The code first defines the AgentScope workflow (set the agent's `model` to `model_tuner`):

```python
agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg["content"],
    model=model_tuner,
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

env = workflow_task.gym_env

for step in range(model_tuner.config.astuner.rollout.multi_turn.max_steps):
    # agentscope deal with interaction message
    reply_message = await agent(interaction_message)
    # env service protocol
    obs, _, terminate, _ = env.step(
        action={"content": reply_message.content, "role": "assistant"}
    )
    # generate new message from env output
    interaction_message = Msg(name="env", content=obs, role="user")
    # is terminated?
    if terminate:
        break
    if model_tuner.get_context_tracker().context_overflow:
        break
```

In the above code:

- `env.step`: simulates the gym interface. It takes an action as input and returns a four-tuple `(observation, reward, terminate_flag, info)`.
- `model_tuner.get_context_tracker().context_overflow`: checks whether the current context window has exceeded the token limit.


### 3.2 Reward

In `astuner/task_judge/env_service_as_judge.py`, we read the reward signal from the environment via `env.evaluate(...)`.

You can also refer to this file to implement your own Judge for your specific task.

### 3.3 Configuration Details
Copy and modify the key parameters in `tutorial/example_appworld/appworld.yaml`. The parts most relevant to this document are marked with ✨✨✨✨ in the yaml file:

1. **Read tasks** (corresponding config field: `astuner.task_reader`)
2. **Define the workflow** (corresponding config field: `astuner.rollout.agentscope_workflow`)
    - Example: if the AgentScope workflow is defined in the `ExampleAgentScopeWorkflow` class in `tutorial/example_appworld/appworld.py`
    - Then set `astuner.rollout.agentscope_workflow = "tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow"`
3. **Define the scoring function** (corresponding config field: `astuner.task_judge.judge_protocol`)
    - Example: `astuner.task_judge.judge_protocol = "astuner.task_judge.env_service_as_judge->EnvServiceJudge"`
4. **Specify the model** (corresponding config field: `astuner.model.path`)

```yaml
astuner:
  project_name: example_appworld
  experiment_name: "read_yaml_name"
  task_judge:
    # ✨✨✨✨ Implement and select the evaluation function
    judge_protocol: agentscope_tuner.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # ✨✨✨✨ Set the model to be trained
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ Implement and select the Agent
    agentscope_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
    agentscope_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```

## 4. Results

### 4.1 Training Curve

![Training curve (small batch)](https://img.alicdn.com/imgextra/i2/O1CN01toRt2c1Nj8nKDqoTd_!!6000000001605-2-tps-1410-506.png)

> **Visualization:** Training curves are generated by SwanLab. See [Visualization Tools](./visualization.md) for setup and usage.

As training progresses, reward increases. This usually means the agent becomes more stable on **two things**:

* **Following correct API protocols**: it learns to look up API documentation before calling, and uses valid API endpoints instead of hallucinating non-existent ones.
* **Completing multi-step workflows**: it can properly obtain access tokens and chain multiple API calls to accomplish complex tasks.


### 4.2 Case Study

#### Before tuning:

1. Frequently call non-existent APIs

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN015FgjqI20Ip3AJybr0_!!6000000006827-2-tps-1259-683.png)

The agent hallucinates API names without checking whether they exist, leading to repeated failures.

2. Fail to follow the instructions to obtain an access token

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN01bGZ1s01VyjCSrTJte_!!6000000002722-2-tps-1181-954.png)

The agent attempts to call protected APIs without first obtaining the required access token, resulting in authentication errors.

#### After tuning:

1. Look up the API documentation first, and learn to use valid APIs

![After tuning](https://img.alicdn.com/imgextra/i4/O1CN01VRIDy922PoKD1bETl_!!6000000007113-2-tps-1180-944.png)

The agent now checks available APIs before making calls, avoiding hallucinated endpoints.

2. Learn to obtain an access token correctly

![After tuning](https://img.alicdn.com/imgextra/i2/O1CN01xiF9UU20h62dyrZ4x_!!6000000006880-2-tps-1182-793.png)

The agent properly handles the authentication step before accessing protected APIs.

> **Token-level Visualization:** These detailed logs are generated by Beast-Logger. See [Beast-Logger Usage](./beast_logger.md) for more details.
