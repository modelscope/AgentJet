# Werewolves

This tutorial demonstrates how to train **multiple agents** to play the Werewolves game.

## 1. Overview

The Werewolves role-playing game is a typical POMDP (Partially Observable Markov Decision Process) problem. We can train agents in this cooperative multi-agent problem using shared-parameter methods.

Terms explained:
- **Partially Observable**: Agents are only able to receive **local information**. One agent cannot obtain others' perception, even if they are teammates.
- **Markov Decision Process**: Making decisions according to current situations.
- **Shared-parameter**: Using one model as policy for multiple agents. But notice agents **share** policy (model parameters) but **do not share** perception (model input).
- **Cooperative multi-agent problem**: Agents have aligned interests (reward).
- **Environment**: We use static **`Qwen3-235B-A22B`** as the brain of opponents. We use **`Qwen2-7B`** as the brain of trainable agents (`trainable_targets`).

![image](https://img.alicdn.com/imgextra/i2/O1CN012JgVZC2ABczBhAzJs_!!6000000008165-0-tps-2048-2048.jpg)

This page shows how to use the Werewolves social deduction game as a multi-agent environment to prepare data and environment, write an AgentScope Workflow, configure the reward module (Judge), and complete the full process from local debugging to formal training.

Scenario Overview
- Scenario: Classic Werewolves game, including roles such as werewolf, villager, seer, witch, and hunter.
- Goal: Train a specific role (in this example, the `werewolf`) to achieve a higher win rate in games.

## 2. Quick Start

Start training with the following command:
```
# ( astuner --kill="python|ray|vllm" )
astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='trinity' --with-ray
```

<details>
<summary>Quick Debugging (Optional)</summary>

If you want to breakpoint-debug the workflow/judge locally:

```bash
# (optional) recommended cleanup before debug
# astuner --kill="python|ray"

clear && \
astuner --conf tutorial/example_werewolves/math_agent.yaml --backbone='debug' --with-logview
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

### 3.1 Core Process

At a high level, each training iteration follows this flow:
- The task reader generates a new game setup (players, role assignments, initial state).
- The rollout runs the AgentScope workflow to simulate a full game.
- Agents in `trainable_targets` act using the trainable model (via `model_tuner`), while opponents use the fixed model.
- The environment produces rewards / outcomes for the episode.
- Trajectories are collected and passed to the backbone trainer (e.g., `trinity`) to update the trainable model.

### 3.2 Configuration Details

This section corresponds to `tutorial/example_werewolves/werewolves.yaml`. The key configuration items are as follows:
```yaml
astuner:
  task_reader:
    # random seed to shuffle players
    type: random_dummy
  task_judge:
    # Implement and select the evaluation function
    # (in this example you can first set it to null and rely purely on the rollout's internal reward)
    judge_protocol: null
  model:
    # Set the model to be trained
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # Select the AgentScope Workflow entry
    agentscope_workflow: tutorial.example_werewolves.start->ExampleWerewolves
```

### 3.3 Code Map

- `tutorial/example_werewolves/werewolves.yaml`: connects the task reader, judge, model, and workflow entry.
- `tutorial/example_werewolves/start.py`: the AgentScope workflow implementation (`ExampleWerewolves`).
- `tutorial/example_werewolves/game.py`: the Werewolves game logic implementation.
- `tutorial/example_werewolves/prompt.py`: prompt templates related to the game.
- `tutorial/example_werewolves/structured_model.py`: defines structured output formats for different roles.
- `tutorial/example_werewolves/utils.py`: game state management and helper functions.

### 3.4 Reward

When `judge_protocol: null`, training relies on the reward (or win/loss outcome) produced inside the rollout / environment. In this example, the reward is produced in the workflow in `tutorial/example_werewolves/start.py`.

In `ExampleWerewolves.execute()`, the workflow first runs a full game by calling `werewolves_game(players, roles)`, and obtains `good_guy_win` (whether the good-guy side wins).

Then it uses a **turn-level sparse win/loss reward**:
- If `good_guy_win == True` and the training target is not `werewolf` (i.e., you are training a good-guy role), then `raw_reward = 1` and `is_success = True`.
- If `good_guy_win == False` and the training target is `werewolf` (i.e., you are training a werewolf-side role), then `raw_reward = 1` and `is_success = True`.
- Otherwise, the training side did not win: `raw_reward = 0` and `is_success = False`.

Exception / invalid-behavior penalty:
- If an exception is thrown during the game (e.g., the game cannot proceed), all trainable targets are penalized uniformly: `raw_reward = -0.1` and `is_success = False`.

If you need a more fine-grained evaluation (e.g., giving partial credit for key intermediate decisions instead of only win/loss), implement a custom Judge and enable it via `astuner.task_judge.judge_protocol`.

## 4. Results

### 4.1 Training Curves

`Qwen2-7B` is able to reach about 60% win rate in about 20 steps.

![image](https://img.alicdn.com/imgextra/i3/O1CN01ldZYDT1ZqGLHuwsrS_!!6000000003245-2-tps-2000-839.png)

> **Visualization:** Training curves are generated by SwanLab. See [Visualization Tools](./visualization.md) for setup and usage.

### 4.2 Case Study

#### Behavior Shifts

Significant role-playing improvement is observed during the experiment.

1. For example, when voted out, the original model tends to reveal its identity as `werewolf`, but after fine-tuning, the agent will try to cheat its opponents and protect teammates. For example:

![](https://img.alicdn.com/imgextra/i1/O1CN01v8VqLB1aYEMfzyTHr_!!6000000003341-2-tps-2104-1016.png)

> **Token-level Visualization:** These detailed logs are generated by Beast-Logger. See [Beast-Logger Usage](./beast_logger.md) for more details.

2. The agent develops multiple strategies for winning. For example:
- **Misleading opponents**: "Let's keep an eye on the seer and the witch. They could be werewolves trying to hide".
- **Appealing to reason**: "We need to be wary of fake seers and watch for inconsistencies in stories, Player-Y as hunter should act carefully".

3. Sometimes agents can take advantage of suspicion between non-werewolf players to eliminate opponents.

![](https://img.alicdn.com/imgextra/i2/O1CN01Sx7wkU23pHyPXyqPH_!!6000000007304-2-tps-968-575.png)

#### Expanding Qwen2-7B to Qwen2-14B

![](https://img.alicdn.com/imgextra/i1/O1CN01TLZcQF1FJ1HPbpLfj_!!6000000000465-2-tps-1842-1008.png)
