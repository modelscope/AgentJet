# Frozen Lake

## 1. Introduction

**Frozen Lake** is a classic reinforcement learning task from [Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

In this environment, the agent is placed on a randomly generated frozen lake, which consists of safe ice ( _ ), dangerous holes (O), and a goal (G). The agent's position is marked as P. The goal is to navigate from the starting position P to the goal G while avoiding the holes. The agent can move up, down, left, or right, but due to the slippery nature of the ice, there is a probability of moving in an unintended direction.

This example demonstrates how to create a trainable agent workflow to solve this navigation challenge.

## 2. Quick Start

### 2.1 Prepare the Environment

Install the dependencies required for the Frozen Lake:

```bash
pip install gymnasium[toy_text]
```

### 2.2 Start Training

Use the provided configuration file to quickly start training:

```bash
astuner --conf tutorial/example_frozenlake/frozenlake.yaml --backbone=trinity
```

## 3. Implementation Details

### 3.1 Implement the Frozen Lake Environment

The `FrozenLakeEnv` class in `tutorial/example_frozenlake/frozenlake.py` wraps the Gymnasium Frozen Lake environment, mainly exposing the `step` and `reset` methods.

- The `step` method returns the next state (observation), reward, done flag, and additional info based on the agent's action.
    - observation: The state of the lake after the agent moves, represented as a string, e.g.:
        ```
        _  _  G
        _  _  _
        P  O  O
        ```
    - reward: The reward received after each move. The agent receives 1 for reaching the goal G, otherwise 0.
    - done: Boolean value. True if the agent reaches the goal or falls into a hole, otherwise False.
    - info: Additional information.

- The `reset` method regenerates the lake environment based on user parameters.

### 3.2 Implement the Agent

The `FrozenLakeAgent` class in `tutorial/example_frozenlake/frozenlake.py` implements the agent's decision logic, mainly through the `step` method, which takes the current environment observation as input and returns the chosen action. The core is a ReActAgent.

```python
class FrozenLakeAgent:

    def __init__(self, model: ModelTuner, max_steps: int = 20):
        self.agent = ReActAgent(
            name="frozenlake_agent",
            sys_prompt=SYSTEM_PROMPT,
            model=model,
            formatter=DashScopeChatFormatter(),
            max_iters=2,
        )
        # other initialization code

    async def step(self, current_observation: str) -> str:
        # Step 1: Build user prompt based on current_observation
        # Step 2: Call ReActAgent to get raw response
        # Step 3: Parse response and return action
```

### 3.3 Integrate Environment and Agent as a Workflow

The `FrozenLakeWorkflow` class in `tutorial/example_frozenlake/frozenlake.py` integrates the environment and agent, mainly exposing the `step` and `reset` methods for external interaction.

The core process is as follows:

```python
class FrozenLakeWorkflow(Workflow):

    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # init agent and env
        # self.agent = FrozenLakeAgent(...)
        # self.env = FrozenLakeEnv(...)
        # reset environment and get initial `observation_str`
        rewards = []
        for _ in range(self.max_steps):
            action = await self.agent.step(observation_str)
            observation_str, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        return WorkflowOutput(
            reward=sum(rewards),
        )
```

## 4. Performance

### 4.1 Training Curve

*(To be added)*

### 4.2 Case Study

*(To be added)*
