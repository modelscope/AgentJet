# Introduction

**AgentScope Tuner (ASTuner)** is a cutting-edge, user-friendly training framework designed to optimize AgentScope agents and workflows, fine-tuning language model weights behind the scenes.

Simply provide your AgentScope workflow, training data, and reward function, and we will be ready to enhance your agents to their optimal performance!

---

### ✨ Features

We aim to build a easy-to-learn AgentScope tuner that unlock more possibilities for agent developers:

- **Easy and Friendly**. ASTuner helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. ASTuner provides a rich library of [examples](https://github.com/agentscope-ai/agentscope-tuner/tree/main/tutorial) as tutorials.
- **Efficient and Scalable**. ASTuner uses [trinity](https://github.com/modelscope/Trinity-RFT/) as the default backbone (`--backbone=trinity`), accelerating your tuning process via fully asynchronous RFT. Nevertheless, if actor colocating is your preference, you can still fall back to the [verl](./installation.md) backbone.
- **Flexible and Fast**. ASTuner supports [multi-agent workflows](docs/en/workflow.md) and adopts a timeline merging technique, accelerating training by 1.5x to 20x when the workflow involves multi-turn (or multi-agent) conversations.
- **Reliability and Reproducibility**. Our team keeps track of framework performance across multiple [tasks + major-git-version + training-backbones](https://benchmark.agent-matrix.com/) (under construction, still gathering data, comming soon).

For advanced researchers, ASTuner also provides high-resolution logging and debugging solutions:
<!-- For advanced researchers, ASTuner provides high-resolution logging and debugging solutions that are, to our knowledge, unprecedented in other prior projects. -->

- **High-Resolution Logging**: ASTuner allows users to save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.
- **Fast Debugging**: ASTuner also provides the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes and enabling breakpoint debugging in IDEs.

---

### 🚀 Quick Start

#### Installation

We recommend using `uv` for dependency management.

1. **Clone the Repository**:
```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner

```


2. **Set up Environment**:
```bash
uv venv --python=3.10.16 && source .venv/bin/activate
uv pip install -e .[trinity]
# Note: flash-attn must be installed after other dependencies
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir

```


#### Run Training

You can start training your first agent with a single command using a pre-configured YAML file:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```
Details in [Math agent](./example_math_agent.md)

#### Example Library

Explore our rich library of examples to kickstart your journey:

- 🔢 [**Training a math agent that can write python code**](./example_math_agent.md).
- 📱 [**Creating an AppWorld agent using AgentScope and training it**](./example_app_world.md).
- 🐺 [**Developing Werewolves RPG agents and training them**](./example_werewolves.md).
- 👩🏻‍⚕️ [**Learning to ask questions like a doctor**](./example_learning_to_ask.md).
- 🎴 [**Writing a countdown game using AgentScope and solving it**](./example_countdown.md).
- 🚶 [**Solving a frozen lake walking puzzle using ASTuner**](./example_frozenlake.md).


---

### 🧩 Core Concepts

ASTuner makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>

</div>

#### 1. The User-Centric Interface

To optimize an agent, you provide three core inputs:

* **Workflow**: User-defined agent logic (single or multi-agent).
* **Task Dataset**: Training and validation tasks (e.g., GSM8K).
* **Task Judge**: A reward function or rubric used to score agent performance.

#### 2. Internal System Architecture

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

* **Launcher**: Manages background service processes (Ray, vLLM) and routes the backbone.
* **Task Reader**: Handles data ingestion, augmentation, and filtering.
* **Task Rollout**: Bridges LLM engines and manages the Gym environment lifecycle.
* **Task Runner**: Executes the AgentScope workflow and calculates rewards.
* **Model Tuner**: Forwards inference requests from the workflow to the LLM engine.
* **Context Tracker**: Monitors LLM calls and automatically merges shared-history timelines to improve training efficiency by **3x to 10x**.


---

### 🚦 Navigation

* 📖 **[Tutorials]**: From [Installation](./installation.md) to [Tuning your first agent](./tutorial.md) — the essential path for beginners.
* 🛠️ **[Core Components]**: Define your [Trainable Workflow](./workflow.md) and manage [Data](./data_pipeline.md) and [Reward](./tune_your_first_agent.md).
* ⚙️ **[Deep Dive]**: Master advanced [Configuration](./configuration.md).
* 💡 **[Examples]**: Check the [Example Library](#example-library) above for real-world cases like [Math](./example_math_agent.md), [Werewolves game](./example_werewolves.md) and  [Learning to ask task](./example_learning_to_ask.md).
