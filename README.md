# AgentJet

[![Benchmarking](https://img.shields.io/badge/Benchmarking-0078D4?style=for-the-badge&logo=github)](https://benchmark.agent-matrix.com/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Guide-0A7ECC?style=for-the-badge&logo=readthedocs&logoColor=white)](docs/en/installation.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](docs/en/installation.md#requirements)

**AgentJet (AJet)** is a cutting-edge, user-friendly training framework designed to optimize agents and workflows (built with OpenAI SDK, AgentScope, and even vllm http requests), fine-tuning language model weights behind the scenes.

Simply provide your Agent workflow, training data, and reward function, and we will be ready to enhance your agents to their optimal performance!



## 💡 Minimum Example

Let's begin with the simplest example: a math agent with a tool call.

- First, please check out the [installation guide](docs/en/installation.md) to set up the training environment.
- Then, tune your first model using the minimum example.
  ```python
  ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='verl' --with-ray
  ```


## Features

We aim to build a easy-to-learn Agent tuner that unlock more possibilities for agent developers:

- **Easy and Friendly**. ASTuner helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. ASTuner provides a rich library of [examples](https://github.com/modelscope/AgentJet/tree/main/tutorial) as tutorials.
- **Efficient and Scalable**. ASTuner uses [trinity](https://github.com/modelscope/Trinity-RFT/) as the default backbone (`--backbone=verl`), accelerating your tuning process via fully asynchronous RFT. Nevertheless, if actor colocating is your preference, you can still fall back to the [verl](docs/en/installation.md) backbone. **把verl换成默认BB**
- **Flexible and Fast**. ASTuner supports [multi-agent workflows](docs/en/workflow.md) and adopts a context merging technique, accelerating training by 1.5x to 20x when the workflow involves multi-turn (or multi-agent) conversations.
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
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet
```


2. **Set up Environment**:
```bash
uv venv --python=3.10.16 && source .venv/bin/activate
uv pip install -e .[trinity]
# Note: flash-attn must be installed after other dependencies
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir
```


#### Run Training

You can start training your first agent with a single command using a pre-configured YAML file. Take the [Math agent](docs/en/example_math_agent.md) as an example:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

#### Example Library

Explore our rich library of examples to kickstart your journey:

- 🔢 [**Training a math agent that can write python code**](docs/en/example_math_agent.md).
- 📱 [**Creating an AppWorld agent using AgentScope and training it**](docs/en/example_app_world.md).
- 🐺 [**Developing Werewolves RPG agents and training them**](docs/en/example_werewolves.md).
- 👩🏻‍⚕️ [**Learning to ask questions like a doctor**](docs/en/example_learning_to_ask.md).
- 🎴 [**Writing a countdown game using AgentScope and solving it**](docs/en/example_countdown.md).
- 🚶 [**Solving a frozen lake walking puzzle using ASTuner**](docs/en/example_frozenlake.md).


---

### 🧩 Core Concepts

ASTuner makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>

</div>

#### 1. The User-Centric Interface

To optimize an agent, you provide three core inputs:

* [**Trainable Workflow**](docs/en/workflow.md): Define your agent logic by inheriting the Workflow class, supporting both simple agent setups and advanced multi-agent collaborations.
* [**Task Reader**](docs/en/data_pipeline.md): Load training tasks from JSONL files, HuggingFace datasets, interactive environments, or auto-generate them from documents.
* [**Task Judger**](docs/en/task_judger.md): Evaluates agent outputs and assigns rewards to guide training.

#### 2. Internal System Architecture

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

* **Launcher**: Manages background service processes (Ray, vLLM) and routes the backbone.
* **Task Reader**: Handles data ingestion, augmentation, and filtering.
* **Task Rollout**: Bridges LLM engines and manages the Gym environment lifecycle.
* **Task Runner**: Executes the Agent workflow and calculates rewards.
* **Model Tuner**: Forwards inference requests from the workflow to the LLM engine.
* **Context Tracker**: Monitors LLM calls and automatically merges shared-history timelines to improve training efficiency by **3x to 10x**.


---

### 🚦 Navigation

* 📖 **Tutorials**: From [Installation](docs/en/installation.md) to [Tuning your first agent](docs/en/tutorial.md) — the essential path for beginners.
* 🛠️ **Core Components**: Define your [Trainable Workflow](docs/en/workflow.md) and manage [Data](docs/en/data_pipeline.md) and [Reward](docs/en/tune_your_first_agent.md).
* 💡 **Example**: Check the [Example Library](#example-library) above for real-world cases like [Math](docs/en/example_math_agent.md), [Werewolves game](docs/en/example_werewolves.md) and  [Learning to ask task](docs/en/example_learning_to_ask.md).
* ⚙️ **Deep Dive**: Master advanced [Configuration](docs/en/configuration.md).

## 🗺️ Roadmap

ASTuner is a constantly evolving project. We are planning to add the following features in the near future.

- [ ] Advanced LLM-based multi-agent reinforcement learning.
- [ ] Training dataset generation from few-shot samples.
- [ ] Prompt tuning.
- [ ] Multi-modal training support.
- [ ] Cross-process Tuner wrapper to pass though process forking.
- [ ] Providing training → user feedback → data augmentation → retraining data flywheel example.
- [ ] Optimize configurations for long-context adaptation on smaller GPUs.
- [ ] Add LoRA training examples.
- [ ] Covering LangGraph and AutoGen frameworks.