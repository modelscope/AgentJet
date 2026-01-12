# AgentJet

[![Benchmarking](https://img.shields.io/badge/Benchmarking-0078D4?style=for-the-badge&logo=github)](https://benchmark.agent-matrix.com/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Guide-0A7ECC?style=for-the-badge&logo=readthedocs&logoColor=white)](https://doc.agentjet.top/AgentJet)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://doc.agentjet.top/AgentJet/en/installation#requirements)

<div align="center">
<img width="500" alt="AgentJet" src="docs/agentjet.jpg"/>
</div>


**AgentJet (AJet)** is a cutting-edge, user-friendly training framework designed to optimize agents and workflows (built with OpenAI SDK, AgentScope, Langchain, or just HTTP requests), fine-tuning language model weights behind the scenes.

Simply provide your agent **workflow**, training **dataset**, and **reward** function, and **AgentJet** will be ready to enhance your agents to their optimal performance!



## 🛩️ Minimum Example

Let's begin with the simplest example: a math agent with a tool call.

- First, please check out the [installation guide](https://doc.agentjet.top/AgentJet/en/installation/) to set up the training environment.
- Then, tune your first model using the minimum example.
  ```python
  ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='verl'

  # change to --backbone='trinity' if you want to switch to trinity training engine;
  # or --backbone='debug' if you want to debug with only vLLM
  ```


## 🛩️ Features

We aim to build a easy-to-learn Agent tuner that unlock more possibilities for agent developers:

- **Easy and Friendly**. AgentJet helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. AgentJet provides a rich library of [examples](https://github.com/modelscope/AgentJet/tree/main/tutorial) as tutorials.
- **Efficient and Scalable**. AgentJet uses [verl] as the default backbone (`--backbone=verl`). However, we also support [trinity](https://github.com/modelscope/Trinity-RFT/) as alternative backbone, accelerating your tuning process via fully asynchronous RFT.
- **Flexible and Fast**. AgentJet supports [multi-agent workflows](https://doc.agentjet.top/AgentJet/en/workflow.md) and adopts a context merging technique, accelerating training by 1.5x to 10x when the workflow involves multi-turn (or multi-agent) conversations.
- **Reliability and Reproducibility**. Our team keeps track of framework performance across multiple [tasks + major-git-version + training-backbones](https://benchmark.agent-matrix.com/) (under construction, still gathering data, coming soon).

For advanced researchers, AgentJet also provides high-resolution logging and debugging solutions:
<!-- For advanced researchers, AgentJet provides high-resolution logging and debugging solutions that are, to our knowledge, unprecedented in other prior projects. -->

- **High-Resolution Logging**: AgentJet allows users to save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.
- **Fast Debugging**: AgentJet also provides the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes and enabling breakpoint debugging in IDEs.

---

### 🛩️ Quick Start

#### Installation

- **Click here to read the** [**installation guide**](https://doc.agentjet.top/AgentJet/en/installation/).

#### Run Training

- You can start training your first agent with a single command using a pre-configured YAML file. Take the [Math agent](https://doc.agentjet.top/AgentJet/en/example_math_agent/) as an example:

  ```bash
  ajet --conf tutorial/example_math_agent/math_agent.yaml
  ```

#### Example Library

Explore our rich library of examples to kickstart your journey:

- 🔢 [**Training a math agent that can write python code**](https://doc.agentjet.top/AgentJet/en/example_math_agent).
- 📱 [**Creating an AppWorld agent using AgentScope and training it**](https://doc.agentjet.top/AgentJet/en/example_app_world).
- 🐺 [**Developing Werewolves RPG agents and training them**](https://doc.agentjet.top/AgentJet/en/example_werewolves).
- 👩🏻‍⚕️ [**Learning to ask questions like a doctor**](https://doc.agentjet.top/AgentJet/en/example_learning_to_ask).
- 🎴 [**Writing a countdown game using AgentScope and solving it**](https://doc.agentjet.top/AgentJet/en/example_countdown).
- 🚶 [**Solving a frozen lake walking puzzle using AgentJet**](https://doc.agentjet.top/AgentJet/en/example_frozenlake).


---

### 🛩️ Core Concepts

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>

</div>

#### 1. The User-Centric Interface

To optimize an agent, you provide three core inputs:

* [**Trainable Workflow**](https://doc.agentjet.top/AgentJet/en/workflow): Define your agent logic by inheriting the Workflow class, supporting both simple agent setups and advanced multi-agent collaborations.
* [**Task Reader**](https://doc.agentjet.top/AgentJet/en/data_pipeline): Load training tasks from JSONL files, HuggingFace datasets, interactive environments, or auto-generate them from documents.
* [**Task Judger**](https://doc.agentjet.top/AgentJet/en/task_judger): Evaluates agent outputs and assigns rewards to guide training.

#### 2. Internal System Architecture

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

* **Launcher**: Manages background service processes (Ray, vLLM) and routes the backbone.
* **Task Reader**: Handles data ingestion, augmentation, and filtering.
* **Task Rollout**: Bridges LLM engines and manages the Gym environment lifecycle.
* **Task Runner**: Executes the Agent workflow and calculates rewards.
* **Model Tuner**: Forwards inference requests from the workflow to the LLM engine.
* **Context Tracker**: Monitors LLM calls and automatically merges shared-history timelines to improve training efficiency by **1.5x to 10x**.




### 🛩️ Navigation

* **Tutorials**: From [Installation](https://doc.agentjet.top/AgentJet/en/installation) to [Tuning your first agent](https://doc.agentjet.top/AgentJet/en/tune_your_first_agent) — the essential path for beginners.
* **Core Components**: Define your [Trainable Workflow](https://doc.agentjet.top/AgentJet/en/workflow) and manage [Data](https://doc.agentjet.top/AgentJet/en/data_pipeline) and [Reward](https://doc.agentjet.top/AgentJet/en/task_judger).
* **Example**: Check the [Example Library](#example-library) above for real-world cases like [Math](https://doc.agentjet.top/AgentJet/en/example_math_agent), [Werewolves game](https://doc.agentjet.top/AgentJet/en/example_werewolves) and  [Learning to ask task](https://doc.agentjet.top/AgentJet/en/example_learning_to_ask).
* **Deep Dive**: Master advanced [Configuration](https://doc.agentjet.top/AgentJet/en/configuration).

## 🛩️ Roadmap

AgentJet is a constantly evolving project. We are planning to add the following features in the near future.

| Category | Feature | Status |
| :--- | :--- | :--- |
| **Examples** | Covering LangGraph and AutoGen frameworks | Done & Verifying |
| **Examples** | Add LoRA training examples | Todo |
| **Infra** | Cross-process Tuner wrapper to pass though process forking | Done & Verifying |
| **Infra** | Optimize configurations for long-context adaptation on smaller GPUs | In Progress |
| **Capability** | Prompt tuning | In Progress |
| **Capability** | Multi-modal training support | Todo |
| **Capability** | MARL Credit assignment | Todo |
| **Capability** | Training dataset generation from few-shot samples | Done & Verifying |
