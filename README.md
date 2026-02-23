# AgentJet (Beta)

[![Benchmarking](https://img.shields.io/badge/Benchmarking-0078D4?style=for-the-badge&logo=github)](https://benchmark.agentjet.top/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Documents-0A7ECC?style=for-the-badge&logo=readthedocs&logoColor=white)](https://modelscope.github.io/AgentJet)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://modelscope.github.io/AgentJet/en/installation#requirements)

<div align="center">
  <a href="https://modelscope.github.io/AgentJet" target="_blank">
    <img width="500" alt="AgentJet" src="docs/agentjet.jpg"/>
  </a>
</div>


**AgentJet (AJet)** is a cutting-edge, user-friendly agent RL training framework designed to optimize agents and agentic workflows (supporting any agent built with OpenAI SDK, AgentScope, Langchain, or raw HTTP requests), fine-tuning LLM weights to enhance model performance.

**AgentJet (AJet)** has fully-distributed **swarm training** capability, which means that you can **deploy `ajet-swarm start` in GPU server(s) and then start training agents in your laptop(s)**! Simply provide your agent workflow, training dataset, and reward function, and AgentJet will be ready to go!



## ✈️ Fast Introduction

### Classic Mode

Let's begin with the simplest example: a math agent with a tool call. This is a simple & centralized training example.

1. please check out the [installation guide](https://modelscope.github.io/AgentJet/en/installation/) to set up the training environment.
2. tune your first model using the minimum example.
    ```python
    ajet --conf ./tutorial/example_math_agent/math_agent.yaml --backbone='verl'
    ```
<div align="center">
<img width="640" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/classic+swarm+revise.jpg"/>
</div>

### Swarm Mode

Let's begin with the simplest AgentJet Swarm example: also a math agent. In this case, you can use any GPU-less laptop to train the model remotely.

1. Start swarm server and begin swarm overwatch: `ajet-swarm start` and `ajet-swarm overwatch --swarm-url=http://localhost:10086`.
2. From your laptop (or swarm server localhost), run [this simple script](https://github.com/modelscope/AgentJet/blob/main/tutorial/example_math_swarm/math.py) to begin training:
    ```python
    AJET_SWARM_URL="http://swarm-server-ip:10086" python ./tutorial/example_math_swarm/math.py
    ```
<div align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/41ed1e71-8b18-4c4c-b5e2-833399317337"/>
</div>


## ✈️ Features

We aim to build a easy-to-learn Agent tuner that unlock more possibilities for agent developers:

- **Easy and Friendly**. AgentJet helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. AgentJet provides a rich library of [examples](https://github.com/modelscope/AgentJet/tree/main/tutorial) as tutorials.
- **Swarm Training**. [This unique feature](https://modelscope.github.io/AgentJet/en/swarm_intro_blog_english/) of AgentJet opens many possibilities: deploying distributed & self-healing rollout workers, **non-shared-parameter multi-agent** training, **multi-runtime & multi-task cocktail** training. And just like Tinker, you can use AgentJet Swarm to train **models even on **GPU-less laptop(s)**.
- **Efficient and Scalable**. AgentJet uses [verl] as the default backbone (`--backbone=verl`). However, we also support trinity as alternative backbone, accelerating your tuning process via fully asynchronous RFT.
- **Flexible and Fast**. AgentJet supports [multi-agent workflows](https://modelscope.github.io/AgentJet/en/workflow/) and adopts a context merging technique, accelerating training by 1.5x to 10x when the workflow involves multi-turn (or multi-agent) conversations.
- **Reliability and Reproducibility**. Our team keeps track of framework performance across multiple [tasks + major-git-version + training-backbones](https://benchmark.agentjet.top/) (under construction, still gathering data, coming soon).

For advanced researchers, AgentJet also provides high-resolution logging and debugging solutions:
<!-- For advanced researchers, AgentJet provides high-resolution logging and debugging solutions that are, to our knowledge, unprecedented in other prior projects. -->

- **High-Resolution Logging**: AgentJet allows users to save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.
- **Fast Debugging**: AgentJet also provides the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes and enabling breakpoint debugging in IDEs.

<div align="center">
<img width="600" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/ai-generated-1771873242388.jpg"/>
</div>


---

### ✈️ Quick Start

#### Installation

- **Click here to read the** [**installation guide**](https://modelscope.github.io/AgentJet/en/installation/).


#### Example Library

Explore our rich library of examples to kickstart your journey:

- 🔢 [**Training a math agent that can write python code**](https://modelscope.github.io/AgentJet/en/example_math_agent).
- 📱 [**Creating an AppWorld agent using AgentScope and training it**](https://modelscope.github.io/AgentJet/en/example_app_world).
- 🐺 [**Developing Werewolves RPG agents and training them**](https://modelscope.github.io/AgentJet/en/example_werewolves).
- 👩🏻‍⚕️ [**Learning to ask questions like a doctor**](https://modelscope.github.io/AgentJet/en/example_learning_to_ask).
- 🎴 [**Writing a countdown game using AgentScope and solving it**](https://modelscope.github.io/AgentJet/en/example_countdown).
- 🚶 [**Solving a frozen lake walking puzzle using AgentJet**](https://modelscope.github.io/AgentJet/en/example_frozenlake).

Explore our automated benchmarking system [https://benchmark.agentjet.top/](https://benchmark.agentjet.top/):
<div align="center">
<img width="600" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/benchmark.gif"/>
</div>


---

### ✈️ Core Concepts

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>

</div>

#### 1. The User-Centric Interface

To optimize an agent, you provide three core inputs:

* [**Trainable Workflow**](https://modelscope.github.io/AgentJet/en/workflow): Define your agent logic by inheriting the Workflow class, supporting both simple agent setups and advanced multi-agent collaborations.
* [**Task Reader**](https://modelscope.github.io/AgentJet/en/data_pipeline): Load training tasks from JSONL files, HuggingFace datasets, interactive environments, or auto-generate them from documents.
* [**Task Judger**](https://modelscope.github.io/AgentJet/en/task_judger): Evaluates agent outputs and assigns rewards to guide training.

#### 2. Internal System Architecture

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

* **Launcher**: Manages background service processes (Ray, vLLM) and routes the backbone.
* **Task Reader**: Handles data ingestion, augmentation, and filtering.
* **Task Rollout**: Bridges LLM engines and manages the Gym environment lifecycle.
* **Task Runner**: Executes the Agent workflow and calculates rewards.
* **Model Tuner**: Forwards inference requests from the workflow to the LLM engine.
* **Context Tracker**: Monitors LLM calls and automatically merges shared-history timelines to improve training efficiency by **1.5x to 10x**.
* **Swarm Server**: A data interchange center that accept OpenAI-like requests and engine instructions, activated only in AgentJet Swarm mode.




### ✈️ Navigation

* **Tutorials**: From [Installation](https://modelscope.github.io/AgentJet/en/installation) to [Tuning your first agent](https://modelscope.github.io/AgentJet/en/tune_your_first_agent) — the essential path for beginners.
* **Core Components**: Define your [Trainable Workflow](https://modelscope.github.io/AgentJet/en/workflow) and manage [Data](https://modelscope.github.io/AgentJet/en/data_pipeline) and [Reward](https://modelscope.github.io/AgentJet/en/task_judger).
* **Example**: Check the [Example Library](https://modelscope.github.io/AgentJet/#example-library) above for real-world cases like [Math](https://modelscope.github.io/AgentJet/en/example_math_agent), [Werewolves game](https://modelscope.github.io/AgentJet/en/example_werewolves) and  [Learning to ask task](https://modelscope.github.io/AgentJet/en/example_learning_to_ask).
* **Deep Dive**: Master advanced [Configuration](https://modelscope.github.io/AgentJet/en/configuration).

## ✈️ Roadmap

AgentJet is a constantly evolving project. We are planning to add the following features in the near future.

| Category | Feature | Status |
| :--- | :--- | :--- |
| **Examples** | Add LoRA training examples | Todo |
| **Infra** | Optimize configurations for long-context adaptation on smaller GPUs | In Progress |
| **Capability** | Multi-modal training support | Todo |
| **Capability** | MARL Credit assignment | Todo |
| **Capability** | Training dataset generation from few-shot samples | Todo |


## ✈️ Citation

If you use AgentJet in your research, please cite:

```bibtex
@software{
  title  = {AgentJet: A Cutting-Edge Multi-Agent Training Platform for Large Language Models.},
  author = {The AgentJet Team},
  url    = {https://modelscope.github.io/AgentJet/},
  month  = {01},
  year   = {2026}
}
```



<br/>

---
<div align="center">
This project is under active development, we need your help to make it shine! <br/>

[⭐ Star Us](https://github.com/modelscope/AgentJet) · [✈️ Report Bug](https://github.com/modelscope/AgentJet/issues) · [✈️ Request Feature](https://github.com/modelscope/AgentJet/issues)
</div>



<div align="center">
<img width="180" alt="image" src="https://img.alicdn.com/imgextra/i4/O1CN01DJuOtZ1Kgu1UvjaNl_!!6000000001194-2-tps-922-882.png"/>
<br/>
<span>Join AgentJet DingTalk Group to share your idea</span>
</div>



