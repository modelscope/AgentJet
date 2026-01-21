# Introduction

**AgentJet (AJet)** is a cutting-edge, user-friendly agent tuning framework designed to optimize LLM models and agent workflows.

Simply provide your workflow (built from AgentScope, OpenAI SDK, Langchain, raw HTTP requests, or hybrid of all of them), training data, and reward function, and we will be ready to enhance your agents to their optimal performance!


## Features

AgentJet aims to build a state-of-the-art agent tuning platform for both developers and researchers

- **Easy and Friendly**. AgentJet helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. AgentJet provides a rich library of [examples](https://github.com/modelscope/AgentJet/tree/main/tutorial) as tutorials.
- **Efficient and Scalable**. AgentJet uses [verl] as the default backbone (`--backbone=verl`). However, we also support [trinity](https://github.com/modelscope/Trinity-RFT/) as alternative backbone, accelerating your tuning process via fully asynchronous RFT.
- **Flexible and Fast**. AgentJet supports [multi-agent workflows](workflow.md) and adopts a context merging technique, accelerating training by 1.5x to 10x when the workflow involves multi-turn (or multi-agent) conversations.
- **Reliability and Reproducibility**. Our team keeps track of framework performance across multiple [tasks + major-git-version + training-backbones](https://benchmark.agentjet.top/) (under construction, still gathering data, coming soon).

For advanced researchers, AgentJet also provides high-resolution logging and debugging solutions:
<!-- For advanced researchers, AgentJet provides high-resolution logging and debugging solutions that are, to our knowledge, unprecedented in other prior projects. -->

- **High-Resolution Logging**: AgentJet allows users to save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.
- **Fast Debugging**: AgentJet also provides the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes and enabling breakpoint debugging in IDEs.




## Quick Start

### Installation

We recommend using `uv` for dependency management.

=== "Step 1: Clone Repository"

    ```bash
    git clone https://github.com/modelscope/AgentJet.git
    cd AgentJet
    ```

=== "Step 2: Setup Environment"

    ```bash
    uv venv --python=3.10.16 && source .venv/bin/activate
    uv pip install -e .[trinity]
    # Note: flash-attn must be installed after other dependencies
    uv pip install flash_attn==2.8.3 --no-build-isolation --no-cache-dir
    ```

- Train the first agent

You can start training your first agent with a single command using a pre-configured YAML file:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml
```

!!! example "Learn More"
    See the [Math Agent](./example_math_agent.md) example for detailed explanation.


## Example Library {#example-library}

Explore our rich library of examples to kickstart your journey:

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="../example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld Agent</h3></div><p class="card-desc">Creating an AppWorld agent using AgentScope and training it for real-world tasks.</p></a>
<a href="../example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>Werewolves Game</h3></div><p class="card-desc">Developing Werewolves RPG agents and training them for strategic gameplay.</p></a>
<a href="../example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor for medical consultation scenarios.</p></a>
<a href="../example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing a countdown game using AgentScope and solving it with RL.</p></a>
<a href="../example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle using AgentJet's reinforcement learning.</p></a>
</div>

---

## Core Concepts

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="AgentJet Architecture" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>
</div>

### 1. The User-Centric Interface

To optimize an agent, you provide three core inputs:

<div class="card-grid">
<a href="./workflow/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Trainable Workflow</h3></div><p class="card-desc">Define your agent logic by inheriting the Workflow class, supporting both simple and multi-agent setups.</p></a>
<a href="./data_pipeline/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Task Reader</h3></div><p class="card-desc">Load training tasks from JSONL files, HuggingFace datasets, or auto-generate from documents.</p></a>
<a href="./task_judger/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Evaluates agent outputs and assigns rewards to guide the training process.</p></a>
</div>

### 2. Internal System Architecture

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

| Module | Description |
|--------|-------------|
| **Launcher** | Manages background service processes (Ray, vLLM) and routes the backbone |
| **Task Reader** | Handles data ingestion, augmentation, and filtering |
| **Task Rollout** | Bridges LLM engines and manages the Gym environment lifecycle |
| **Task Runner** | Executes the AgentScope workflow and calculates rewards |
| **Model Tuner** | Forwards inference requests from the workflow to the LLM engine |
| **Context Tracker** | Monitors LLM calls and automatically merges shared-history timelines (**1.5x-10x** efficiency boost) |

---

## Next Steps

<div class="card-grid">
<a href="./installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>Installation</h3></div><p class="card-desc">Set up AgentJet environment and dependencies.</p></a>
<a href="./quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training in minutes.</p></a>
</div>
