# AgentJet

**AgentJet (AgentJet)** is a cutting-edge, user-friendly training framework designed to optimize AgentScope agents and workflows, fine-tuning language model weights behind the scenes.

Simply provide your AgentScope workflow, training data, and reward function, and we will be ready to enhance your agents to their optimal performance!

---

## Key Features

<div class="card-grid">
<a href="en/configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:rocket.svg" class="card-icon card-icon-agent" alt=""><h3>Easy and Friendly</h3></div><p class="card-desc">AgentJet helps you tune models behind your agent workflows easily. Zero-config defaults and intuitive YAML-based configuration.</p></a>
<a href="#example-library" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:book-open.svg" class="card-icon card-icon-general" alt=""><h3>Rich Tutorial Library</h3></div><p class="card-desc">Rich library of examples as tutorials: Math Agent, Werewolves Game, AppWorld and more with step-by-step guides.</p></a>
<a href="en/workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:zap.svg" class="card-icon card-icon-tool" alt=""><h3>Efficient and Scalable</h3></div><p class="card-desc">Uses Trinity as the default backbone with fully asynchronous RFT. Context merging technique for 1.5x to 20x training acceleration.</p></a>
</div>

---

## Quick Start

### Installation

We recommend using `uv` for dependency management.

**1. Clone the Repository:**

```bash
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet
```

**2. Set up Environment:**

```bash
uv venv --python=3.10.16 && source .venv/bin/activate
uv pip install -e .[trinity]
# Note: flash-attn must be installed after other dependencies
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir
```

### Run Training

You can start training your first agent with a single command using a pre-configured YAML file:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

---

## Example Library {#example-library}

Explore our rich library of examples to kickstart your journey:

<div class="card-grid">
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="en/example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld Agent</h3></div><p class="card-desc">Creating an AppWorld agent using AgentScope and training it for real-world tasks.</p></a>
<a href="en/example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>Werewolves Game</h3></div><p class="card-desc">Developing Werewolves RPG agents and training them for strategic gameplay.</p></a>
<a href="en/example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor for medical consultation scenarios.</p></a>
<a href="en/example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing a countdown game using AgentScope and solving it with RL.</p></a>
<a href="en/example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle using AgentJet's reinforcement learning.</p></a>
</div>

---

## Core Concepts

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

<div align="center">
<img width="480" alt="AgentJet Architecture" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>
</div>

### The User-Centric Interface

To optimize an agent, you provide three core inputs:

<div class="card-grid">
<a href="en/workflow/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Trainable Workflow</h3></div><p class="card-desc">Define your agent logic by inheriting the Workflow class, supporting both simple and multi-agent setups.</p></a>
<a href="en/data_pipeline/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Task Reader</h3></div><p class="card-desc">Load training tasks from JSONL files, HuggingFace datasets, or auto-generate from documents.</p></a>
<a href="en/task_judger/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Evaluates agent outputs and assigns rewards to guide the training process.</p></a>
</div>

### Internal System Architecture

The internal system orchestrates several specialized modules:

| Module | Description |
|--------|-------------|
| **Launcher** | Manages background service processes (Ray, vLLM) and routes the backbone |
| **Task Rollout** | Bridges LLM engines and manages the Gym environment lifecycle |
| **Task Runner** | Executes the AgentScope workflow and calculates rewards |
| **Model Tuner** | Forwards inference requests from the workflow to the LLM engine |
| **Context Tracker** | Monitors LLM calls and automatically merges shared-history timelines (3x-10x efficiency boost) |

---

## For Advanced Researchers

AgentJet provides high-resolution logging and debugging solutions:

!!! tip "High-Resolution Logging"
    Save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.

!!! info "Fast Debugging"
    Use the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes.

---

## Next Steps

<div class="card-grid">
<a href="en/installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>Installation</h3></div><p class="card-desc">Set up AgentJet environment and dependencies.</p></a>
<a href="en/quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training in minutes.</p></a>
<a href="en/tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-general" alt=""><h3>First Agent</h3></div><p class="card-desc">Build and train your own agent from scratch.</p></a>
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Examples</h3></div><p class="card-desc">Explore detailed training examples.</p></a>
</div>

---

## 中文文档

<div class="card-grid">
<a href="zh/intro/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:translate.svg" class="card-icon card-icon-multimodal" alt=""><h3>查看中文文档</h3></div><p class="card-desc">完整的中文教程和指南。</p></a></div>

