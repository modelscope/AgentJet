# AgentJet

[![Benchmarking](https://img.shields.io/badge/Benchmarking-0078D4?style=for-the-badge&logo=github)](https://benchmark.agentjet.top/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Documents-0A7ECC?style=for-the-badge&logo=readthedocs&logoColor=white)](https://modelscope.github.io/AgentJet)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://modelscope.github.io/AgentJet/en/installation#requirements)

<div align="center">
<img width="500" alt="AgentJet" src="agentjet.jpg"/>
</div>

**AgentJet (AJet)** is a cutting-edge, user-friendly agent RL training framework designed to optimize agents and agentic workflows (supporting any agent built with OpenAI SDK, AgentScope, Langchain, or raw HTTP requests), fine-tuning LLM weights to enhance model performance.

**AgentJet (AJet)** has fully-distributed **swarm training** capability, which means that you can **deploy `ajet-swarm start` in GPU server(s) and then start training agents in your laptop(s)**! Simply provide your agent workflow, training dataset, and reward function, and AgentJet will be ready to go!


## Fast Introduction

### Classic Mode

Let's begin with the simplest example: a math agent with a tool call. This is a simple & centralized training example.

1. please check out the [installation guide](en/installation/) to set up the training environment.
2. tune your first model using the minimum example.
    ```python
    ajet --conf ./tutorial/example_math_agent/math_agent.yaml --backbone='verl'
    ```
<div align="center">
<img width="840" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/classic+swarm+revise.jpg"/>
</div>

### Swarm Mode

Let's begin with the simplest AgentJet Swarm example: also a math agent. In this case, you can use any GPU-less laptop to train the model remotely.

1. Start swarm server and begin swarm overwatch: `ajet-swarm start` and `ajet-swarm overwatch`. (Alternative: if you are a fan of docker, use our [prebuilt docker image here](en/ajet-swarm-docker.md) without setting up dependencies)
2. From your laptop (or swarm server localhost), run [this simple script](https://github.com/modelscope/AgentJet/blob/main/tutorial/example_math_swarm/math.py) to begin training:
    ```python
    AJET_SWARM_URL="http://swarm-server-ip:10086" python ./tutorial/example_math_swarm/math.py
    ```
<div align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/41ed1e71-8b18-4c4c-b5e2-833399317337"/>
</div>


## Key Features

<div class="card-grid">
    <a href="en/swarm_best_practice/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/streamline-cyber:network.svg"
                class="card-icon card-icon-agent" alt="">
            <h3>Swarm Training Mode</h3>
        </div>
        <p class="card-desc">
            Swarm Training in AgentJet opens many possibilities: deploying distributed & self-healing rollout workers, <b>non-shared-parameter multi-agent</b> training, <b>multi-runtime & multi-task cocktail</b> training. And just like Tinker, you can use AgentJet Swarm to train <b>models even on </b>GPU-less laptop(s)</b>.
        </p>
    </a>
    <a href="en/tune_your_first_agent/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:rocket.svg"
                class="card-icon card-icon-agent" alt="">
            <h3>Get Started with Ease</h3>
        </div>
        <p class="card-desc">
            AgentJet simplifies the process of tuning the models that power your agent workflows. It supports nearly all major agent frameworks (e.g. <b>agentscope</b>, <b>langchain</b>), as well as <b>framework-less</b> agents built from HTTP requests.
        </p>
    </a>
    <a href="#example-library" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:library.svg"
                class="card-icon card-icon-general" alt="">
            <h3>Rich Tutorial Library</h3>
        </div>
        <p class="card-desc">
            Rich examples as beginner's tutorial: <b>math agent</b>, <b>werewolves rpg</b>, <b>appworld</b> ... All with step-by-step
            guides. Covering various agentic frameworks.</p>
    </a>
    <a href="https://benchmark.agentjet.top/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:shield-check.svg" class="card-icon card-icon-tool"
                alt="">
            <h3>Reliable and Reproducible</h3>
        </div>
        <p class="card-desc">
        Checkout AgentJet's community-powered, robot-assisted <b>open-benchmarking system</b>.
        Share progress, compare training backbones, discover bugs and iterate faster than ever!
        Click here to see AgentJet performance across tasks/versions/backbones.
        </p>
    </a>
    <a href="en/classic_workflow/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:users.svg" class="card-icon card-icon-tool"
                alt="">
            <h3>Multi-agent and Multi-turn</h3>
        </div>
        <p class="card-desc">
            Built to support advanced <b>multi-agent</b> and <b>multi-turn</b> LLM workflows,
            AgentJet integrates timeline-merging algorithms that
            automatically analyze and consolidate each agent's LLM timeline,
            <b>accelerating</b> training speed 1.5x ~ 10x.
        </p>
    </a>
    <a href="en/beast_logger/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:microscope.svg" class="card-icon card-icon-tool"
                alt="">
            <h3>High Resolution Logging</h3>
        </div>
        <p class="card-desc">
            Log <b>token-level</b> rollout details, capturing token IDs, token <b>loss masks</b>, and token <b>log probabilities</b> with <b>web UI display</b>. This supports workflow development, agent diagnostics, and facilitates research on advanced LLM algorithm studies.
        </p>
    </a>
</div>

<div align="center">
<img width="999" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/ai-generated-1771873242388.jpg"/>
</div>




## Example Library {#example-library}

Explore our rich library of examples to kickstart your journey:

<div align="center">
<img width="600" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/benchmark.gif"/>
</div>

<div class="card-grid">
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="en/example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld Agent</h3></div><p class="card-desc">Creating an AppWorld agent using AgentScope and training it for real-world tasks.</p></a>
<a href="en/example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>Werewolves Game</h3></div><p class="card-desc">Developing Werewolves RPG agents and training them for strategic gameplay.</p></a>
<a href="en/example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor for medical consultation scenarios.</p></a>
<a href="en/example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing a countdown game using AgentScope and solving it with RL.</p></a>
<a href="en/example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle using AgentJet's reinforcement learning.</p></a>
</div>


## Project Structure

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

**Basic Modules**

To optimize an agent, you provide three core inputs:

<div class="card-grid">
<a href="en/classic_workflow/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Trainable Workflow</h3></div><p class="card-desc">Define your agent logic by inheriting the Workflow class, supporting both simple and multi-agent setups.</p></a>
<a href="en/data_pipeline/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Task Reader</h3></div><p class="card-desc">Load training tasks from JSONL files, HuggingFace datasets, or auto-generate from documents.</p></a>
<a href="en/task_judger/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Evaluates agent outputs and assigns rewards to guide the training process.</p></a>
</div>

<div align="center">
<img width="840" alt="AgentJet Architecture" src="https://img.alicdn.com/imgextra/i2/O1CN01PdCJym1jqr1jWGMZ4_!!6000000004600-0-tps-2013-870.jpg"/>
</div>

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

| Module | Description |
|--------|-------------|
| **Launcher** | Manages background service processes (Ray, vLLM) and routes the backbone |
| **Task Rollout** | Bridges LLM engines and manages the Gym environment lifecycle |
| **Task Runner** | Executes the agent workflow and calculates rewards |
| **Model Tuner** | Forwards inference requests from the workflow to the LLM engine |
| **Context Tracker** | Monitors LLM calls and automatically merges shared-history timelines (1.5x-10x efficiency boost) |
| **Swarm Server** | A data interchange center that accepts OpenAI-like requests and engine instructions, activated only in AgentJet Swarm mode |

**Swarm Architecture**

When swarm training mode is enabled, an additional component will be activated:

* **Swarm Data Interchange Server**: Maintains HTTP service, listens to swarm instructions and OpenAI compatible requests. Establishes a high-speed zmq communication channel to coordinate other modules.

<div align="center">
<img width="600" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/arch.jpg"/>
</div>


## Navigation

* **Tutorials**: From [Installation](en/installation) to [Tuning your first agent](en/tune_your_first_agent) — the essential path for beginners.
* **Core Components**: Define your [Classic Workflow](en/classic_workflow) or [Swarm Workflow](en/swarm_workflow), and manage [Data](en/data_pipeline) and [Reward](en/task_judger).
* **Example**: Check the [Example Library](#example-library) above for real-world cases like [Math](en/example_math_agent), [Werewolves game](en/example_werewolves) and [Learning to ask task](en/example_learning_to_ask).
* **Deep Dive**: Master advanced [Configuration](en/configuration).


## Roadmap

AgentJet is a constantly evolving project. We are planning to add the following features in the near future.

| Category | Feature | Status |
| :--- | :--- | :--- |
| **Examples** | Add LoRA training examples | Todo |
| **Infra** | Optimize configurations for long-context adaptation on smaller GPUs | In Progress |
| **Capability** | Multi-modal training support | Todo |
| **Capability** | MARL Credit assignment | Todo |
| **Capability** | Training dataset generation from few-shot samples | Todo |


## Citation

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


## Next Steps

<div class="card-grid">
<a href="en/installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>Installation</h3></div><p class="card-desc">Set up AgentJet environment and dependencies.</p></a>
<a href="en/quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training in minutes.</p></a>
<a href="en/tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-general" alt=""><h3>First Agent</h3></div><p class="card-desc">Build and train your own agent from scratch.</p></a>
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Examples</h3></div><p class="card-desc">Explore detailed training examples.</p></a>
</div>


---

<div align="center">
This project is under active development, we need your help to make it shine! <br/>

[⭐ Star Us](https://github.com/modelscope/AgentJet) · [Report Bug](https://github.com/modelscope/AgentJet/issues) · [Request Feature](https://github.com/modelscope/AgentJet/issues)
</div>



<div align="center">
<img width="180" alt="image" src="https://img.alicdn.com/imgextra/i4/O1CN01DJuOtZ1Kgu1UvjaNl_!!6000000001194-2-tps-922-882.png"/>
<br/>
<span>Join AgentJet DingTalk Group to share your idea</span>
</div>


<!-- ## 中文文档

<div class="card-grid">
<a href="zh/intro/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:translate.svg" class="card-icon card-icon-multimodal" alt=""><h3>查看中文文档</h3></div><p class="card-desc">完整的中文教程和指南。</p></a></div> -->
