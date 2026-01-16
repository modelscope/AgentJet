# AgentJet


<div align="center">
<img width="500" alt="AgentJet" src="agentjet.jpg"/>
</div>

**AgentJet (AJet)** is a cutting-edge, user-friendly agent tuning framework designed to optimize LLM models and agent workflows.


## ✈️ Key Features

<div class="card-grid">
    <a href="en/tune_your_first_agent/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:rocket.svg"
                class="card-icon card-icon-agent" alt="">
            <h3>Get Started with Ease</h3>
        </div>
        <p class="card-desc">
            AgentJet simplifies the process of tuning the models that power your agent workflows. It supports nearly all major agent frameworks (e.g. <b>agentscope</b>, <b>langchain</b>), as well as <b>framwork-less</b> agents built from HTTP requests.
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
    <a href="https://benchmark.agent-matrix.com/" class="feature-card">
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
    <a href="en/workflow/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:users.svg" class="card-icon card-icon-tool"
                alt="">
            <h3>Multi-agent and Multi-turn</h3>
        </div>
        <p class="card-desc">
            Built to support advanced <b>multi-agent</b> and <b>multi-turn</b> LLM workflows,
            AgentJet intergrates timeline-merging algorithms that
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
            Log <b>token-level</b> rollout details, capturing token IDs, token <b>loss masks</b>, and token <b>log probabilities</b> with <b>web UI display</b>. This Support workflow development, agent diagnostics, and facilitate research on advanced LLM algorithm studies.
        </p>
    </a>
    <a href="en/installation/" class="feature-card">
        <div class="card-header"><img src="https://api.iconify.design/lucide:cpu.svg" class="card-icon card-icon-tool"
                alt="">
            <h3>Any Training Engine</h3>
        </div>
        <p class="card-desc">
            Support <b>multiple training engines</b> as backbone (<b>VeRL</b> and <b>Trinity-RFT</b>). Tinker backbone support will be released soon.
            Choose from <b>vLLM</b> and <b>SGLang</b> as you wish. Say goodbye to training engine gaps.
        </p>
    </a>
</div>



## ✈️ Quick Start

<div class="card-grid">
<a href="en/installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>Click Here for the Full Installation Document</h3></div></a>
</div>


We recommend using `uv` for dependency management. [Click here](en/installation.md) for details and other training backbone (e.g. Trinity-RFT) options.

- Clone the Repository:
    ```bash
    git clone https://github.com/modelscope/AgentJet.git
    cd AgentJet
    ```

- Set up Environment:
    ```bash
    uv venv --python=3.10.16 && source .venv/bin/activate
    uv pip install -e .[verl]

    # Note: flash-attn must be installed after other dependencies
    uv pip install flash_attn==2.8.3 --no-build-isolation --no-cache-dir
    ```

- Train the First Agent:
    ```bash
    # You can start training your first agent with a single command using a pre-configured YAML file

    ajet --conf tutorial/example_math_agent/math_agent.yaml
    ```




## ✈️ Example Library {#example-library}

Explore our rich library of examples to kickstart your journey:

<div class="card-grid">
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="en/example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld Agent</h3></div><p class="card-desc">Creating an AppWorld agent using AgentScope and training it for real-world tasks.</p></a>
<a href="en/example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>Werewolves Game</h3></div><p class="card-desc">Developing Werewolves RPG agents and training them for strategic gameplay.</p></a>
<a href="en/example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor for medical consultation scenarios.</p></a>
<a href="en/example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing a countdown game using AgentScope and solving it with RL.</p></a>
<a href="en/example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle using AgentJet's reinforcement learning.</p></a>
</div>


## ✈️ Core Concepts

AgentJet makes agent fine-tuning straightforward by separating the developer interface from the internal execution logic.

**✈️ The User-Centric Interface**

To optimize an agent, you provide three core inputs:

<div class="card-grid">
<a href="en/workflow/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>Trainable Workflow</h3></div><p class="card-desc">Define your agent logic by inheriting the Workflow class, supporting both simple and multi-agent setups.</p></a>
<a href="en/data_pipeline/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Task Reader</h3></div><p class="card-desc">Load training tasks from JSONL files, HuggingFace datasets, or auto-generate from documents.</p></a>
<a href="en/task_judger/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Evaluates agent outputs and assigns rewards to guide the training process.</p></a>
</div>

<div align="center">
<img width="840" alt="AgentJet Architecture" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>
</div>

**✈️ Internal System Architecture**

The internal system orchestrates several specialized modules to handle the complexities of RL training and agent interactions.

| Module | Description |
|--------|-------------|
| **Launcher** | Manages background service processes (Ray, vLLM) and routes the backbone |
| **Task Rollout** | Bridges LLM engines and manages the Gym environment lifecycle |
| **Task Runner** | Executes the AgentScope workflow and calculates rewards |
| **Model Tuner** | Forwards inference requests from the workflow to the LLM engine |
| **Context Tracker** | Monitors LLM calls and automatically merges shared-history timelines (1.5x-10x efficiency boost) |




## ✈️ Next Steps

<div class="card-grid">
<a href="en/installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>Installation</h3></div><p class="card-desc">Set up AgentJet environment and dependencies.</p></a>
<a href="en/quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training in minutes.</p></a>
<a href="en/tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-general" alt=""><h3>First Agent</h3></div><p class="card-desc">Build and train your own agent from scratch.</p></a>
<a href="en/example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Examples</h3></div><p class="card-desc">Explore detailed training examples.</p></a>
</div>


<!-- ## 中文文档

<div class="card-grid">
<a href="zh/intro/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:translate.svg" class="card-icon card-icon-multimodal" alt=""><h3>查看中文文档</h3></div><p class="card-desc">完整的中文教程和指南。</p></a></div> -->
