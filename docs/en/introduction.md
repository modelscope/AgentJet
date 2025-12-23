# AgentScope Tuner

AgentScope Tuner, or **ASTuner**, is an advanced agent training framework designed to optimize AgentScope workflows and agents.
Simply provide your AgentScope workflow, training data, and reward function to enhance your agents to their optimal performance.



## ✨ Features

- **Data Augmentation & User Feedback Tracing**: Automatically augment training data and trace user feedback when training data is limited.
- **Auto Rubrics**:  Generate LLM-as-judge reward functions by learning from few-shot examples.
- **Multi-Agent Support**: Build advanced cooperative multi-agent systems with ease.
- **Highly Efficient Async Training-Inference Separation**: Powered by Trinity-RFT for optimized performance.
- **Training-Debugging Integration**: Seamlessly toggle between training and debugging modes using a simple `--backbone` switch (`--backbone=trinity` or `--backbone=debug`).
- **Comprehensive Logging**: Integrate message-level logging from AgentScope Studio and token-level logging for detailed insights.



## 🚀 Quick Start

### Installation

We recommend using `uv` for dependency management, though `conda` is also supported.

1. Clone the repository:

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

3. Create virtual environment and install dependencies:

```bash
uv venv --python=3.10.16  # Create virtual environment
source .venv/bin/activate  # Activate virtual environment

uv pip install -e .[trinity]
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir
```


### Get Started

Run and train an agent in one command:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

Explore our rich library of examples to kickstart your journey:

- 🚀 [**Quick Start**](./quickstart.md): Learn the framework and train your first agent from scratch.
- 🍳 [**Build a Simple Math Agent**](./example_math_agent.md): Specialized in GSM8K problems and learn how to train it.
- 🍳 [**Build an AppWorld Agent**](./example_app_world.md): Create an AppWorld agent using AgentScope and train it.
- 🍳 [**Build Multi-Agent Werewolf Gameplay**](./example_werewolves.md): Develop multiple Werewolves RPG agents and train them.
- 🫣 [**Build a "Learning to Ask" Agent**](./example_learning_to_ask.md): Train a proactive agent to ask high-value questions using LLM-as-a-judge rewards.
- 📔 [**Tracing-Feedback Training**](./example_tracing_feedback_loop.md): Learn how to train using user feedback tracing.

To learn the details of each component, please refer to:

- ⚙️ [**Configuration**](./configuration.md): Configure the data, optimization algorithms, rewards, etc.
- 💼 [**Workflow**](./workflow.md): Build your own agent with trainable workflow.
- 📊 [**Data Pipeline & Generation**](./data_pipeline.md): Includes building dataset tasks from document materials and expanding datasets from few samples.


## 🏗️ Project Architecture

AgentScope Tuner makes agent fine-tuning unprecedentedly straightforward. It encapsulates complex fine-tuning training into a simple module driven by three core inputs:

- AgentScope workflows (which can directly utilize your pre-written AgentScope workflows)
- Task datasets (providing training data)
- Reward Judge (assessing performance quality).


![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764207569988-69b6926f-301b-4766-9199-3823974aab99.png)

Of course, fine-tuning the workflow would not be possible without the silent support of the following core modules:

![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764705947150-753d77f0-a1a7-4491-8b8b-a0f9f998ed0a.png)
- **launcher**: The entry point of the project, helping developers quickly switch between debugging the backbone and training the backbone. It also launches and intelligently monitors the environment service processes related to training in the background.
- **task rollout**: Bridges different LLM engines (such as FSDP, VLLM, etc.), implements a retry mechanism, and passes the tasks read by the task reader. If the gym environment is required, it initializes the gym environment and ensures resource cleanup.
- **task runner**: The front-line worker responsible for actually executing the user-provided AgentScope workflow. It also runs the judge and performs preliminary reward calculations.
- **model tuner**: When the AgentScope workflow sends an LLM inference request, this busy component directly receives and forwards the request to the LLM engine.
- **context tracker**: A loyal recorder that monitors every LLM call and automatically identifies and archives LLM requests belonging to the same Agent and the same timeline. At the end of the task, it marks the loss mask, merges the recorded LLM input-output timelines, and improves training efficiency by 3 to 10 times.



## 🗺️ Project Roadmap

Working in progress:

- Enhance data generation module functionality
- Provide a training → user feedback → data augmentation → retraining data flywheel example
- Offer refined post-processing options for multi-agent samples
- Support training with multiple models
- Optimize configurations for long-context adaptation on smaller GPUs
- Add LoRA training examples
