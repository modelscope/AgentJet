# AgentScope Tuner Documentation

[![Benchmarking](https://img.shields.io/badge/Benchmarking-0078D4?style=for-the-badge&logo=github)](https://benchmark.agent-matrix.com/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Guide-0A7ECC?style=for-the-badge&logo=readthedocs&logoColor=white)](docs/en/installation.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](docs/en/installation.md#requirements)

**AgentScope Tuner (ASTuner)** is a cutting-edge, user-friendly training framework designed to optimize AgentScope agents and workflows, fine-tuning language model weights behind the scenes.

Simply provide your AgentScope workflow, training data, and reward function, and we will be ready to enhance your agents to their optimal performance!



## 💡 Minimum Example

Let's begin with the simplest example: a math agent with a tool call.

- First, please check out the [installation guide](docs/en/installation.md) to set up the training environment.
- Then, tune your first model using the minimum example below (suppose you have written an Agent called `MathToolWorkflow`).
  ```python
  from astuner import AstunerJob
  from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow
  model_path = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct"
  job = AstunerJob(n_gpu=8, algorithm='grpo', model=model_path)
  job.set_workflow(MathToolWorkflow)
  job.set_data(type="hf", dataset_path='openai/gsm8k')
  # [Optional: Save yaml file for manual adjustment]  job.dump_job_as_yaml('saved_experiments/math.yaml')
  # [Optional: Load yaml file from manual adjustment] job.load_job_from_yaml('saved_experiments/math.yaml')
  tuned_model = job.tune()  # Equivalent to `astuner --conf ./saved_experiments/math.yaml` in the terminal
  ```


## 🚀 Features

We aim to build a easy-to-learn AgentScope tuner that unlock more possibilities for agent developers:

- **Easy and Friendly**. ASTuner helps you tune models behind your agent workflows easily, optimizing your agents for top performance with minimal effort.
- **Rich Tutorial Library**. ASTuner provides a rich library of [examples](https://github.com/agentscope-ai/agentscope-tuner/tree/main/tutorial) as tutorials.
- **Efficient and Scalable**. ASTuner uses [trinity](https://github.com/modelscope/Trinity-RFT/) as the default backbone (`--backbone=trinity`), accelerating your tuning process via fully asynchronous RFT. Nevertheless, if actor colocating is your preference, you can still fall back to the [verl](docs/en/installation.md) backbone.
- **Flexible and Fast**. ASTuner supports [multi-agent workflows](docs/en/workflow.md) and adopts a timeline merging technique, accelerating training by 1.5x to 20x when the workflow involves multi-turn (or multi-agent) conversations.
- **Reliability and Reproducibility**. Our team keeps track of framework performance across multiple [tasks + major-git-version + training-backbones](https://benchmark.agent-matrix.com/) (under construction, still gathering data, comming soon).

For advanced researchers, ASTuner also provides high-resolution logging and debugging solutions:
<!-- For advanced researchers, ASTuner provides high-resolution logging and debugging solutions that are, to our knowledge, unprecedented in other prior projects. -->

- **High-Resolution Logging**: ASTuner allows users to save and inspect token-level rollout details, recording token IDs, token loss masks, and even token logprobs to facilitate workflow development and agent diagnostics.
- **Fast Debugging**: ASTuner also provides the `--backbone=debug` option for the best debugging experience, shortening your wait period from minutes to seconds after code changes and enabling breakpoint debugging in IDEs.


## 🪐Getting Started

### Getting Started: Installation

There are many options to install ASTuner. Please refer to [`installation.md`](docs/en/installation.md) for detailed guidelines. Here we demonstrate how to install ASTuner using uv.

1. Clone the Repository.
Clone the AgentScope Tuner repository from GitHub and navigate into the project directory:
    ```bash
    git clone https://github.com/agentscope-ai/agentscope-tuner.git
    cd agentscope-tuner
    ```

2. Set up dependencies.
    ```bash
    uv venv --python=3.10 && source .venv/bin/activate
    uv pip install -e .[trinity]  # or `uv pip install -e .[verl]` if you prefer using verl as backbone
    uv pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache  # Hint: flash-attn must be installed after other deps
    ```

### Getting Started: Tutorial Example Library

Explore our rich library of examples to kickstart your journey:

- 🚀 [**Training a math agent that can write python code**](docs/en/example_math_agent.md).
- 🍳 [**Developing Werewolves RPG agents and training them**](docs/en/example_werewolves.md).
- ⚙️ [**Learning to ask questions like a doctor**](docs/en/example_learning_to_ask.md).
- 💼 [**Creating an AppWorld agent using AgentScope and training it**](docs/en/example_app_world.md).
- 📊 [**Writing a countdown game using AgentScope and solving it**](docs/en/example_countdown.md).
- 🚀 [**Solving a frozen lake walking puzzle using ASTuner**](docs/en).
- 📔 [**Learn how to train using user feedback tracing**](tutorial/example_feedback_tracing/README.md).


## 💡 Architecture

AgentScope Tuner makes agent fine-tuning unprecedentedly straightforward. It encapsulates complex fine-tuning training into a simple module driven by three core inputs:

- **Workflow**: User-defined agent workflow to finish customized tasks. Accepts single-agent or multi-agent workflows, including workflows that mix trainable and non-trainable agents.
- **Task Dataset**: Tasks used in reinforcement learning or a Gym environment, expected to contain a training set and a validation set.
- **Task Judge**: User-defined reward function used to generate a score after each workflow episode. **If the workflow computes reward inside itself** and returns a non-empty reward, the task judge will be **ignored**.



<div align="center">
<img width="480" alt="image" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>

</div>

Of course, fine-tuning the workflow would not be possible without the support of the following core modules:

- Launcher Module: Routing program according to `--backbone`. Managing background service processes.
- Task Rollout Module: Unifying LLM engine APIs and managing the Gym environment and retry mechanism.
- Task Runner Module: Running the **Workflow** and **Task Judge**.
- Model Tuner Module: Forwarding the request to the LLM engine.
- Context Tracker Module: Tracking LLM requests and responses, automatically merging shared-history timelines, and generating training samples.





<!-- - context tracker: A loyal recorder that monitors every LLM call and automatically identifies and archives LLM requests belonging to the same Agent and the same timeline. At the end of the task, it marks the loss mask, merges the recorded LLM input-output timelines, and improves training efficiency by 3 to 10 times. -->
<!-- - Model Tuner Module: When the AgentScope workflow sends an LLM inference request, this busy component directly receives and forwards the request to the LLM engine. -->
<!-- - Task Runner Module: The worker responsible for executing the user-provided AgentScope workflow and performing reward calculations. -->
<!-- - Task Rollout Module: Bridges different LLM engines (such as FSDP, VLLM, etc.) from different training backbones, implements a retry mechanism, and passes the tasks read by the task reader. If the gym environment is required, it initializes the gym environment and ensures resource cleanup. -->
<!-- As a multi-backbone project, astuner ensures that users can toggle between debug mode and training mode with a flip of switch (the `--backbone` argument). It can also manage background service processes related to training, such as ray, external vllm, etc. -->
<!-- - Launcher: The entry point of the project, helping developers quickly switch between debugging the backbone and training the backbone. It also launches and intelligently monitors the environment service processes related to training in the background. -->
<!-- (which can directly utilize your pre-written AgentScope workflows) -->
<!-- ![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764207569988-69b6926f-301b-4766-9199-3823974aab99.png) -->
<!-- ![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764705947150-753d77f0-a1a7-4491-8b8b-a0f9f998ed0a.png) -->


## 🗺️ Roadmap

ASTuner is a constantly evolving project. We are planning to add the following features in the near future.

- [ ] Advanced LLM-based multi-agent reinforcement learning.
- [ ] Training dataset generation from few-shot samples.
- [ ] Multi-modal training support.
- [ ] Cross-process Tuner wrapper to pass though process forking.
- [ ] Providing training → user feedback → data augmentation → retraining data flywheel example.
- [ ] Optimize configurations for long-context adaptation on smaller GPUs.
- [ ] Add LoRA training examples.
- [ ] Covering LangGraph and AutoGen frameworks.


<!-- - **Data Augmentation & User Feedback Tracing**
  Automatically augment training data and trace user feedback when training data is limited.

- **Auto Rubrics**
  Generate LLM-as-judge reward functions by learning from few-shot examples.

- **Multi-Agent Support**
  Build advanced cooperative multi-agent systems with ease.

- **Highly Efficient Async Training-Inference Separation**
  Powered by Trinity-RFT for optimized performance.

- **Training-Debugging Integration**
  Seamlessly toggle between training and debugging modes using a simple `--backbone` switch (`--backbone=trinity` or `--backbone=debug`).

- **Comprehensive Logging**
  Integrate message-level logging from AgentScope Studio and token-level logging for detailed insights. -->




<!-- ### Get Started with Tutorials -->
<!--
Explore our rich library of examples to kickstart your journey:

- Build a math agent specialized in GSM8K problems and [learn how to train it 🚀](tutorial/math_agent.md).
- Create an AppWorld agent using AgentScope and [train it 🪐](tutorial/appworld.md).
- Develop a Werewolves RPG agent and [train it 🚀](tutorial/werewolves/werewolves.md).
- @qingxu: Building advanced multiagent workflow. [Here](docs/en/multiagent_workflow.md)
- @chencheng: Training using user feedback tracing. [Here](docs/en/tracing_user_feedback.md)
- @liuqi: Buildind dataset tasks from document materials. [Here](docs/en/dataset_from_docs.md)
- @yongyi: Expanding dataset from just a few samples. [Here](docs/en/dataset_expansion.md)
- @zhuohua/@lipeng: Writing LLM-as-Judge. [Here](docs/en/llm_as_judge.md)
- @lipeng: Learn to build LLM reward from few-shot example. [Here](docs/en/rubrics_judge.md) -->

<!-- **AgentScope Tune (ASTuner)** is a cutting-edge, user-friendly training framework engineered to optimize AgentScope agents & workflows, and perform fine-tuning of underlying LLM weights seamlessly.

Simply provide your AgentScope workflow, training data, and reward function, and we are ready to enhance your agents to their optimal performance! -->

<!-- <div align="center">
<img width="480" alt="image" src="https://github.com/user-attachments/assets/441dc0bc-34e0-43a7-b5ab-895f695104a4"/>
    <p style="margin-top: 10px; color: #666; font-size: 10px;">
    <em>Deep-Tune Your AgentScope Workflow and Unleash Its Full Potential.</em>
  </p>
</div> -->
