# AgentScope Tuner Documentation

AgentScope Tune, or **ASTuner**, is an advanced agent training framework designed to optimize AgentScope workflows and agents.
Simply provide your AgentScope workflow, training data, and reward function to enhance your agents to their optimal performance.

<div align="center">
<img width="480" alt="image" src="https://github.com/user-attachments/assets/441dc0bc-34e0-43a7-b5ab-895f695104a4"/>
    <p style="margin-top: 10px; color: #666; font-size: 10px;">
    <em>Deep-Tune Your AgentScope Workflow and Unleash Its Full Potential.</em>
  </p>
</div>

## ✨ Features

- **Data Augmentation & User Feedback Tracing**
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
  Integrate message-level logging from AgentScope Studio and token-level logging for detailed insights.



## 🚀 Train Your Agents

### Installation

<details>
<summary>1. Native conda or uv deployment</summary>

We recommend using `uv` for dependency management, though `conda` is also supported.

1. Clone the repository and Trinity module:
    ```bash
    git clone https://github.com/..../agentscope-tuner.git astuner
    git clone https://github.com/binary-husky/Trinity-RFT astuner/external/trinity
    cd astuner
    ```

2. Install the Trinity training backbone:
    ```bash
    uv venv --python=3.10.16  # Create virtual environment
    source .venv/bin/activate  # Activate virtual environment

    # Install dependencies (execute in order)
    uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
    uv pip install -r scripts/requirements_trinity.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow  #  for conda, remove `--prerelease=allow` option
    uv pip install -e external/trinity -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
    uv pip install agentscope==1.0.7 -i https://mirrors.aliyun.com/pypi/simple/
    uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation  # Install flash attention (must be executed at last)
    ```
</details>

<details>
<summary>2. Docker container installation</summary>

@xuchen

</details>

### Get Started with Tutorials

Explore our rich library of examples to kickstart your journey:

- Build a math agent specialized in GSM8K problems and [learn how to train it 🚀](tutorial/math_agent.md).
- Create an AppWorld agent using AgentScope and [train it 🪐](tutorial/appworld.md).
- Develop a Werewolves RPG agent and [train it 🚀](tutorial/werewolves/werewolves.md).
- @qingxu: Building advanced multiagent workflow. [Here](docs/en/multiagent_workflow.md)
- @chencheng: Training using user feedback tracing. [Here](docs/en/tracing_user_feedback.md)
- @liuqi: Buildind dataset tasks from document materials. [Here](docs/en/dataset_from_docs.md)
- @yongyi: Expanding dataset from just a few samples. [Here](docs/en/dataset_expansion.md)
- @zhuohua/@lipeng: Writing LLM-as-Judge. [Here](docs/en/llm_as_judge.md)
- @lipeng: Learn to build LLM reward from few-shot example. [Here](docs/en/rubrics_judge.md)

## 🏗️ Project Overview

### Architecture

1. **Task Reader** (config field: `astuner.task_reader`)
   - `astuner/task_reader/task_reader_base.py`
     - `TaskReaderEnvService`
     - `TaskReaderJsonl`
     - `TaskReaderHuggingFace`

2. **Workflow Definition** (config field: `astuner.rollout.agentscope_learn_protocol`)
   - `tutorial/appworld.py`
   - `tutorial/math_agent.py`

3. **Reward Function** (config field: `astuner.task_judge.judge_protocol`)
   - `astuner/task_judge/judge_base.py`
   - `astuner/task_judge/env_service_as_judge.py`
     - `EnvServiceJudge`
   - `astuner/task_judge/math_answer_as_judge.py`
     - `MathAnswerAsJudge`
     - `MathAnswerAndLlmAsJudge`

4. **Model Specification** (config field: `astuner.model.path`)

5. **Configuration System** (under improvement)
   - Default Configurations:
     - `astuner/default_config/default.yaml` (default VERL training config, overridden by `--conf` YAML)
     - `astuner/default_config/trinity_default.yaml` (default Trinity config, overridden via `trinity.xxx` in `--conf` YAML)
   - Auto-Alignment:
     - `astuner/default_config/config_auto_convertion_verl.jsonc`
     - `astuner/default_config/config_auto_convertion_trinity.jsonc`

6. **ASTuner & AgentScope Interaction System V0.5**
   - Managed by `astuner/context_tracker/agentscope.py`:
     - Processes tokens generated by AgentScope
     - Caches data required for judging (e.g., dialogue messages, env_service handles, task metadata)
     - Bridges LLM interactions
     - Merges timelines



## 🗺️ Project Roadmap

Working in progress:

- Enhance data generation module functionality
- Provide a training → user feedback → data augmentation → retraining data flywheel example
- Offer refined post-processing options for multi-agent samples
- Support training with multiple models
- Optimize configurations for long-context adaptation on smaller GPUs
- Add LoRA training examples
