# 项目简介

**AgentJet (AgentJet)** 是一款前沿且易用的训练框架，专为优化 AgentScope 中的智能体和工作流（Workflow）而设计，能够自动微调背后的语言模型权重。

您只需提供 AgentScope 工作流、训练数据和奖励函数，我们就能助您的智能体达到最佳性能！

---

### 特性

我们致力于构建一个易于上手的 AgentScope 微调工具，为智能体开发者解锁更多可能性：

* **简单友好**：AgentJet 帮助您轻松微调智能体工作流背后的模型，以极小的开发成本优化智能体性能。
* **丰富的教程库**：AgentJet 提供了丰富的 [示例库](#example-library) 作为学习教程。
* **高效且可扩展**：AgentJet 默认使用 [trinity](https://github.com/modelscope/Trinity-RFT/) 作为后端（`--backbone=trinity`），通过全异步 RFT 加速微调过程。如果您更倾向于 Actor 共位置部署，也可以回退到 [verl](./installation.md) 后端。
* **灵活且快速**：AgentJet 支持 [多智能体工作流](./workflow.md)，并采用了上下文合并技术。当工作流涉及多轮（或多 智能体）对话时，可将训练速度提升 1.5 倍至 20 倍。
* **可靠性与可复现性**：我们的团队持续追踪框架在多个 [任务 + 主分支版本 + 训练后端](https://benchmark.agent-matrix.com/) 上的表现（正在建设中，数据收集后即将上线）。

针对进阶研究者，AgentJet 还提供了高分辨率的日志记录和调试方案：

* **高分辨率日志**：AgentJet 允许用户保存并检查 Token 级的 Rollout 详情，记录 Token ID、Token Loss Mask，甚至是 Token Logprobs，以便于工作流开发和 智能体 诊断。
* **快速调试**：AgentJet 提供了 `--backbone=debug` 选项，提供极致的调试体验，将代码修改后的等待时间从分钟级缩短至秒级，并支持在 IDE 中进行断点调试。

---

### 快速上手

#### 安装

我们推荐使用 `uv` 进行依赖管理。

1. **克隆仓库**：

```bash
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet

```

2. **设置环境**：

```bash
uv venv --python=3.10.16 && source .venv/bin/activate
uv pip install -e .[trinity]
# 注意：flash-attn 必须在其他依赖安装完成后安装
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir

```

#### 运行训练

您可以使用预配置的 YAML 文件，通过单条命令开始训练您的第一个 智能体。以 [数学 智能体](./example_math_agent.md) 为例：

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray

```

<span id="example-library"></span>
#### 示例库

探索我们丰富的示例库，开启您的探索之旅：

* <img src="https://api.iconify.design/lucide:calculator.svg" class="inline-icon" /> **[训练一个可以编写 Python 代码的数学智能体](./example_math_agent.md)**。
* <img src="https://api.iconify.design/lucide:smartphone.svg" class="inline-icon" /> **[使用 AgentScope 创建并训练 AppWorld 智能体](./example_app_world.md)**。
* <img src="https://api.iconify.design/lucide:users.svg" class="inline-icon" /> **[开发并训练"狼人杀"RPG智能体](./example_werewolves.md)**。
* <img src="https://api.iconify.design/lucide:stethoscope.svg" class="inline-icon" /> **[学习像医生一样进行提问](./example_learning_to_ask.md)**。
* <img src="https://api.iconify.design/lucide:timer.svg" class="inline-icon" /> **[使用 AgentScope 编写并解决"倒计时"游戏](./example_countdown.md)**。
* <img src="https://api.iconify.design/lucide:footprints.svg" class="inline-icon" /> **[使用 AgentJet 解决"冰湖"行走谜题](./example_frozenlake.md)**。

---

### 核心概念

AgentJet 通过将开发者接口与内部执行逻辑分离，使智能体微调变得直观明了。

<div align="center">
<img width="480" alt="image" src="[https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg](https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg)"/>

</div>

#### 1. 以用户为中心的接口

为了优化智能体，您需要提供三个核心输入：

* **[可训练工作流 (Trainable Workflow)](./workflow.md)**：通过继承 `Workflow` 类来定义您的智能体逻辑，支持简单的智能体设置以及高级的多智能体协作。
* **[任务读取器 (Task Reader)](./data_pipeline.md)**：从 JSONL 文件、HuggingFace 数据集、交互式环境加载训练任务，或从文档自动生成任务。
* **[任务判别器 (Task Judger)](./task_judger.md)**：评估智能体输出并分配奖励（Reward）以指导训练。

#### 2. 内部系统架构

内部系统协调多个专门模块，以处理强化学习（RL）训练和智能体交互的复杂性。

* **启动器 (Launcher)**：管理后台服务进程（Ray, vLLM）并路由后端。
* **任务读取器 (Task Reader)**：处理数据摄取、增强和过滤。
* **任务展开 (Task Rollout)**：连接大语言模型（LLM）引擎并管理 Gym 环境生命周期。
* **任务运行器 (Task Runner)**：执行 AgentScope 工作流并计算奖励。
* **模型微调器 (Model Tuner)**：将工作流中的推理请求转发至 LLM 引擎。
* **上下文追踪器 (Context Tracker)**：监控 LLM 调用，并自动合并共享历史的时间线，将训练效率提升 **3 到 10 倍**。

---

### 导航

* <img src="https://api.iconify.design/lucide:book-open.svg" class="inline-icon" /> **教程**：从 [安装](./installation.md) 到 [微调您的第一个智能体](./tune_your_first_agent.md) —— 初学者的必经之路。
* <img src="https://api.iconify.design/lucide:wrench.svg" class="inline-icon" /> **核心组件**：定义您的 [可训练工作流](./workflow.md)，并管理 [数据](./data_pipeline.md) 和 [奖励](./tune_your_first_agent.md)。
* <img src="https://api.iconify.design/lucide:lightbulb.svg" class="inline-icon" /> **示例**：查看上方的 [示例库](#example-library)，了解 [数学](./example_math_agent.md)、[狼人杀游戏](./example_werewolves.md) 和 [学习提问](./example_learning_to_ask.md) 等真实案例。
* <img src="https://api.iconify.design/lucide:settings.svg" class="inline-icon" /> **深入了解**：掌握高级 [配置方案](./configuration.md)。
