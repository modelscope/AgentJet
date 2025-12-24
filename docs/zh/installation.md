# 项目简介

**AgentScope Tuner (ASTuner)** 是一款领先且易用的训练框架，专为优化 AgentScope 中的 Agent 和工作流（Workflow）而设计，能够自动微调后台大语言模型的权重。

只需提供您的 AgentScope 工作流、训练数据以及奖励函数（Reward Function），ASTuner 即可助力您的 Agent 达到最佳性能状态！

---

### ✨ 特性

我们致力于构建一个简单易学的 AgentScope 微调器，为 Agent 开发者开启更多可能性：

* **简单友好**：ASTuner 帮助您轻松微调 Agent 工作流背后的模型，以极小的开发成本实现 Agent 性能的飞跃。
* **丰富的教程库**：ASTuner 提供了丰富的 [示例库](https://github.com/agentscope-ai/agentscope-tuner/tree/main/tutorial) 作为学习教程。
* **高效且可扩展**：ASTuner 默认使用 [trinity](https://github.com/modelscope/Trinity-RFT/) 作为后端（`--backbone=trinity`），通过全异步 RFT 加速微调过程。如果您更偏好 Actor 共位置部署，也可以切换至 [verl](./installation.md) 后端。
* **灵活且快速**：ASTuner 支持 [多 Agent 工作流](./workflow.md)，并采用了时间线合并技术（Timeline Merging），在涉及多轮或多 Agent 对话的工作流中，可将训练速度提升 1.5 倍至 20 倍。
* **可靠性与可复现性**：我们的团队持续追踪框架在多个 [任务 + 主分支版本 + 训练后端](https://benchmark.agent-matrix.com/) 上的表现（正在建设中，数据收集后即将上线）。

针对资深研究人员，ASTuner 还提供了高分辨率的日志记录和调试方案：

* **高分辨率日志**：ASTuner 允许用户保存并检查 Token 级的 Rollout 详情，记录 Token ID、Loss Mask 甚至 Token Logprobs，以便进行工作流开发和 Agent 诊断。
* **快速调试**：ASTuner 提供了 `--backbone=debug` 选项，提供极致的调试体验，将代码修改后的等待时间从分钟级缩短至秒级，并支持在 IDE 中进行断点调试。

---

### 🚀 快速上手

#### 安装

我们推荐使用 `uv` 进行依赖管理。

1. **克隆仓库**：

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner

```

2. **配置环境**：

```bash
uv venv --python=3.10.16 && source .venv/bin/activate
uv pip install -e .[trinity]
# 注意：flash-attn 必须在其他依赖安装完成后安装
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir

```

#### 运行训练

您可以使用预配置的 YAML 文件，通过单条命令开始训练您的第一个 Agent：

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray

```

详情请参阅 [数学 Agent 示例](./example_math_agent.md)

#### 示例库

探索我们丰富的示例库，开启您的开发之旅：

* 🔢 **[训练一个能编写 Python 代码的数学 Agent](./example_math_agent.md)**。
* 📱 **[使用 AgentScope 创建并训练 AppWorld Agent](./example_app_world.md)**。
* 🐺 **[开发并训练“狼人杀”RPG Agent](./example_werewolves.md)**。
* 👩🏻‍⚕️ **[学习像医生一样进行问诊](./example_learning_to_ask.md)**。
* 🎴 **[使用 AgentScope 编写并解决“倒计时”游戏](./example_countdown.md)**.
* 🚶 **[使用 ASTuner 解决“冰湖”行走谜题](./example_frozenlake.md)**。

---

### 🧩 核心概念

ASTuner 通过将开发者接口与内部执行逻辑分离，使 Agent 微调变得简单直观。

<div align="center">
<img width="480" alt="image" src="[https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg](https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg)"/>

</div>

#### 1. 以用户为中心的接口

为了优化 Agent，您需要提供三个核心输入：

* **[可训练工作流 (Trainable Workflow)](./workflow.md)**：通过继承 `Workflow` 类来定义 Agent 逻辑，支持简单的单 Agent 设置和复杂的多 Agent 协作。
* **[任务读取器 (Task Reader)](./data_pipeline.md)**：从 JSONL 文件、HuggingFace 数据集、交互式环境加载训练任务，或从文档自动生成任务。
* **[任务判别器 (Task Judger)](./task_judger.md)**：评估 Agent 的输出并分配奖励（Reward）以指导训练。

#### 2. 内部系统架构

内部系统协调多个专门模块，处理强化学习（RL）训练和 Agent 交互的复杂性。

* **启动器 (Launcher)**：管理后台服务进程（Ray, vLLM）并路由后端（Backbone）。
* **任务读取器 (Task Reader)**：负责数据的摄取、增强和过滤。
* **任务展开 (Task Rollout)**：连接 LLM 引擎并管理 Gym 环境的生命周期。
* **任务运行器 (Task Runner)**：执行 AgentScope 工作流并计算奖励。
* **模型微调器 (Model Tuner)**：将工作流中的推理请求转发至 LLM 引擎。
* **上下文追踪器 (Context Tracker)**：监控 LLM 调用，并自动合并共享历史的时间线，将训练效率提升 **3 到 10 倍**。

---

### 🚦 导航

* 📖 **教程**：从 [安装](./installation.md) 到 [微调您的第一个 Agent](./tutorial.md) —— 初学者的必经之路。
* 🛠️ **核心组件**：定义您的 [可训练工作流](./workflow.md)，并管理 [数据](./data_pipeline.md) 与 [奖励](./tune_your_first_agent.md)。
* 💡 **示例**：查看上方的 [示例库](%23%E7%A4%BA%E4%BE%8B%E5%BA%93)，了解 [数学](./example_math_agent.md)、[狼人杀游戏](./example_werewolves.md) 和 [学习提问任务](./example_learning_to_ask.md) 等真实案例。
* ⚙️ **深入了解**：掌握高级 [配置方案](./configuration.md)。

---

**您想让我为您详细解释其中某个特定的示例（如数学 Agent 训练）的实现代码吗？**