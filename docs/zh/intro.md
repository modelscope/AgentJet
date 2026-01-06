# 项目简介

**AgentScope Tuner (ASTuner)** 是一款前沿且易用的训练框架，专为优化 AgentScope 中的智能体和工作流（Workflow）而设计，能够自动微调背后的语言模型权重。

您只需提供 AgentScope 工作流、训练数据和奖励函数，我们就能助您的智能体达到最佳性能！

---

## 特性

我们致力于构建一个易于上手的 AgentScope 微调工具，为智能体开发者解锁更多可能性：

<div class="key-features" markdown>

- <img src="https://api.iconify.design/lucide:rocket.svg" class="inline-icon" />&nbsp;**简单友好** — ASTuner 帮助您轻松微调智能体工作流背后的模型，以极小的开发成本优化智能体性能。

- <img src="https://api.iconify.design/lucide:book-open.svg" class="inline-icon" />&nbsp;**丰富的教程库** — ASTuner 提供了丰富的[示例库](#example-library)作为学习教程。
    - 数学智能体、狼人杀游戏、AppWorld 等 <a href="#example-library" class="feature-link">查看示例 <span class="link-arrow">→</span></a>

- <img src="https://api.iconify.design/lucide:zap.svg" class="inline-icon" />&nbsp;**高效且可扩展** — 默认使用 [Trinity](https://github.com/modelscope/Trinity-RFT/) 作为后端（`--backbone=trinity`），通过全异步 RFT 加速微调过程。
    - 支持 [verl](./installation.md) 后端作为备选方案 <a href="./installation/" class="feature-link">了解更多 <span class="link-arrow">→</span></a>

</div>

!!! tip "多智能体支持"
    ASTuner 支持 [多智能体工作流](./workflow.md)，并采用上下文合并技术，当工作流涉及多轮（或多智能体）对话时，可将训练速度提升 **1.5 倍至 20 倍**。

!!! info "可靠性与可复现性"
    我们的团队持续追踪框架在多个 [任务 + 主分支版本 + 训练后端](https://benchmark.agent-matrix.com/) 上的表现。

### 面向进阶研究者

ASTuner 还提供了高分辨率的日志记录和调试方案：

| 功能 | 说明 |
|------|------|
| **高分辨率日志** | 保存并检查 Token 级的 Rollout 详情，记录 Token ID、Token Loss Mask，甚至是 Token Logprobs |
| **快速调试** | 使用 `--backbone=debug` 选项，将代码修改后的等待时间从分钟级缩短至秒级 |

---

## 快速上手

### 安装

我们推荐使用 `uv` 进行依赖管理。

=== "步骤 1：克隆仓库"

    ```bash
    git clone https://github.com/agentscope-ai/agentscope-tuner.git
    cd agentscope-tuner
    ```

=== "步骤 2：设置环境"

    ```bash
    uv venv --python=3.10.16 && source .venv/bin/activate
    uv pip install -e .[trinity]
    # 注意：flash-attn 必须在其他依赖安装完成后安装
    uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir
    ```

### 运行训练

您可以使用预配置的 YAML 文件，通过单条命令开始训练您的第一个智能体：

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

!!! example "了解更多"
    查看 [数学智能体](./example_math_agent.md) 示例获取详细说明。

---

## 示例库 {#example-library}

探索我们丰富的示例库，开启您的探索之旅：

<div class="card-grid">
<a href="./example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>数学智能体</h3></div><p class="card-desc">训练一个可以编写 Python 代码解决数学问题的智能体。</p></a>
<a href="./example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld 智能体</h3></div><p class="card-desc">使用 AgentScope 创建并训练 AppWorld 智能体。</p></a>
<a href="./example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>狼人杀游戏</h3></div><p class="card-desc">开发狼人杀 RPG 智能体并训练它们进行策略博弈。</p></a>
<a href="./example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>学会提问</h3></div><p class="card-desc">学习像医生一样在医疗咨询场景中进行提问。</p></a>
<a href="./example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>倒计时游戏</h3></div><p class="card-desc">使用 AgentScope 编写倒计时游戏并用 RL 求解。</p></a>
<a href="./example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>冰湖问题</h3></div><p class="card-desc">使用 ASTuner 的强化学习解决冰湖行走谜题。</p></a>
</div>

---

## 核心概念

ASTuner 通过将开发者接口与内部执行逻辑分离，使智能体微调变得直观明了。

<div align="center">
<img width="480" alt="ASTuner 架构图" src="https://img.alicdn.com/imgextra/i1/O1CN01xnkGyf1j8szYYxt5U_!!6000000004504-0-tps-2261-1471.jpg"/>
</div>

### 1. 以用户为中心的接口

为了优化智能体，您需要提供三个核心输入：

<div class="card-grid">
<a href="./workflow/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:graph.svg" class="card-icon card-icon-agent" alt=""><h3>可训练工作流</h3></div><p class="card-desc">通过继承 Workflow 类定义您的智能体逻辑，支持简单和多智能体设置。</p></a>
<a href="./data_pipeline/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>任务读取器</h3></div><p class="card-desc">从 JSONL 文件、HuggingFace 数据集加载任务，或自动从文档生成。</p></a>
<a href="./task_judger/" class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>任务评判器</h3></div><p class="card-desc">评估智能体输出并分配奖励以指导训练过程。</p></a>
</div>

### 2. 内部系统架构

内部系统协调多个专门模块，以处理强化学习（RL）训练和智能体交互的复杂性。

| 模块 | 说明 |
|------|------|
| **启动器 (Launcher)** | 管理后台服务进程（Ray, vLLM）并路由后端 |
| **任务读取器 (Task Reader)** | 处理数据摄取、增强和过滤 |
| **任务展开 (Task Rollout)** | 连接大语言模型（LLM）引擎并管理 Gym 环境生命周期 |
| **任务运行器 (Task Runner)** | 执行 AgentScope 工作流并计算奖励 |
| **模型微调器 (Model Tuner)** | 将工作流中的推理请求转发至 LLM 引擎 |
| **上下文追踪器 (Context Tracker)** | 监控 LLM 调用，并自动合并共享历史的时间线（**3-10 倍**效率提升） |

---

## 下一步

<div class="card-grid">
<a href="./installation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:download.svg" class="card-icon card-icon-tool" alt=""><h3>安装指南</h3></div><p class="card-desc">设置 ASTuner 环境和依赖项。</p></a>
<a href="./quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>快速开始</h3></div><p class="card-desc">几分钟内运行您的第一次训练。</p></a>
</div>
