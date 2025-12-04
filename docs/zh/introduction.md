# AgentScope Tuner

AgentScope Tuner，简称 **ASTuner**，是一个专用于优化 AgentScope 智能体的先进训练框架。
你只需提供 AgentScope 工作流、训练数据与奖励函数，ASTuner 即可帮助将智能体优化到最佳状态。



## ✨ 特性

- **数据增强 & 数据回流追踪**：当训练数据有限时，自动进行数据增强，并追踪用户反馈。
- **自动 Rubrics 构建**：通过少量示例学习，自动构建基于 LLM-as-a-judge 的奖励函数。
- **多智能体支持**：轻松构建高级协作多智能体系统。
- **高效的异步训练-推理分离**：基于 Trinity-RFT 实现性能优化。
- **训练-调试一体化**：通过简单的 `--backbone` 开关即可在训练与调试模式间无缝切换（`--backbone=trinity` 或 `--backbone=debug`）。
- **完备的日志记录**：集成来自 AgentScope Studio 的消息级日志与 token 级日志，便于深入分析。



## 🚀 快速开始

### 安装

优先推荐使用 `uv`，也同样支持 `conda`。

1. 克隆仓库：

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

2. 创建虚拟环境并安装依赖：

```bash
uv venv --python=3.10.16  # 创建虚拟环境
source .venv/bin/activate  # 激活虚拟环境

uv pip install -e .[dev]
uv pip install flash_attn==2.8.1 --no-build-isolation --no-cache-dir
```


### 开始使用

你可以从我们丰富的示例库入手学习使用方法：

- 🚀 **教程（Tutorial）**：手把手训练你的第一个智能体。
    - **安装（Installation）**：了解如何安装 AgentScope Tuner。
  	- **快速开始（Quick Start）**：从零开始训练你的第一个智能体。
	- **配置（Configuration）**：配置数据、优化算法、奖励函数等。
- ⚙️ **组件（Component）**：理解各个组件的工作原理。
	- **工作流（Workflow）**：构建你自己的、可训练的智能体工作流。
	- **数据流水线与生成（Data Pipeline & Generation）**：包含从文档材料构建数据集任务，以及从少量样本扩展数据集。
	- **奖励建模（Reward Modeling）**：学习如何优雅地实现基于 rubrics 的智能体训练奖励。
	- **追踪-反馈训练（Tracing-Feedback Training）**：学习如何利用数据回流来进行训练。
- 🍳 **Cookbook 示例**
    - **构建一个简单的数学智能体**：基于 GSM8K 任务，学习如何训练一个简单的 Agent。
    - **构建 AppWorld 智能体**：基于 AgentScope 构建一个较复杂的 AppWorld 智能体，并对其进行训练。
    - **构建多智能体狼人杀游戏**：开发能参与狼人杀博弈的多智能体训练。


## 🏗️ 项目概览

### 架构

AgentScope Tuner 简化了智能体微调流程，将复杂的训练封装进三类核心模块：

- AgentScope 工作流（可直接复用已有 AgentScope 工作流）
- 任务数据集（提供训练数据）
- 奖励评判器（Reward Judge，用于评估表现质量）


![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764207569988-69b6926f-301b-4766-9199-3823974aab99.png)

为顺利完成工作流微调，我们在三类模块之下实现了以下核心模块

![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/144856612/1764705947150-753d77f0-a1a7-4491-8b8b-a0f9f998ed0a.png) 
- launcher：项目的入口，帮助开发者在调试 backbone 与训练 backbone 之间快速切换，同时负责启动并智能监控与训练相关的环境服务进程。
- task rollout：桥接不同的 LLM 引擎（如 FSDP、VLLM 等），实现重试机制，并传递 task reader 读取的任务。同时负责 gym 环境的初始化与资源清理。
- task runner：任务执行者，负责真正运行用户提供的 AgentScope 工作流，同时运行评判器并完成初步奖励计算。
- model tuner：当前端 AgentScope 工作流发出 LLM 推理请求时，该组件会直接接收并将请求转发给 LLM 引擎。
- context tracker：上下文记录员，监控每一次 LLM 调用，自动识别并归档属于同一 Agent、同一时间线的 LLM 请求。在任务结束时，负责标记 loss 掩膜，合并 LLM 输入输出，从而将训练效率提升约 3～10 倍。

## 🗺️ 项目规划

正在进行中的工作：

- 增强数据生成模块的功能
- 提供一个「训练 → 用户反馈 → 数据增强 → 重新训练」的飞轮式示例
- 为多智能体样本提供更精细的后处理方案
- 支持多模型联合训练
- 针对小显存 GPU 优化长上下文适配配置
- 增加基于 LoRA 的训练示例
