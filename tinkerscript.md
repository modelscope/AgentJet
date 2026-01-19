# TinkerScript Design Blueprint / TinkerScript 设计蓝图

[English](#english-version) | [中文](#chinese-version)

---

<a id="english-version"></a>
## 🇬🇧 English Version

### 1. Overview
**TinkerScript** is an experimental component of AgentJet designed to decouple the **Training Logic** from the **Agent Execution Logic**. It allows users to train **full-weight LLM models** on machines without GPUs (e.g., a laptop) by offloading the actual model computation to a remote GPU server.

Unlike traditional setups where the user code must run inside the training cluster, TinkerScript allows you to verify and run your agent logic locally while the heavy lifting (training & inference) happens remotely.

### 2. Core Architecture

The system involves two main parties: the **TinkerScript Server** (running on the GPU cluster) and the **TinkerScript Client** (running on your local machine).

```mermaid
graph TD
    subgraph "GPU Cluster (Server Side)"
        TrainingLoop["Training Loop (AgentJet/GRPO)"]
        TSS["TinkerScript Server (FastAPI)"]
        ZMQ["ZeroMQ / IPC"]
        SharedMem[("Shared Memory")]
        LLM["LLM Engine (vLLM/SGLang)"]
    end

    subgraph "User Laptop / CPU Cluster (Client Side)"
        UserScript["User Script (python while loop)"]
        AgentLogic["Agent Logic / Tools"]
    end

    TrainingLoop -- "1. Generate Task" --> SharedMem
    SharedMem -- "2. Register Episode" --> TSS

    UserScript -- "3. Claim Episode (HTTP)" --> TSS
    TSS -- "4. Return API Key & Base URL" --> UserScript

    UserScript -- "5. Inference (OpenAI API)" --> LLM
    LLM -- "Token Stream" --> UserScript

    UserScript -- "6. Submit Reward (HTTP)" --> TSS
    TSS -- "7. Push Result" --> ZMQ
    ZMQ -- "8. Update Weights" --> TrainingLoop
```

### 3. Detailed Workflow

The workflow relies on a "Claim & Submit" model. The training loop generates tasks ("Episodes") and waits for external workers to pick them up.

```mermaid
sequenceDiagram
    participant TL as Training Loop (Internal)
    participant S as Server (FastAPI)
    participant C as Client (User Script)
    participant M as LLM Model

    Note over TL, S: 1. Task Generation
    TL->>S: Register Episode (Status: Unclaimed)

    Note over C, S: 2. Task Acquisition
    loop Worker Loop
        C->>S: POST /claim_episode
        alt No Tasks
            S-->>C: Retry Later
        else Task Available
            S->>S: Mark as "Claimed"
            S-->>C: Return {EpisodeID, OpenAI_BaseURL, API_Key}
        end

        Note over C, M: 3. Execution (Rollout)
        C->>M: Chat Completion Request (Inference)
        M-->>C: Response (Generation)
        C->>C: Calculate Reward (e.g., Verify Math Answer)

        Note over C, S: 4. Result Submission
        C->>S: POST /end_episode {Reward, Metadata}
        S->>TL: Forward Result via ZeroMQ
        S->>S: Delete Episode Record (Complete)
    end
```

### 4. Episode State Machine

To handle network failures or client crashes, the server maintains a state machine for every episode.

```mermaid
stateDiagram-v2
    [*] --> Registered
    Registered --> Unclaimed_Queue : Add to Queue

    Unclaimed_Queue --> Claimed : Client requests task

    Claimed --> Completed : Client submits result
    Claimed --> Registered : Client Timeout / Crash

    Completed --> [*] : Removed from Memory
```

*   **Registered**: Task created by the training algorithm.
*   **Claimed**: A client is currently working on it.
*   **Timeout**: If a client claims a task but doesn't report back within `allow_discard_timeout`, the server reverts the status to **Registered** so another client can try.

### 5. Implementation Example

The user experience is designed to be minimal. You simply query the remote server for a "job", do the work, and report the "score".

```python
# User-side Code Concept
def rollout(task):
    # 1. Handshake & Claim (Get credentials for this specific episode)
    api_baseurl_key = tinkerjet_remote.begin_episode()

    # 2. Run your existing agent logic using standard OpenAI format
    workflow_output = execute_agent(task, api_baseurl_key)

    # 3. Submit results
    tinkerjet_remote.end_episode(workflow_output)
    return workflow_output.reward
```

### 6. Limitations

1.  **Strict OpenAI Protocol**: Users must use the OpenAI `base_url` + `api_key` pattern. Internal access (like direct model object access) is not available.
2.  **Implicit Multi-Agent Handling**: AgentJet cannot explicitly distinguish different agents in a multi-agent scenario via API, though it attempts to merge timeline shards automatically.
3.  **No Prompt Tuning**: TinkerScript is designed for full-weight model training, not for soft-prompt tuning.

---

<a id="chinese-version"></a>
## 🇨🇳 中文版本 (Chinese Version)

### 1. 概述 (Overview)
**TinkerScript** 是 AgentJet 的一个实验性组件，旨在将 **训练逻辑 (Training Logic)** 与 **Agent 执行逻辑 (Execution Logic)** 解耦。它允许用户在 **没有 GPU** 的机器上（例如普通笔记本电脑）训练 **全参数 LLM 模型**，计算压力完全由远程 GPU 服务器承担。

与传统的将用户代码嵌入训练集群的方式不同，TinkerScript 允许你在本地运行并验证 Agent 逻辑，通过网络与远程训练循环交互。

### 2. 核心架构 (Core Architecture)

系统包含两个主要部分：运行在 GPU 集群上的 **TinkerScript Server** 和运行在本地的 **TinkerScript Client**。

```mermaid
graph TD
    subgraph "GPU 集群 (Server 端)"
        TrainingLoop["训练循环 (AgentJet/GRPO)"]
        TSS["TinkerScript Server (FastAPI)"]
        ZMQ["ZeroMQ / IPC 通信"]
        SharedMem[("共享内存")]
        LLM["LLM 推理引擎 (vLLM/SGLang)"]
    end

    subgraph "用户笔记本 / CPU 集群 (Client 端)"
        UserScript["用户脚本 (Python While Loop)"]
        AgentLogic["Agent 业务逻辑 / 工具调用"]
    end

    TrainingLoop -- "1. 生成任务 (Task)" --> SharedMem
    SharedMem -- "2. 注册 Episode" --> TSS

    UserScript -- "3. 领取任务 (HTTP Claim)" --> TSS
    TSS -- "4. 返回 API Key 与 Base URL" --> UserScript

    UserScript -- "5. 推理请求 (OpenAI 协议)" --> LLM
    LLM -- "生成 Token 流" --> UserScript

    UserScript -- "6. 提交 Reward (HTTP End)" --> TSS
    TSS -- "7. 推送结果" --> ZMQ
    ZMQ -- "8. 更新权重" --> TrainingLoop
```

### 3. 详细工作流 (Detailed Workflow)

基于“领取 (Claim) - 提交 (Submit)”模式。训练循环生成任务（Episode），等待外部 Worker 领取执行。

```mermaid
sequenceDiagram
    participant TL as 训练循环 (内部)
    participant S as Server (FastAPI)
    participant C as Client (用户脚本)
    participant M as LLM 模型服务

    Note over TL, S: 1. 任务生成阶段
    TL->>S: 注册 Episode (状态: Unclaimed)

    Note over C, S: 2. 任务领取阶段
    loop Worker Loop
        C->>S: POST /claim_episode (请求任务)
        alt 无可用任务
            S-->>C: 请稍后重试
        else 有可用任务
            S->>S: 标记为 "Claimed"
            S-->>C: 返回 {EpisodeID, OpenAI_BaseURL, API_Key}
        end

        Note over C, M: 3. 执行阶段 (Rollout)
        C->>M: Chat Completion 请求 (推理通过网络回传)
        M-->>C: 返回生成结果
        C->>C: 计算 Reward (例如: 验证数学答案)

        Note over C, S: 4. 结果提交阶段
        C->>S: POST /end_episode {Reward, Metadata}
        S->>TL: 通过 ZeroMQ 转发结果给训练器
        S->>S: 删除 Episode 记录 (完成)
    end
```

### 4. 状态机管理 (Episode State Machine)

为了处理网络波动或客户端崩溃（Crash），服务端为每个 Episode 维护了一个状态机。

```mermaid
stateDiagram-v2
    [*] --> Registered (已注册)
    Registered --> Unclaimed_Queue : 加入待领取队列

    Unclaimed_Queue --> Claimed (已被领取) : 客户端请求任务

    Claimed --> Completed (已完成) : 客户端提交结果
    Claimed --> Registered (已注册) : 客户端超时 / 崩溃

    Completed --> [*] : 从内存中移除
```

*   **Registered (已注册)**: 训练算法生成了该任务，等待被执行。
*   **Claimed (已被领取)**: 某个 Client 正在处理该任务。
*   **Timeout (超时)**: 如果 Client 领取任务后在规定时间 (`allow_discard_timeout`) 内未提交结果，服务器会将状态重置为 **Registered**，允许其他 Client 重新领取该任务（容错机制）。

### 5. 实现代码示例

用户侧的代码非常简洁。简而言之：向远程服务器要一个“活儿”，干完活，上报“得分”。

```python
# 用户侧代码概念演示
def rollout(task):
    # 1. 握手 & 领取任务 (获取当前 Episode 专属的鉴权信息)
    api_baseurl_key = tinkerjet_remote.begin_episode()

    # 2. 运行你现有的 Agent 逻辑 (使用标准 OpenAI 接口)
    workflow_output = execute_agent(task, api_baseurl_key)

    # 3. 提交结果
    tinkerjet_remote.end_episode(workflow_output)
    return workflow_output.reward
```

### 6. 局限性 (Limitations)

1.  **严格依赖 OpenAI 协议**: 用户必须使用 OpenAI `base_url` + `api_key` 的方式与模型交互。无法获取模型内部对象（Weights/Gradients）。
2.  **隐式多智能体处理**: 在多智能体（Multi-Agent）场景下，AgentJet 无法通过 API 显式区分不同的 Agent 角色，但后台会尝试自动合并时间线片段。
3.  **不支持 Prompt Tuning**: TinkerScript 专为全量模型微调设计，不支持 Soft-Prompt Tuning 等轻量级微调。
