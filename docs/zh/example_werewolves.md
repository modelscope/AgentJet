# 狼人杀游戏

本教程展示了如何使用 AgentScope Tuner 来处理多智能体训练，令多个智能体在狼人杀游戏中决策、对抗和协作。

---

## 概述

<div class="callout-tip">
<p>
狼人杀角色扮演游戏是一个典型的 POMDP（部分可观测马尔可夫决策过程）问题。我们可以在这个协作型多智能体问题中使用「参数共享」的方法来训练 Agent。
</p>
</div>

| 术语 | 说明 |
|------|------|
| **部分可观测** | Agent 只能接收局部信息。即使属于同一阵营，一个智能体也无法获得其他智能体的观测信息 |
| **马尔可夫决策过程** | 根据当前局面做决策 |
| **参数共享** | 多个智能体使用同一个模型作为策略，但不共享感知（模型输入） |
| **协作型多智能体问题** | Agent 之间目标一致（奖励一致） |

!!! info "实验环境"
    - 使用静态 **Qwen3-235B-A22B** 作为对手（不可训练 Agent）
    - 使用 **Qwen2-7B** 作为可训练 Agent（即 `trainable_targets`）

<div align="center">
<img width="480" alt="狼人杀游戏" src="https://img.alicdn.com/imgextra/i2/O1CN012JgVZC2ABczBhAzJs_!!6000000008165-0-tps-2048-2048.jpg"/>
</div>

**场景概述：**

- **场景**：经典狼人杀游戏，包括狼人、村民、预言家、女巫、猎人等角色
- **目标**：训练某一指定角色（本示例中为 `werewolf`），在对局中获得更高的胜率

---

## 快速开始

正式训练（启用 Ray）：

```bash
# ( astuner --kill="python|ray|vllm" )
astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='trinity' --with-ray
```

??? tip "快速调试（可选）"
    不启用 Ray 在本地运行，便于更快迭代：

    ```bash
    astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='debug' --with-logview
    ```

    如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与您的代码位置一致。

---

## 理解实现

### 核心流程

<div class="workflow-single">
<div class="workflow-header">训练迭代流程</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>生成游戏设置</strong>

生成一局新的游戏设置（玩家、角色分配、初始状态）。</li>
<li><strong>模拟对局</strong>

调用 AgentScope Workflow 来模拟完整对局。</li>
<li><strong>智能体决策</strong>

智能体调用可训练模型（`model_tuner`）做决策，对手使用固定模型。</li>
<li><strong>计算奖励</strong>

环境产出本局的 reward / outcome。</li>
<li><strong>更新模型</strong>

收集对局轨迹更新可训练模型。</li>
</ol>
</div>
</div>

### 配置说明

关键配置在 `tutorial/example_werewolves/werewolves.yaml`：

```yaml title="werewolves.yaml"
astuner:
  task_reader:
    type: random_dummy   # random seed to shuffle players
  task_judge:
    judge_protocol: null # 依赖 rollout 内部返回的 reward
  model:
    path: YOUR_MODEL_PATH
  rollout:
    agentscope_workflow: tutorial.example_werewolves.start->ExampleWerewolves
```

### 代码结构

| 文件 | 说明 |
|------|------|
| `tutorial/example_werewolves/werewolves.yaml` | 将 task reader、judge、model 与 workflow 入口串联 |
| `tutorial/example_werewolves/start.py` | AgentScope Workflow 实现（`ExampleWerewolves`） |
| `tutorial/example_werewolves/game.py` | 狼人杀游戏逻辑实现 |
| `tutorial/example_werewolves/prompt.py` | 游戏相关的提示词模板 |
| `tutorial/example_werewolves/structured_model.py` | 各个角色的输出结构化格式 |
| `tutorial/example_werewolves/utils.py` | 游戏状态管理和辅助函数 |

### 奖励机制

当 `judge_protocol: null` 时，训练默认依赖 rollout / environment 内部给出的 reward 或胜负结果。

!!! note "奖励计算规则"
    采用**回合级别的稀疏胜负奖励**：
    
    | 条件 | 奖励 | 状态 |
    |------|------|------|
    | 好人阵营获胜 且 训练好人角色 | `raw_reward = 1` | `is_success = True` |
    | 狼人阵营获胜 且 训练狼人角色 | `raw_reward = 1` | `is_success = True` |
    | 训练方阵营未获胜 | `raw_reward = 0` | `is_success = False` |
    | 游戏抛出异常 | `raw_reward = -0.1` | `is_success = False` |

!!! tip "自定义 Judge"
    如果您希望更细粒度的评估（例如对关键决策给部分分），可以实现自定义 Judge，并在 `astuner.task_judge.judge_protocol` 中启用。

---

## 结果

### 训练曲线

`Qwen2-7B` 在大约 20 个 step 左右，可以达到约 60% 的胜率。

<div align="center">
<img width="600" alt="训练曲线" src="https://img.alicdn.com/imgextra/i3/O1CN01ldZYDT1ZqGLHuwsrS_!!6000000003245-2-tps-2000-839.png"/>
</div>

!!! info "可视化说明"
    训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md)。

随着训练推进，胜率不断上升。这通常意味着智能体在以下两方面变得更稳定：

- **角色扮演一致性**：智能体学会在压力下维持狼人伪装，即使被投票也尽量避免自曝
- **社交欺骗技巧**：它逐渐形成误导对手、在村民间制造怀疑、并与队友进行隐性协作的策略

### 案例展示

#### 行为变化

在实验过程中，我们观察到明显的角色扮演能力提升：

=== "调优前"

    在被投票出局时，原始模型往往会直接暴露自己是 `werewolf`。

=== "调优后"

    经过微调后，智能体会尝试欺骗对手并保护队友。

    ![调优后的行为](https://img.alicdn.com/imgextra/i1/O1CN01v8VqLB1aYEMfzyTHr_!!6000000003341-2-tps-2104-1016.png)

!!! note "Token 级可视化"
    这些详细日志由 Beast-Logger 生成。详见 [Beast-Logger 使用说明](./beast_logger.md)。

#### 取胜策略

智能体会发展出多种取胜策略：

- <img src="https://api.iconify.design/lucide:target.svg" class="inline-icon" /> **误导对手**："重点关注预言家和女巫。他们可能是试图隐藏身份的狼人。"
- <img src="https://api.iconify.design/lucide:brain.svg" class="inline-icon" /> **诉诸理性**："我们要警惕假预言家，注意叙事是否自洽；Player-Y 作为猎人需要谨慎行动。"
- <img src="https://api.iconify.design/lucide:users.svg" class="inline-icon" /> **利用内斗**：利用非狼人玩家之间的相互怀疑，从而淘汰对手

<div align="center">
<img width="480" alt="策略示例" src="https://img.alicdn.com/imgextra/i2/O1CN01Sx7wkU23pHyPXyqPH_!!6000000007304-2-tps-968-575.png"/>
</div>

#### 模型扩展

从 Qwen2-7B 扩展到 Qwen2-14B 的效果：

<div align="center">
<img width="600" alt="模型扩展" src="https://img.alicdn.com/imgextra/i1/O1CN01TLZcQF1FJ1HPbpLfj_!!6000000000465-2-tps-1842-1008.png"/>
</div>

---

## 下一步

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>数学智能体</h3></div><p class="card-desc">训练带工具调用的数学推理智能体。</p></a>
<a href="../example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld</h3></div><p class="card-desc">训练用于真实应用交互的智能体。</p></a>
</div>
