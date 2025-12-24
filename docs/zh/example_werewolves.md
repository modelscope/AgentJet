# 狼人杀

本教程展示了如何使用 AgentScope Tuner 来处理多 智能体 训练，令多个 智能体 在狼人杀游戏中决策、对抗和协作。

## 1. 概述

狼人杀角色扮演游戏是一个典型的 POMDP（部分可观测马尔可夫决策过程，Partially Observable Markov Decision Process）问题。我们可以在这个协作型多 智能体 问题中使用「参数共享」的方法来训练 Agent。

术语说明：
- **部分可观测（Partially Observable）**：Agent 只能接收**局部信息**。即使属于同一阵营，一个 智能体 也无法获得其他 智能体 的观测信息。
- **马尔可夫决策过程（Markov Decision Process）**：根据当前局面做决策。
- **参数共享（Shared-parameter）**：多个 智能体 使用同一个模型作为策略。但需要注意：Agent **共享**策略（模型参数），但**不共享**感知（模型输入）。
- **协作型多 智能体 问题（Cooperative multi-agent problem）**：Agent 之间目标一致（奖励一致）。
- **环境（Environment）**：使用静态 **`Qwen3-235B-A22B`** 作为对手（不可训练 Agent），使用 **`Qwen2-7B`** 作为可训练 Agent（即 `trainable_targets`）。

![image](https://img.alicdn.com/imgextra/i2/O1CN012JgVZC2ABczBhAzJs_!!6000000008165-0-tps-2048-2048.jpg)

本页展示如何将狼人杀这种社交推理类游戏作为多 Agent 环境，完成：准备数据与环境、编写 AgentScope Workflow、配置奖励模块，以及从本地调试到正式训练的完整流程。

场景概述
- **场景**：经典狼人杀游戏，包括狼人（werewolf）、村民（villager）、预言家（seer）、女巫（witch）、猎人（hunter）等角色。
- **目标**：训练某一指定角色（本示例中为 `werewolf`），在对局中获得更高的胜率。

## 2. 快速开始

正式训练（启用 Ray）：
```bash
# ( astuner --kill="python|ray|vllm" )
astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='trinity' --with-ray
```

<details>
<summary>快速调试（可选）</summary>

不启用 Ray 在本地运行，便于更快迭代：

```bash
astuner --conf tutorial/example_werewolves/learn2ask.yaml --backbone='debug' --with-logview
```

如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与你的代码位置一致。

</details>

## 3. 理解实现

### 3.1 核心流程

从训练迭代视角来看，整体流程可以概括为：
- 生成一局新的游戏设置（玩家、角色分配、初始状态）。
- 调用 AgentScope Workflow 来模拟完整对局。
- 智能体 调用可训练模型（`model_tuner`）做决策，对手使用固定模型。
- 环境产出本局的 reward / outcome。
- 收集对局轨迹更新可训练模型。

### 3.2 配置说明

本小节对应 `tutorial/example_werewolves/werewolves.yaml`，关键配置项如下：

```yaml
astuner:
  task_reader:
    # random seed to shuffle players
    type: random_dummy
  task_judge:
    # 编写并选择评估函数
    # （在本示例中，你可以先将其设为 null，仅依赖 rollout 内部返回的 reward）
    judge_protocol: null
  model:
    # 设置需要训练的模型
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # 选择 AgentScope Workflow 的入口
    agentscope_workflow: tutorial.example_werewolves.start->ExampleWerewolves
```

### 3.3 代码解读

- `tutorial/example_werewolves/werewolves.yaml`：将 task reader、judge、model 与 workflow 入口串联起来。
- `tutorial/example_werewolves/start.py`：AgentScope Workflow 实现（`ExampleWerewolves`）。
- `tutorial/example_werewolves/game.py`：狼人杀游戏逻辑实现。
- `tutorial/example_werewolves/prompt.py`：游戏相关的提示词模板。
- `tutorial/example_werewolves/structured_model.py`：定义了各个角色的输出结构化格式。
- `tutorial/example_werewolves/utils.py`：包含游戏状态管理和辅助函数。

### 3.4 奖励

当 `judge_protocol: null` 时，训练默认依赖 rollout / environment 内部给出的 reward 或胜负结果。在本示例中，reward 在 `tutorial/example_werewolves/start.py` 的 workflow 中给出。

在 `ExampleWerewolves.execute()` 中，workflow 会先运行一整局游戏：调用 `werewolves_game(players, roles)`，并得到 `good_guy_win`（好人阵营是否获胜）。

随后 reward 采用**回合级别的稀疏胜负奖励**：
- 若 `good_guy_win == True` 且训练目标不是 `werewolf`（即训练好人阵营角色），则 `raw_reward = 1`，并设置 `is_success = True`。
- 若 `good_guy_win == False` 且训练目标是 `werewolf`（即训练狼人阵营角色），则 `raw_reward = 1`，并设置 `is_success = True`。
- 其他情况表示训练方阵营未获胜，`raw_reward = 0`，`is_success = False`。

异常/违规惩罚：
- 若游戏过程中抛出异常（对局无法继续等），会统一对可训练目标进行惩罚：`raw_reward = -0.1`，`is_success = False`。

如果你希望更细粒度的评估（例如对关键决策给部分分、而不仅仅是 win/loss），可以实现自定义 Judge，并在 `astuner.task_judge.judge_protocol` 中启用。

## 4. 结果

### 4.1 训练曲线

`Qwen2-7B` 在大约 20 个 step 左右，可以达到约 60% 的胜率。

![image](https://img.alicdn.com/imgextra/i3/O1CN01ldZYDT1ZqGLHuwsrS_!!6000000003245-2-tps-2000-839.png)

> **可视化说明：** 训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md).

随着训练推进，胜率不断上升。这通常意味着 智能体 在以下两方面变得更稳定：

- **角色扮演一致性**： 智能体 学会在压力下维持狼人伪装，即使被投票也尽量避免自曝。
- **社交欺骗技巧**：它逐渐形成误导对手、在村民间制造怀疑、并与队友进行隐性协作的策略。

### 4.2 案例展示

#### 行为变化

在实验过程中，我们观察到明显的角色扮演能力提升：

1. 例如，在被投票出局时，原始模型往往会直接暴露自己是 `werewolf`；而经过微调后， 智能体 会尝试欺骗对手并保护队友。例如：

![](https://img.alicdn.com/imgextra/i1/O1CN01v8VqLB1aYEMfzyTHr_!!6000000003341-2-tps-2104-1016.png)

> **Token级可视化：** 这些详细日志由 Beast-Logger 生成。详见 [Beast-Logger 使用说明](./beast_logger.md).

2. 智能体 会发展出多种取胜策略。例如：
- **误导对手**："重点关注预言家和女巫。他们可能是试图隐藏身份的狼人。"
- **诉诸理性**："我们要警惕假预言家，注意叙事是否自洽；Player-Y 作为猎人需要谨慎行动。"

3. 有时 智能体 还能利用非狼人玩家之间的相互怀疑，从而淘汰对手。

![](https://img.alicdn.com/imgextra/i2/O1CN01Sx7wkU23pHyPXyqPH_!!6000000007304-2-tps-968-575.png)

#### 从 Qwen2-7B 扩展到 Qwen2-14B

![](https://img.alicdn.com/imgextra/i1/O1CN01TLZcQF1FJ1HPbpLfj_!!6000000000465-2-tps-1842-1008.png)
