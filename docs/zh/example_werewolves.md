# 狼人杀

狼人杀角色扮演游戏是一个典型的 POMDP（部分可观测马尔可夫决策过程，Partially Observable Markov Decision Process）问题。我们可以在这个协作型多智能体问题中，使用「参数共享」的方法来训练多个 Agent。

术语说明：

- **部分可观测（Partially Observable）**：每个 Agent 只能看到自己的**局部信息**。即使是同一阵营的队友，也无法直接获得对方的观测信息。
- **马尔可夫决策过程（Markov Decision Process）**：Agent 根据当前状态（局面）来做决策。
- **参数共享（Shared-parameter）**：多个 Agent 共用同一个模型作为策略网络。但需要注意，Agent **共享**的是策略（模型参数），而**不共享**各自的观测（模型输入）。
- **协作型多智能体问题（Cooperative multi-agent problem）**：多个 Agent 之间目标一致（共享奖励）。
- **环境（Environment）**：本示例中，我们使用静态的 **`Qwen3-235B-A22B`** 作为对手（非可训练 Agent）的“脑”，并使用 **`Qwen2-7B`** 作为可训练 Agent（即 `trainable_targets`）。

![image](https://img.alicdn.com/imgextra/i2/O1CN012JgVZC2ABczBhAzJs_!!6000000008165-0-tps-2048-2048.jpg)

本页展示如何将狼人杀这种社交推理类游戏作为一个多智能体环境，完成：准备数据与环境、编写 AgentScope Workflow、配置奖励模块（Judge）、以及从本地调试到正式训练的完整流程。

## 1. 场景概述

- **场景**：经典狼人杀游戏，包括狼人（werewolf）、村民（villager）、预言家（seer）、女巫（witch）、猎人（hunter）等角色。
- **目标**：训练某一指定角色（例如本示例中的女巫 witch），在对局中获得更高的胜率。

下面将依次介绍如何准备 AgentScope Workflow、配置 YAML、调试并启动训练。

## 2. 准备 AgentScope Workflow

示例代码位于 `tutorial/example_werewolves/start.py`。

你也可以在项目中的任意位置编写自己的 AgentScope Workflow，只要在 YAML 中正确配置入口即可。

首先，定义 AgentScope Workflow：

```python
class ExampleWerewolves(AgentScopeLearnProtocol):

    async def execute(self, init_messages, astune_proxy: ModelTuner, config) -> WorkflowOutput:
        train_which_role = "werewolf"
        roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]

        # Set random seed for reproducibility
        workflow_task = astune_proxy.get_agentscope_input_dictionary()["workflow_task"]
        task_id = workflow_task.task.task_id

        np.random.seed(int(task_id))
        np.random.shuffle(roles)

        players = [
            get_official_agents(
                f"Player{x + 1}", roles[x], train_which_role, astune_proxy
            )
            for x in range(9)
        ]

        good_guy_win = await werewolves_game(players, roles)
        raw_reward = 1 if (good_guy_win and train_which_role != "werewolf") or (
            not good_guy_win and train_which_role == "werewolf"
        ) else 0

        astune_proxy.update_judge_input_dictionary(raw_reward=raw_reward)
        astune_proxy.update_judge_input_dictionary(is_success=(raw_reward == 1))
        return astune_proxy
```

这里通过 `roles` 列表和 `train_which_role` 指定当前要训练的是哪个角色，并在对局结束后，根据好人阵营是否获胜来计算 `raw_reward`，再通过 `astune_proxy.update_judge_input_dictionary` 将 reward 和是否成功（`is_success`）写入，供后续 Judge 或训练逻辑使用。

## 3. 配置

### 3.1 配置 YAML

本小节对应的配置文件为：`tutorial/example_werewolves/werewolves.yaml`。

你可以在该文件的基础上拷贝并修改关键参数。与本文示例最相关的部分在 YAML 中用 ✨✨✨✨ 标出。

关键配置项示例如下：

```yaml
astuner:
  task_reader:
    # random seed to shuffle players
    type: random_dummy
  task_judge:
    # ✨✨✨✨ 编写并选择评估函数
    # （在本示例中，你可以先将其设为 null，仅依赖 rollout 内部返回的 reward）
    judge_protocol: null
  model:
    # ✨✨✨✨ 设置需要训练的模型
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ 编写并选择 AgentScope Workflow 的入口
    agentscope_workflow: tutorial.example_werewolves.start->ExampleWerewolves
```

你可以在上述结构的基础上增加或替换自己的 Workflow / Judge / Model，只要确保路径和类名与实际代码保持一致即可。

### 3.2 调试

在进行正式训练之前，推荐先使用 `--backbone='debug'` 进行单机快速调试，此时不会启用 Ray：

```
# 建议在启动前先杀掉所有与 ray、env_service、vllm 相关的进程
# ( astuner --kill="python|ray|vllm" )
astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='debug' --with-logview
```

当 `--backbone=debug` 时，程序在本地以非 Ray 模式运行，便于进行断点调试。你可以在 VSCode 中配置 `launch.json`，例如：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Launch rollout",
      "type": "debugpy",
      "request": "launch",
      "program": "launcher.py",
      "console": "integratedTerminal",
      "args": [
        "--backbone", "debug",
        "--conf", "tutorial/example_werewolves/werewolves.yaml"
      ],
      "env": {}
    }
  ]
}
```

## 4. 启动训练

在完成 Debug 模式下的验证之后，只需要将 `backbone` 切换为 `trinity` 并开启 Ray，即可启动正式训练：

```
# 建议在启动前先杀掉所有与 ray、vllm、env_service 相关的进程
# ( astuner --kill="python|ray|vllm" )
astuner --conf tutorial/example_werewolves/werewolves.yaml --backbone='trinity' --with-ray
```

## 5 实验结果

`Qwen2-7B` 在大约 20 个 step 左右，就可以达到约 60% 的胜率。

![image](https://img.alicdn.com/imgextra/i3/O1CN01ldZYDT1ZqGLHuwsrS_!!6000000003245-2-tps-2000-839.png)

### 行为变化（Behavior Shifts）

在实验过程中，我们观察到明显的角色扮演能力提升：

1. 例如，在被投票出局时，原始模型往往会直接暴露自己是 `werewolf`；而经过微调之后，Agent 会尝试欺骗对手、保护队友。例如：

  ![](https://img.alicdn.com/imgextra/i1/O1CN01v8VqLB1aYEMfzyTHr_!!6000000003341-2-tps-2104-1016.png)


2. Agent 会发展出多种取胜策略。例如：

   - **误导对手**："Let's keep an eye on the seer and the witch. They could be werewolves trying to hide"。
   - **诉诸理性**："We need to be wary of fake seers and watch for inconsistencies in stories, Player-Y as hunter should act carefully"。

3. 有时 Agent 还能利用非狼人玩家之间的相互怀疑，从而淘汰对手。

  ![](https://img.alicdn.com/imgextra/i2/O1CN01Sx7wkU23pHyPXyqPH_!!6000000007304-2-tps-968-575.png)

### 从 Qwen2-7B 扩展到 Qwen2-14B

![](https://img.alicdn.com/imgextra/i1/O1CN01TLZcQF1FJ1HPbpLfj_!!6000000000465-2-tps-1842-1008.png)
