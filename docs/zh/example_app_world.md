# AppWorld 智能体

本教程介绍如何训练一个智能体与 AppWorld 交互并解决复杂的任务。

---

## 概述

<div class="callout-tip">
<p>
AppWorld 是一个模拟现实 APP 操作的沙盒环境，包含 9 个日常应用，可通过 457 个 API 操作，并预置了 106 个在模拟世界中生活的数字用户行为数据。我们的目标是调优一个智能体，使其能够有效地在这些应用中执行并完成复杂任务。
</p>
</div>

本文结构如下：

1. 快速开始
2. 理解实现：Workflow 核心流程、配置、代码位置、奖励机制
3. 结果：训练曲线与案例对比

---

## 快速开始

### 准备工作

首先，需要准备 AppWorld 所需的环境服务：

```bash
base_path="/tmp"
export APPWORLD_PATH="${base_path}/pack_all_in_one"
export APPWORLD_SCRIPT="bash EnvService/env_sandbox/appworld.sh"

rm -rf "${APPWORLD_PATH}"
rm -f ./appworld_pack_v2.tar.gz

wget -q "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v2.tar.gz" -O appworld_pack_v2.tar.gz
tar -xzf ./appworld_pack_v2.tar.gz -C "${base_path}"
```

!!! warning "环境变量设置"
    每次开启新的 shell 窗口都需要运行以下环境变量设置：
    
    ```bash
    export BASE_PATH=/tmp
    export APPWORLD_PATH="${BASE_PATH}/pack_all_in_one"
    export APPWORLD_SCRIPT="bash EnvService/env_sandbox/appworld.sh"
    ```

### 开始训练

运行训练脚本：

```bash
astuner --conf tutorial/example_appworld/appworld.yaml --backbone='trinity' --with-ray
```

??? tip "快速调试（可选）"
    不启用 Ray 在本地运行，便于更快迭代：

    ```bash
    astuner --conf tutorial/example_appworld/appworld.yaml --backbone='debug' --with-logview
    ```

    如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与您的代码位置一致。

---

## 理解实现

### 核心流程

AppWorld 示例所使用的 AgentScope Workflow 代码位于：`tutorial/example_appworld/appworld.py`。

```python title="Workflow 核心代码"
agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg["content"],
    model=model_tuner,
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

env = workflow_task.gym_env
for step in range(model_tuner.config.astuner.rollout.multi_turn.max_steps):
    # agentscope 处理交互消息
    reply_message = await agent(interaction_message)
    # env_service 协议
    obs, _, terminate, _ = env.step(
        action={"content": reply_message.content, "role": "assistant"}
    )
    # 使用环境的输出构造新的交互消息
    interaction_message = Msg(name="env", content=obs, role="user")
    # 是否终止？
    if terminate:
        break
    if model_tuner.get_context_tracker().context_overflow:
        break
```

| 关键点 | 说明 |
|--------|------|
| `env.step` | 模拟 gym 接口。输入一个 action，返回 `(observation, reward, terminate_flag, info)` |
| `context_overflow` | 检查当前上下文窗口是否已经超过 token 限制 |

### 奖励机制

在 `astuner/task_judge/env_service_as_judge.py` 中，通过 `env.evaluate(...)` 从环境中读取奖励信号。

!!! tip "自定义 Judge"
    您可以参考该文件，为自己的任务实现专用的 Judge 模块。

### 配置说明

`tutorial/example_appworld/appworld.yaml` 中的关键配置参数：

```yaml title="appworld.yaml"
astuner:
  project_name: example_appworld
  experiment_name: "read_yaml_name"
  task_judge:
    # [关键] 编写并选择评估函数
    judge_protocol: agentscope_tuner.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # [关键] 设置需要训练的模型
    path: YOUR_MODEL_PATH
  rollout:
    # [关键] 编写并选择智能体
    agentscope_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
    agentscope_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```

| 配置项 | 说明 |
|--------|------|
| `task_reader` | 读取任务 |
| `agentscope_workflow` | 定义 Workflow |
| `judge_protocol` | 定义评分函数 |
| `model.path` | 指定模型 |

---

## 结果

### 训练曲线

<div align="center">
<img width="600" alt="训练曲线" src="https://img.alicdn.com/imgextra/i2/O1CN01toRt2c1Nj8nKDqoTd_!!6000000001605-2-tps-1410-506.png"/>
</div>

!!! info "可视化说明"
    训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md)。

随着训练的进展，奖励也会增加。这通常意味着智能体在**两个方面**变得更加稳定：

- <img src="https://api.iconify.design/lucide:check-circle.svg" class="inline-icon" /> **遵循正确的 API 协议**：学会在调用前查阅 API 文档，并使用有效的 API 端点
- <img src="https://api.iconify.design/lucide:workflow.svg" class="inline-icon" /> **完成多步工作流**：能够正确获取 access token，并串联多个 API 调用以完成复杂任务

### 案例展示

=== "调优前"

    **问题 1：频繁调用不存在的 API**
    
    ![调优前 - 幻觉 API](https://img.alicdn.com/imgextra/i1/O1CN015FgjqI20Ip3AJybr0_!!6000000006827-2-tps-1259-683.png)
    
    智能体在不检查 API 是否存在的情况下产生幻觉，导致重复失败。
    
    **问题 2：没有学会按照说明获取 access token**
    
    ![调优前 - Token 问题](https://img.alicdn.com/imgextra/i1/O1CN01bGZ1s01VyjCSrTJte_!!6000000002722-2-tps-1181-954.png)
    
    智能体在未先获取所需的访问令牌的情况下尝试调用受保护的 API，导致认证错误。

=== "调优后"

    **改进 1：会先查阅 API 文档，使用有效的 API**
    
    ![调优后 - 正确的 API](https://img.alicdn.com/imgextra/i4/O1CN01VRIDy922PoKD1bETl_!!6000000007113-2-tps-1180-944.png)
    
    智能体现在会先检查可用的 API 再发起调用，从而避免臆造不存在的接口端点。
    
    **改进 2：学会正确获取 access token**
    
    ![调优后 - Token 正确](https://img.alicdn.com/imgextra/i2/O1CN01xiF9UU20h62dyrZ4x_!!6000000006880-2-tps-1182-793.png)
    
    智能体在访问受保护的 API 之前，会先正确完成认证步骤。

!!! note "Token 级可视化"
    这些详细日志由 Beast-Logger 生成。详见 [Beast-Logger 使用说明](./beast_logger.md)。

---

## 下一步

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>数学智能体</h3></div><p class="card-desc">训练带工具调用的数学推理智能体。</p></a>
<a href="../example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>狼人杀游戏</h3></div><p class="card-desc">探索多智能体协作训练。</p></a>
</div>
