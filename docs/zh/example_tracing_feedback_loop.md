# 数据回流训练

ASTune 允许你回收 Agent 在执行过程中产生的对话日志，并通过迭代训练持续优化该 Agent，我们将这一过程称为 **数据回流训练**。它主要提供以下能力：

+ 从 agentscope studio 数据库中加载追踪日志
+ 将日志转换为结构化的训练数据
+ 结合自定义过滤器筛选高质量样本
+ 将样本打包成数据集，用于迭代训练


在下面的小节中，我们将演示如何利用数据回流训练来改进一个 Agent。

> **AgentScope 与 Studio 版本兼容性**
>
> 推荐使用匹配的版本组合：
>
> + AgentScope (v1.0.7)
> + Studio (23eb7c0b1185486d1baca36aea0ce8b85ea9de48)
>

## 环境准备

要使用追踪日志进行训练，通常需要你已经有一个基于 **agentscope** 编写的 Agent，并在 **agentscope-studio** 中运行了一段时间（通常是部署运行），也就是说你已经：

1. 使用 [agentscope](https://github.com/agentscope-ai/agentscope) 编写好了你的 Agent；
2. 按照 [文档](https://doc.agentscope.io/tutorial/task_tracing.html) 启用了 tracing 模块；
3. 部署了 Agent 并收集到了数据库文件。

默认情况下，agentscope-studio 会将追踪日志存储在
`~/AgentScope-Studio/database.sqlite` 中，其中包含了用户与 Agent 之间的全部对话记录。


我们在 `tutorials/example_feedback_tracing/agent_deployed.py` 中准备了一个示例 Agent。你可以通过它模拟生成追踪日志，并得到对应的数据库文件。

## 开始数据回流训练

当我们拿到日志文件（`database.sqlite`）后，就可以基于数据回流训练出一个新的 Agent。

1. 在配置文件中将参数 `astuner.task_reader.type` 设置为 `tracing`，以开启数据回流模式；
2. 在 `astuner.task_reader.feedback_tracing` 字段中配置数据库路径和过滤相关选项；
3. 像普通训练流程一样配置其他训练参数以及 Reward（奖励信号）。

```yaml
astuner:
  # ...
  task_reader:
    # 使用 tracing 日志作为任务来源
    type: tracing
    feedback_tracing:
      # 数据库路径
      base_url: ./tutorial/example_feedback_tracing/database.sqlite
      # 模块写入缓存的路径
      train_output_path: ./tutorial/example_feedback_tracing/tasks.jsonl
      # 过滤阶段使用的模型
      alien_llm_model: qwen3-235b-a22b-instruct-2507
      alien_llm_response_length: 2048
      # 过滤器定义
      filters:
        # 默认过滤器 llm_evaluate
        - type: llm_evaluate
          enabled: true
          params:
            # 编写 rubric，用于丢弃低质量任务
            custom_rubrics: |
              1. Check the answer and drop the task if it does not answer or answer is wrong.
              2. Consider a response is invalid if it does not wrap the final answer in \boxed{}.
            # LLM temperature
            temperature: 0.5
            # 是否打印调试日志
            print_reason: false
            max_thread: 16
```

当一切准备就绪后，可以通过 `launcher.py` 启动训练：

```bash
# 启动示例训练
python launcher.py --conf tutorial/example_feedback_tracing/example_feedback_tracing.yaml --backbone='trinity' --with-ray
```

训练完成后，你可以将新的 Agent 部署回生产环境，并继续收集新的日志。通过这样的闭环，你可以持续进行迭代的数据回流训练，不断提升 Agent 的效果。

## 自定义

### 过滤器

模块提供了 Filter，用于从日志中筛选出高质量样本用于训练。用户可以根据自己的任务需求，自定义具体的筛选规则。

要编写规则，只需要修改配置文件中对应的字段：

```yaml
astuner:
  # ...
  task_reader:
    # ...
    feedback_tracing:
      # ...
      filters:
        - type: llm_evaluate
          enabled: true # 启用该过滤器
          params:
            # 定义你的规则
            custom_rubrics: |
              1. 检查回答内容，如果没有回答问题或回答错误，则丢弃该任务。
              2. 如果回答没有使用 \boxed{} 包裹最终答案，则认为该回答无效。
            temperature: 0.5
            print_reason: false
            max_thread: 16
```
