# 快速开始

AgentScope Tuner 提供了一套完整的智能体调优功能。你可以立刻尝试启动一个智能体的训练：

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

### 最小示例

我们从最简单的示例开始：一个带工具调用的数学智能体。

* 首先，请查看 [installation guide](docs/en/installation.md) 来搭建训练环境。
* 然后，使用下面的最小示例来调优你的第一个模型（假设你已经编写了一个名为 `MathToolWorkflow` 的 Agent）。

  ```python
  from astuner import AstunerJob
  from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow
  model_path = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct"
  job = AstunerJob(n_gpu=8, algorithm='grpo', model=model_path)
  job.set_workflow(MathToolWorkflow)
  job.set_data(type="hf", dataset_path='openai/gsm8k')
  # [可选：保存 yaml 文件以便手动调整]  job.dump_job_as_yaml('saved_experiments/math.yaml')
  # [可选：从手动调整后的 yaml 文件加载] job.load_job_from_yaml('saved_experiments/math.yaml')
  tuned_model = job.tune()  # 等价于在终端执行 `astuner --conf ./saved_experiments/math.yaml`
  ```

### 浏览示例

浏览我们丰富的示例库，为你的旅程快速起步：

* 🔢 [**训练一个能写 python 代码的数学智能体**](./example_math_agent.md)。
* 📱 [**使用 AgentScope 创建 AppWorld 智能体并训练它**](./example_app_world.md)。
* 🐺 [**开发狼人杀 RPG 智能体并训练它们**](./example_werewolves.md)。
* 👩🏻‍⚕️ [**学习像医生一样提问**](./example_learning_to_ask.md)。
* 🎴 [**使用 AgentScope 编写倒计时游戏并求解**](./example_countdown.md)。
* 🚶 [**使用 ASTuner 解决 Frozen Lake 行走谜题**](./example_frozenlake.md)。

### 从零开始调优你的第一个智能体

开始构建你自己的智能体，并按照我们的文档进行调优：

* 📚 [**调优你的第一个智能体**](./tune_your_first_agent.md)。
