# 快速开始

AgentJet 提供了一套完整的智能体调优功能。您可以立刻尝试启动一个智能体的训练：

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

---

## 最小示例

我们从最简单的示例开始：一个带工具调用的数学智能体。

<div class="workflow-single">
<div class="workflow-header">快速上手流程</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>设置环境</strong>

查看 [安装指南](./installation.md) 来搭建训练环境。</li>
<li><strong>定义工作流</strong>

编写一个继承自 Workflow 基类的 Agent 类（例如 `MathToolWorkflow`）。</li>
<li><strong>配置并运行</strong>

使用 `AgentJetJob` API 配置并启动训练。</li>
</ol>
</div>
</div>

### 代码示例

```python title="train_math_agent.py"
from ajet import AgentJetJob
from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow

model_path = "YOUR_MODEL_PATH"
job = AgentJetJob(n_gpu=8, algorithm='grpo', model=model_path)
job.set_workflow(MathToolWorkflow)
job.set_data(type="hf", dataset_path='openai/gsm8k')

# [可选] 保存 yaml 文件以便手动调整
# job.dump_job_as_yaml('saved_experiments/math.yaml')

# [可选] 从手动调整后的 yaml 文件加载
# job.load_job_from_yaml('saved_experiments/math.yaml')

# 开始训练
tuned_model = job.tune()
```

!!! tip "命令行替代方案"
    上述代码等价于在终端执行：
    ```bash
    ajet --conf ./saved_experiments/math.yaml
    ```

---

## 浏览示例

探索我们丰富的示例库，为您的旅程快速起步：

<div class="card-grid">
<a href="./example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>数学智能体</h3></div><p class="card-desc">训练一个能写 Python 代码解决数学问题的智能体。</p></a>
<a href="./example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld 智能体</h3></div><p class="card-desc">使用 AgentScope 创建 AppWorld 智能体并训练它。</p></a>
<a href="./example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>狼人杀游戏</h3></div><p class="card-desc">开发狼人杀 RPG 智能体并训练它们。</p></a>
<a href="./example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>学会提问</h3></div><p class="card-desc">学习像医生一样提问。</p></a>
<a href="./example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>倒计时游戏</h3></div><p class="card-desc">使用 AgentScope 编写倒计时游戏并用 RL 求解。</p></a>
<a href="./example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>冰湖问题</h3></div><p class="card-desc">解决冰湖行走谜题。</p></a>
</div>

---

## 下一步

<div class="card-grid">
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-agent" alt=""><h3>调优你的第一个智能体</h3></div><p class="card-desc">从零开始构建您自己的智能体的完整详细指南。</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>数学智能体示例</h3></div><p class="card-desc">数学智能体训练示例的详细演练。</p></a>
</div>
