# 任务评测器

Task Judger 用于在训练过程中评估智能体（agent）的输出，并据此分配奖励。本页面介绍了常见场景下的内置评测器，以及如何为特定评测需求实现自定义评测器。

---

## 概览

Task Judger 会评估智能体的执行结果，并返回两个值：

* **`raw_reward`** (`float`)：表示输出质量的数值分数（通常为 0.0 到 1.0）
* **`is_success`** (`bool`)：任务是否成功完成

这两个值会引导 RL 训练过程，帮助智能体学习哪些行为能带来更好的结果。

---

## 基础接口

所有 Task Judger 都继承自 `BaseJudge`，并实现 `compute_reward` 方法：

```python
from agentscope_tuner.task_judge.base_judge import BaseJudge
from agentscope_tuner.workflow import WorkflowOutput, WorkflowTask

class BaseJudge:
    def __init__(self, config):
        self.config = config

    def compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput
    ) -> tuple:
        """
        Args:
            workflow_task: 包含任务数据，包括 metadata 中的参考答案等信息
            workflow_output: 包含智能体输出，包括 metadata 中的生成答案等信息

        Returns:
            tuple: (raw_reward: float, is_success: bool)
        """
        raise NotImplementedError
```

---

## 内置 Task Judger

AgentScope Tuner 提供了 4 个内置评测器，用于覆盖常见的评测场景：

### 1. MathAnswerAsJudge

通过**精确字符串匹配**来评估数学答案，适用于答案以 LaTeX `\boxed{}` 形式输出的任务。

**适用场景：**

* 数学解题类任务
* 答案确定且唯一的任务
* 答案格式为 `\boxed{result}`

**配置方式：**

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge
```

**工作原理：**

1. 从智能体输出中提取 `\boxed{...}` 内的答案
2. 与 `workflow_task.task.metadata["answer"]` 中的参考答案进行比对
3. 正确返回 `(1.0, True)`，否则返回 `(0.0, False)`

**必需的 metadata：**

* `workflow_output.metadata["final_answer"]`：智能体答案（包含 `\boxed{}` 格式）
* `workflow_task.task.metadata["answer"]`：参考答案

---

### 2. CountdownAnswerAsJudge

评估数学等式，并支持对“格式正确但结果错误”的输出给予**部分奖励**。

**适用场景：**

* 数字谜题类任务（例如 Countdown game）
* 需要部分得分的任务
* 希望即使答案不对，也能奖励良好格式的场景

**配置方式：**

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: tutorial.example_countdown.countdown_answer_as_judge->CountdownAnswerAsJudge
```

**评分规则：**

* `0.0`：答案无效或缺失
* `0.1`：等式格式正确，但结果错误
* `1.0`：等式与结果都正确

**必需的 metadata：**

* `workflow_output.metadata["final_answer"]`：等式字符串（包含 `\boxed{}` 格式）
* `workflow_output.metadata["target"]`：目标数字
* `workflow_output.metadata["nums"]`：可用数字列表（构造等式所允许使用的数字）

---

### 3. EnvServiceJudge

将评测委托给外部环境服务，适用于复杂的交互式环境。

**适用场景：**

* 依赖外部模拟器的任务（例如 AppWorld）
* 基于状态的复杂评估
* 环境自身带有 evaluator 的交互式任务

**配置方式：**

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: agentscope_tuner.task_judge.env_service_as_judge->EnvServiceJudge
```

**工作原理：**

1. 调用 `workflow_task.gym_env.evaluate()` 从环境获取 score
2. 将 score 转换为归一化 reward：

   * 成功（score ≥ 1）：`1.0 + score * 0.5`
   * 失败（score < 1）：`0.0 + score * 0.5`

---

### 4. AutoGraderJudge（高级）

通过 LLM 自动从参考样本生成评测 Rubric，用于主观性较强的评测任务。

**适用场景：**

* 主观评测任务（例如写作质量、对话连贯性）
* 没有明确标准答案的任务
* 有标注训练样本，但需要更灵活的评测标准

**配置方式：**

```yaml
astuner:
  task_judge:
    judge_type: rubrics_auto_grader
    rubrics_auto_grader:
      # 用于 rubric 生成与评测的模型
      model_name: "qwen-plus"  # or "gpt-4", "claude-3-sonnet"
      api_key: "your-api-key"  # or set DASHSCOPE_API_KEY env var

      # 评测模式
      grader_mode: "pointwise"  # or "listwise" for ranking multiple outputs
      language: "en"  # or "zh"

      # pointwise 模式配置
      min_score: 0
      max_score: 10

      # 用于 rubric 生成的参考数据
      input_data_type: "dataset_file"
      dataset_file:
        training:
          file_path: "path/to/rubrics_train.jsonl"

      # 字段映射
      query_field: "main_query"      # query 对应字段名
      answer_field: "final_answer"   # workflow output 中的字段
      reference_field: "answer"      # task metadata 中的字段
```

**工作原理：**

1. **训练阶段**：基于带标注的样本，通过迭代式优化生成 rubrics
2. **评测阶段**：使用生成的 rubrics，通过 LLM 调用对新输出打分

**所需数据格式：**

**pointwise** 模式（对单个输出打分）：

```json
{
  "main_query": "Explain quantum entanglement",
  "metadata": {
    "answer": "Quantum entanglement is...",
    "score": 8
  }
}
```

**listwise** 模式（对多个候选输出排序）：

```json
{
  "main_query": "Write a poem about spring",
  "metadata": {
    "candidates": [
      {"answer": "Flowers bloom...", "rank": 1},
      {"answer": "Spring is here...", "rank": 2}
    ]
  }
}
```

---

## 创建自定义 Task Judger

当你有更专门的评测需求时，可以通过继承 `BaseJudge` 来实现自己的评测器。

### Step 1：实现评测器

创建一个新文件（例如 `tutorial/my_task/my_judge.py`）：

```python
import re
from agentscope_tuner.task_judge.base_judge import BaseJudge
from agentscope_tuner.workflow import WorkflowOutput, WorkflowTask

class MyCustomJudge(BaseJudge):
    def __init__(self, config):
        super().__init__(config)
        # 初始化所需资源（例如外部 API、模型等）
        self.threshold = 0.8

    def compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput
    ) -> tuple:
        # 从 workflow_output 中读取数据
        agent_answer = workflow_output.metadata.get("final_answer", "")

        # 从 workflow_task 中读取参考答案
        reference_answer = workflow_task.task.metadata.get("answer", "")

        # 自定义评测逻辑
        similarity = self._compute_similarity(agent_answer, reference_answer)

        # 基于阈值判断是否成功
        is_success = similarity >= self.threshold

        # 返回 (reward, success)
        return similarity, is_success

    def _compute_similarity(self, text1: str, text2: str) -> float:
        # 自定义相似度度量
        # 这里只是一个简单示例
        return len(set(text1.split()) & set(text2.split())) / max(
            len(text1.split()), len(text2.split()), 1
        )
```

### Step 2：配置评测器

在 YAML 配置中：

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: tutorial.my_task.my_judge->MyCustomJudge
```

### Step 3：向评测器传递数据

在 workflow 中，将评测器所需字段写入 `workflow_output.metadata`：

```python
class MyWorkflow(Workflow):
    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # 智能体逻辑
        final_answer = await self.agent.reply(msg)

        # 返回包含 metadata 的输出，供 judger 使用
        return WorkflowOutput(
            reward=None,  # 将由 judger 填充
            metadata={
                "final_answer": final_answer,
                # 添加其他你的 judger 需要的字段
            }
        )
```

---

## 配置速查

### 使用内置 Judger

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: agentscope_tuner.task_judge.<module>-><ClassName>
```

### 使用 Auto Grader

```yaml
astuner:
  task_judge:
    judge_type: rubrics_auto_grader
    rubrics_auto_grader:
      model_name: "qwen-plus"
      grader_mode: "pointwise"
      # ... 其他配置
```
