# Countdown 任务

## 1. 介绍

Countdown 任务是一个数学益智游戏：给定一组数字和一个目标数字，玩家需要利用这组数字，通过加、减、乘、除四则运算，构造一个算式，使其计算结果等于目标数字。每个数字只能使用一次，但可以自由使用括号改变运算顺序。

## 2. 快速开始

### 2.1 准备数据集
下载 `Jiayi-Pan/Countdown-Tasks-3to4` 数据集，并划分训练、测试集：

```bash
python tutorial/example_countdown/prepare_data.py --target=Jiayi-Pan/Countdown-Tasks-3to4 --path=/the/path/to/store/dataset
```

Countdown 数据集包含 `target` 和 `nums` 两个字段，需要自定义数据格式化逻辑。例如：使用 `huggingface_dat_repo` 的读取方式时，需要修改 `astuner/task_reader/hf_dataset_reader.py` 中的 `_load_dataset_split` 方法：

```python
task = Task(
    main_query=json.dumps({'target': example["target"], 'nums': example["nums"]}),
    init_messages=[],
    task_id=str(idx),
    env_type="no_env",
    metadata=example,
)
```

### 2.2 启动训练

直接运行以下命令：

```bash
# 建议在启动前先杀掉所有 ray、vllm 和 env_service 相关进程（ python launcher.py --kill="python|ray|vllm" ）
astuner --conf tutorial/example_countdown/countdown.yaml --backbone='verl'
```

## 3. 准备

本节将介绍本教程的实现细节。

### 3.1 准备 AgentScope Workflow
详见 `tutorial/example_countdown/countdown.py`。你可以在项目中的任意位置编写新的 AgentScope Workflow 代码。

- **定义 AgentScope Workflow**

```python
self.agent = ReActAgent(
    name="countdown_react_agent",
    sys_prompt=system_prompt,
    model=model_tuner,
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    max_iters=2,
)
msg = Msg("user", query, role="user")
result = await self.agent.reply(msg)
```

在 AgentScope Workflow 中，你需要将评估函数所需的关键信息写入，例如：

```python
WorkflowOutput(
    reward=None,
    metadata={
        "final_answer": final_answer,
        "target": target,
        "nums": nums,
    }
)
```

### 3.2 准备 Reward
在 `astuner/task_judge/countdown_answer_as_judge.py.py` 中提供了一个简单的 Judge 示例。你也可以在项目任意位置实现自己的 Judge 逻辑。

Judge 的输入参数包括：

```
workflow_task: 任务信息（如果包含参考答案，可以从这里获取）
workflow_output: 任务信息输出（final_answer需要手动添加）
```

Judge 的返回值包括：

- raw_reward
- is_success

### 3.3 启动训练

#### 3.1 配置
拷贝并修改 `tutorial/example_countdown/countdown.yaml` 中的关键配置参数。yaml 中与本示例最相关的部分已经用 ✨✨✨✨ 标出。

1. 读取任务（对应配置字段 `astuner.task_reader`）
2. 定义 Workflow（对应配置字段 `astuner.rollout.agentscope_workflow`）
   - 示例：如果 AgentScope Workflow 定义在 `tutorial/example_countdown/countdown.py` 的 `ExampleCountdownLearn` 类中
   - 则配置 `astuner.rollout.agentscope_workflow`=`tutorial.example_countdown.countdown->ExampleCountdownLearn`
3. 定义评分函数（对应配置字段 `astuner.task_judge.judge_protocol`）
   - 示例：如果评分逻辑定义在 `astuner/task_judge/countdown_answer_as_judge.py` 的 `CountdownAnswerAsJudge` 类中
   - 则配置 `astuner.task_judge.judge_protocol`=`astuner.task_judge.countdown_answer_as_judge->CountdownAnswerAsJudge`
4. 指定模型（对应配置字段 `astuner.model.path`）

```yaml
astuner:
    task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` 或 `dataset_file` 或 `huggingface_dat_repo` 或 `data_generation`
    rollout:
        agentscope_workflow: tutorial.example_countdown.countdown->ExampleCountdownLearn # ✨✨✨✨ 编写并选择 Agent
    task_judge:
        # ✨✨✨✨ 编写并选择评估函数
        judge_protocol: astuner.task_judge.countdown_answer_as_judge->CountdownAnswerAsJudge
    model:
        # ✨✨✨✨ 设置需要训练的模型
        path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct
```

#### 3.2 启动训练

```bash
# 建议在启动前先杀掉所有 ray、vllm 和 env_service 相关进程（ python launcher.py --kill="python|ray|vllm" ）
astuner --conf tutorial/example_countdown/countdown.yaml --backbone='verl'
```

## 4. 参考结果

### 4.1 训练曲线

![Tracing curve](https://img.alicdn.com/imgextra/i4/O1CN01TtaeD91rnfBF736Zu_!!6000000005676-2-tps-1328-630.png)

### 4.2 案例分析

在训练初期，Agent 已经能够处理一些简单问题；但由于小模型的能力限制，仍然会产生许多无法完全满足要求的答案。在一些题目中，Agent 没有严格遵循指令，例如输出格式不正确、重复使用数字、或缺少表达式输出。

```
bad case 1: 回答格式不符合要求。
<|im_start|>assistant
<tool_call>
{"name": "generate_response", "arguments": "Let's start by subtracting 28 from 76, which gives us 48. Now we need to get to 55 from 48. We can achieve this by adding 7 from the remaining numbers. However, 7 is not available. The next best number is 69. If we subtract 69 from 78, we get -11. If we subtract -11 from 48, we actually end up adding 11, which isn't one of our numbers. The correct approach would be to add 7 from the remaining numbers, but since 7 isn't available, we can use 69 and adjust our strategy. We can add 7 from the remaining numbers in another way. Let's try subtracting 69 from 78 to get -11, and then add 11 to 48. 11 can be obtained by subtracting 69 from 80, but since 80 isn't available, we'll use the next best approach. We'll subtract 69 from 78 to get -11, and then add 16 to 48 to reach 55. We can get 16 by subtracting 7 from 23, but since 23 isn't available, we need to find an alternative. The best way would be to use 78 - 69 = -11, and then 48 + 11 = 59 which is close but not 55. However, we can achieve 55 by using 78 - 69 + 7 = 55. So, the correct calculation is 78 - 69 + 7 = 55."}
</tool_call><|im_end|>

bad case 2: 没有遵循指令，重复使用数字。
<|im_start|>assistant
 NavParams = {"response": "Let's solve this step by step. We have the numbers 72, 20, and 4, and we need to reach the target number 38 using arithmetic operations.\n\nFirst, let's try using subtraction and addition since 72 is quite large and can help us get closer to our target if we subtract something. We can subtract 20 from 72 to get 52, which is close but still far from 38. Now, we can subtract 4 from 52 to get 48, and then subtract 10 to reach 38. Since we don't have a 10 directly, we can use 20 - 4 = 16, and then subtract 16 from 48 to get 32, and finally add 6 (which is 4 + 2) to reach 38.\n\nLet's try a more efficient approach: 72 - 20 = 52, and 52 - 4 = 48, now we need to reduce it by 10, which is 20 - 14, and 14 can be obtained by 20 - 4 - 2. So, 48 - (20 - 4 - 2) = 48 - 14 = 34, then we need 4 more to reach 38, which is simply 4. Thus, the solution is (72 - 20) - 4 - (20 - 4 - 2) + 4 = 38."}
<tool_call>
{"name": "generate_response", "arguments": {"response": "The solution is \boxed{(72 - 20) - 4 - (20 - 4 - 2) + 4}. Let's break it down: \n1. 72 - 20 = 52\n2. 52 - 4 = 48\n3. 20 - 4 - 2 = 14\n4. 48 - 14 = 34\n5. 34 + 4 = 38\nTherefore, we reach the target number 38."}}
</tool_call><|im_end|>

bad case 3: 没有正确给出表达式。
<|im_start|>assistant
<tool_call>
{"name": "generate_response", "arguments": {"response": "In this case, one straightforward way to reach the target number 64 is by directly using the number 64 itself, since it matches the target exactly. No other calculations are needed."}}
boxing the answer as \boxed{{64}}.<|im_end|>
```

不过，经过调优后，这些问题会得到明显改善，例如下图所示：


![After tuning](https://img.alicdn.com/imgextra/i1/O1CN011HjAxo20DKIcPAPVr_!!6000000006815-2-tps-1658-506.png)
![After tuning](https://img.alicdn.com/imgextra/i4/O1CN01C3kUnV221zjPi30rd_!!6000000007061-2-tps-1650-730.png)
