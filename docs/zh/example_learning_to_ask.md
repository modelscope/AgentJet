# 学习提问

训练一个智能体去**提出下一个最合适的问题**（而不是直接回答）。奖励来自一个 **LLM-as-a-judge**：它会评估“这个问题是否有帮助、是否相关”，并给出打分。

### 1. 概览

在 **Learning to Ask** 中，每条训练样本是一段简短的**医生–患者对话历史**。智能体的输出是医生接下来应该问的**一个**问题（可选地带有多项选择答案），而不是给出诊断或治疗建议。


```{figure} https://img.alicdn.com/imgextra/i4/O1CN01m9WJCM1WJL1aJCSaS_!!6000000002767-2-tps-1024-559.png
图：为什么“Learning to Ask”很重要。左：LLM 在信息不足的情况下直接给出诊断。右：LLM 先提出清晰的追问，再给出结论，这会让人更安心。
```

本教程分为两步：

1. **Run it**：使用默认的 YAML 配置启动训练。
2. **Understand & customize**：理解并自定义数据预处理、工作流（ExampleLearn2Ask）以及奖励（reward_fn + llm_reward）。

---

### 2. 快速开始

#### 2.1 准备工作

从 HuggingFace 下载 [RealMedConv](https://huggingface.co/datasets/datajuicer/RealMedConv) 数据集，并将文件放到：`data/realmedconv`

然后进行预处理：

```bash
export DASHSCOPE_API_KEY=your_api_key

cd tutorial/example_learn2ask/data_preprocess
./run_process.sh data/realmedconv
```

预处理完成后，你应该会得到：`train.jsonl` 和 `test.jsonl`。

#### 2.2 启动训练

```bash
ajet --conf tutorial/example_learn2ask/learn2ask.yaml --backbone='trinity' --with-ray
```

<details>
<summary>快速调试（可选）</summary>

不启用 Ray 在本地运行，便于更快迭代：

```bash
ajet --conf tutorial/example_learn2ask/learn2ask.yaml --backbone='debug' --with-logview
```

如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与你的代码位置一致。

</details>

---

### 3. 理解实现

#### 3.1 每个训练 step

本教程训练的目标是：基于一段简短的医生–患者对话历史，让模型学会**提出下一个最合适的问题**。具体来说，每个训练 step 会从 `train.jsonl` 中取出一条对话上下文，让智能体生成**恰好一个**追问（可选地带有答案选项），随后使用一个 LLM judge 来评估这个问题是否**有用**且**相关**。AgentJet 将该评分作为奖励信号更新策略，于是模型会逐渐学会提出更好的问题，而不是直接给出回答。

#### 3.2 YAML 配置说明

整个例子主要通过 YAML 完成配置信息，实现代码则集中在一个文件里。在 YAML 中，`task_reader` 提供数据集划分；`rollout.agentscope_workflow` 告诉 AgentJet 对每条样本需要运行哪个 workflow；`task_judge` 提供封装了 LLM judge 的奖励入口；`model` 部分决定训练从哪个预训练 backbone 开始。

```yaml
ajet:
  task_reader:
    type: dataset_file
    # train_path: data/realmedconv/train.jsonl
    # test_path:  data/realmedconv/test.jsonl

  rollout:
    # For each sample: conversation context -> one next question
    agentscope_workflow: tutorial.example_learn2ask.learn2ask->ExampleLearn2Ask

  task_judge:
    # Reward function used by the trainer (internally calls the LLM judge)
    # judge_protocol: tutorial.example_learn2ask.learn2ask->reward_fn

  model:
    # pretrained backbone to start from
    # path: /path/to/your/model
```



#### 3.3 代码解读

在代码层面，所有实现都在 `tutorial/example_learn2ask/learn2ask.py` 中：

* `ExampleLearn2Ask` 定义 workflow：如何将对话上下文转成智能体的 prompt/input，以及期望的输出格式是什么（一个追问，可选选项）。
* `reward_fn` 定义如何把 “judge 的反馈” 转成训练可用的标量奖励。

#### 3.4 奖励

`llm_reward` 是 `reward_fn` 内部调用的 LLM-as-a-judge，用来给模型的输出打分。评估遵循以下规则：

- **只评估医生的最后一句话**（doctor’s last message），不看更早的医生回复。
- 输出两个分数：**Format Score** + **Content Score**（分别打分，后续由 `reward_fn` 组合成训练用 reward）。

**Format Score（格式分）**：根据“最后一句话里问题的数量”计分
- 1.0：恰好 **1 个问题**，或者在判断对话结束时正确输出了 `<stop />`
- 0.5：包含 **2 个问题**
- 0.0：包含 **3 个及以上问题**

**Content Score（内容分）**：根据问题是否命中 `Reference Information` 中“医生尚未知晓的缺失信息”计分
- 1.0：问题**直接询问** `Reference Information` 里的某个缺失项，或者在信息足够时及时结束对话
- 0.1：问题过于泛化（对任何症状都适用的通用问题）
- 0.0：问题与 `Reference Information` 的缺失项**无关**
- 另外：**含糊/无信息量的问题按低质量处理**（例如指代不明的提问），通常会得到 0 或接近 0 的得分

---


### 4. 结果

#### 4.1 训练曲线

![curve](https://img.alicdn.com/imgextra/i2/O1CN01STeE8G1ricp6jySto_!!6000000005665-2-tps-866-614.png)

> **可视化说明：** 训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md)。

这张训练曲线反映的是reward 变化趋势。reward包括：Format Score（是否只问一个问题）和 Content Score（是否直接追问 Reference Information 中缺失的关键信息）。

因此，曲线升高表示模型逐渐学会：把输出稳定成“一个清晰问题”，并且更频繁地命中缺失信息，而不是泛泛而问或提出含糊、无信息量的问题。

#### 4.2 案例展示

**Before training**

```text
User: Bitten by a midge

Agent: Has irritation subsided past initial acute swelling phase afterwards unnoticed scratched overly causing secondary bacterial dermatitis chance affecting skin rendering needed topical antibiotic application options among azelaic acid cream!?
```

智能体给出了一个“看似有点相关、但非常不合理”的回答，并且没有形成任何有效的追问。

**During training (~50%)**

```text
User: Bitten by a midge

Agent: Have these itchy bumps appeared soon after getting bitten?(A.) Occasionally immediately afterward ; B fkknore ×later throughout day )/)
```

智能体开始提出更合理的追问，尝试收集症状信息，并提供选项。

**After training**

```text
User: Bitten by a midge

Agent: Has itching or reddening appeared around this bite site recently without healing despite scratching frequently?(A,Yes indeed worse every evening B,No pain but small swollen spots persist day after incident/)
```

问题变得更具体、更有信息量，能够引导用户补充临床相关的关键细节。

> 想进一步了解该任务以及在更大模型上的结果，可参考论文：[Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs](https://arxiv.org/abs/2510.25441)。
