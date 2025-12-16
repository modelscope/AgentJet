
# 学会提问（Learning to Ask）

传统的 LLM 通常主要通过对给定提示或问题生成直接答案/续写文本来工作，而 **Learning to Ask** 任务的核心是训练一个 Agent 学会如何提出问题，以获取最有价值的信息，并最大化推进任务进展。

![](https://img.alicdn.com/imgextra/i4/O1CN01m9WJCM1WJL1aJCSaS_!!6000000002767-2-tps-1024-559.png)

本文档展示了如何准备数据、构建 Agent 与 Workflow、设置奖励，并最终训练一个 7B 规模的 Agent 来完成该任务。

## 1. 准备数据集

从 HuggingFace 下载 [RealMedConv](https://huggingface.co/datasets/datajuicer/RealMedConv) 数据集，并将文件放置到 `data/realmedconv`。

运行以下命令对数据集进行预处理：

```bash
export DASHSCOPE_API_KEY=your_api_key

python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/train_original.jsonl --output_file data/realmedconv/train_processed.jsonl
python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/train_processed.jsonl --output_file data/realmedconv/train.jsonl

python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/test_original.jsonl --output_file data/realmedconv/test_processed.jsonl
python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/test_processed.jsonl --output_file data/realmedconv/test.jsonl
```

你将得到两个数据集文件：
- `train.jsonl`：训练集切分
- `test.jsonl`：测试集切分

接下来，我们将准备一个 Workflow，以使用这些数据训练一个 Agent。

## 2. 准备 Workflow

Workflow 细节请参考 `tutorial/example_learn2ask/learn2ask.py`。

在该 Workflow 中，我们实现了：
- `ExampleLearn2Ask`：Workflow 与 Agent
- `reward_fn`：learn2ask 任务的奖励函数
- `llm_reward`：在 `reward_fn` 中使用的 llm-as-a-judge

Agent 会读取医患间的对话消息，并尝试预测下一句应该提出的问题，以收集信息并对患者进行诊断。所提出的问题会由一个 LLM 进行评判，以检查其是否有价值且相关。

## 3. 准备训练配置

接下来，我们需要一个 YAML 配置文件，用来告知 ASTuner 要使用哪些数据、Workflow 与奖励。

在 `tutorial/example_learn2ask/learn2ask.yaml` 中，我们分别进行了如下设置：
- 设置实验名称。
- 使用 `dataset_file` reader 读取训练与验证数据。
- 设置预训练模型。
- 使用 `ExampleLearn2Ask` 作为Workflow。
- 设置其他训练参数。

你可以直接使用该现有配置文件，并在 `tutorial/example_learn2ask/learn2ask.yaml` 中填入数据集路径。

## 3. 开始训练

使用以下命令启动训练：

```bash
python astuner --conf tutorial/example_learn2ask/learn2ask.yaml --backbone='trinity' --with-ray
```

## 4. 结果

### 主要结果

![curve](https://img.alicdn.com/imgextra/i2/O1CN01STeE8G1ricp6jySto_!!6000000005665-2-tps-866-614.png)

## 分析

### 样例 1


**训练前。**
```
User: Bitten by a midge

Agent: Has irritation subsided past initial acute swelling phase afterwards unnoticed scratched overly causing secondary bacterial dermatitis chance affecting skin rendering needed topical antibiotic application options among azelaic acid cream!?
```

Agent 给出了一个略相关但完全不合理的回答，并且未能向用户提出任何有效问题。

**训练中，约 50% step。**
```
User: Bitten by a midge

Agent: Have these itchy bumps appeared soon after getting bitten?(A.) Occasionally immediately afterward ; B fkknore ×later throughout day )/)
```

Agent 给出了一个相对有效的问题，尝试收集用户的一般症状，并提供了两个选项。

**训练后。**
```
User: Bitten by a midge

Agent: Has itching or reddening appeared around this bite site recently without healing despite scratching frequently?(A,Yes indeed worse every evening B,No pain but small swollen spots persist day after incident/)
```

Agent 提出的问题更精确且信息量更大，给出了两个具体且有效的选项。这有助于引导用户回忆症状细节。

## 下一步

若要了解更多关于该任务以及在更大模型上的结果，请参阅 [Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs](https://arxiv.org/abs/2510.25441)。
