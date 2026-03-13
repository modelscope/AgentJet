# DeepFinance: 通过强化学习训练金融深度研究 Agent

## 概述

DeepFinance 是基于 AgentJet 框架构建的金融深度研究 Agent 训练方案。其核心目标是：通过 GRPO 强化学习，训练 LLM 自主调用金融工具、收集多源数据、进行交叉验证，并最终生成结构化、有据可查的投资研究报告。

与传统 SFT 微调不同，DeepFinance 不依赖人工标注的「标准回答」来监督训练，而是设计了一套 **多维度奖励体系** 作为 RL 训练信号——让模型在「写报告」的过程中自行探索最优策略，并通过 5 个正交维度的评分反馈来持续改进。

**训练闭环**：

```plain
金融问题 → Agent 调用工具收集数据 → 生成研究报告 → 多维度 Judge 评分 → GRPO 策略更新 → 下一轮生成
```

------

## Pipeline

整个训练流水线由 4 个核心模块组成：

| 模块         | 文件                               | 职责                                                |
| ------------ | ---------------------------------- | --------------------------------------------------- |
| **Reader**   | `deep_finance_reader.py`           | 加载 JSON 训练数据，组装 System Prompt + User Query |
| **Workflow** | `deep_finance.py`                  | 定义 ReAct Agent 的多轮交互逻辑，维护对话历史       |
| **Judge**    | `deep_finance_judge.py` + `judge/` | 多维度奖励评分（核心创新）                          |
| **配置**     | `deep_finance.yaml` / `*.sh`       | 训练参数、奖励权重、环境配置                        |

```plain
┌─────────────────────────────────────────────────────────────┐
│                    AgentJet 训练框架                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────┐               │
│  │ DeepFinance   │    │  ExampleDeepResearch │               │
│  │ Reader        │───>│  Protocol (Workflow) │               │
│  │ 数据加载 +     │    │  ReAct Agent 多轮交互 │               │
│  │ Prompt 组装   │    └──────────┬───────────┘               │
│  └──────────────┘               │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  EnvService (FinWorld)  │               │
│                    │  19 个金融工具 + MCP    │               │
│                    │  MongoDB 缓存加速       │               │
│                    └────────────┬───────────┘               │
│                                 │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  DeepFinanceJudge      │               │
│                    │  多 维 Reward 评分       │               │
│                    │  (基于 OpenJudge)       │               │
│                    └────────────┬───────────┘               │
│                                 │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  GRPO Trainer (verl)    │               │
│                    │  多机多卡 Ray 集群       │               │
│                    └────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

------

## Workflow设计

### 两阶段深度研究流程

Agent 的 System Prompt（`prompt/finance_analyst_prompt.md`）要求遵循两阶段研究方法：

**第一阶段：先大纲后调研**

1. 理解用户问题类型（个股分析/行业研究/事件解读/宏观分析/股票检索）
2. **先输出研究大纲**（一级/二级标题 + 每节的 Key Questions），此阶段不调用工具
3. 按大纲逐段调研，每轮调用工具后做小结

**第二阶段：深度分析与报告生成**

1. 当数据充分后，基于真实数据生成 Markdown 格式研究报告
2. 写作中发现证据不足时允许追加 1-2 轮工具调用补充取证
3. 报告末尾添加 `[TASK_COMPLETED]` 标记

### 引用规范

Agent 被要求使用学术论文风格的引用标注：

- 所有关键事实句句末必须添加引用编号 `[n]`
- 报告末尾必须包含 `## References` 小节
- 引用必须可追溯到实际工具返回的数据，禁止伪造

------

## 工具体系

DeepFinance 集成了 **19 个金融工具**，通过 MCP（Model Context Protocol）协议与 EnvService 交互，覆盖金融研究的完整数据需求。

| 类别               | 工具                    | 功能                                |
| ------------------ | ----------------------- | ----------------------------------- |
| **实体与计算**     | `extract_entities_code` | 从自然语言中提取金融实体并查找代码  |
|                    | `history_calculate`     | A股历史股价分析（支持自然语言提问） |
| **通用能力**       | `dashscope_search`      | 互联网搜索                          |
|                    | `execute_code`          | Python 代码执行                     |
|                    | `execute_shell`         | Shell 命令执行                      |
| **同花顺专项数据** | `crawl_ths_company`     | 上市公司基本资料                    |
|                    | `crawl_ths_holder`      | 股东研究信息                        |
|                    | `crawl_ths_operate`     | 经营分析信息                        |
|                    | `crawl_ths_finance`     | 财务分析信息                        |
|                    | `crawl_ths_worth`       | 盈利预测信息                        |
|                    | `crawl_ths_news`        | 新闻公告信息                        |
|                    | `crawl_ths_concept`     | 概念题材信息                        |
|                    | `crawl_ths_equity`      | 股本结构信息                        |
|                    | `crawl_ths_capital`     | 资本运作信息                        |
|                    | `crawl_ths_position`    | 主力持仓信息                        |
|                    | `crawl_ths_bonus`       | 分红融资信息                        |
|                    | `crawl_ths_event`       | 公司大事信息                        |
|                    | `crawl_ths_field`       | 行业对比信息                        |

工具调用规范：

- 每次最多调用 **3 个工具**，采用多轮次渐进式调研
- Agent 必须先搜索确认信息（如股票代码），再进行深度查询
- 每轮工具调用后先做小结，再决定下一步调研方向

------

## 奖励设计（Reward Design）

这是 DeepFinance 的核心创新。我们设计了 **5 个正交维度** 的评分器（Grader），通过可配置的权重加权融合为最终 reward，并额外引入工具调用惩罚机制。

### 总体公式

```plain
final_reward = Σ(w_i × grader_i_score) + tool_penalty
```

其中各 grader 权重归一化（`Σw_i = 1`），`tool_penalty` 为额外惩罚项。

### 5 个评分维度总览

| 维度             | 名称                | 评估对象           | 核心问题                                         |
| ---------------- | ------------------- | ------------------ | ------------------------------------------------ |
| **分析充分性**   | RM Gallery          | 报告整体质量       | 分析是否充分？逻辑是否合理？                     |
| **呈现质量**     | PresentationQuality | 报告排版与结构     | 读者体验好不好？信息是否易获取？                 |
| **引用规范性**   | Grounding           | 引用的覆盖与真实性 | 关键事实是否都有引用？引用是否真实？             |
| **证据溯源**     | EBTU                | 原子断言的证据锚定 | 每个数字/事实能否追溯到工具返回的原始数据？      |
| **引用逻辑审计** | Audit               | 引用的逻辑蕴含关系 | 引用是否真正支撑了对应的陈述？有没有夸大或捏造？ |

默认权重配置（可在 shell 脚本中调整）：

```bash
RM_WEIGHT=0.5                       # 分析充分性
PRESENTATION_QUALITY_WEIGHT=0.2    # 呈现质量
GROUNDING_WEIGHT=0.1               # 引用规范性
EBTU_WEIGHT=0.2                     # 证据溯源（可选启用）
AUDIT_WEIGHT=0.0                    # 引用逻辑审计（可选启用）
```

------

### 1) 分析充分性（RM Gallery）

**目标**：评估报告的分析深度、覆盖面和逻辑性——回答「分析得好不好」。

**机制**：使用 `finance_composition` 评估器，通过独立的 Judge LLM（ `qwen-max`）对生成报告与参考答案进行对比评估。

**评估维度（按金融 domain 分域）**：

- 分析深度：对核心问题的挖掘是否足够深入
- 覆盖面：是否覆盖了问题涉及的多个分析维度（基本面、财务、估值、行业、新闻等）
- 逻辑性：分析推理链条是否完整、结论是否有据可依

**输入输出**：

- 输入：用户 Query + Agent 生成的报告 + 参考答案
- 输出：`[0, 1]` 归一化分数

------

### 2) 呈现质量（Presentation Quality）

**目标**：评估报告的用户体验与信息架构——回答「写得好不好看、好不好读」。

**严格不评估**：事实真伪、引用准确性、内容深度（这些由其他 Grader 负责）。

**8 项子指标（1/3/5 分制）**：

| 分类                       | 指标            | 5分标准                                          |
| -------------------------- | --------------- | ------------------------------------------------ |
| **Scan 可扫描性**          | A1 结论先行     | 开头有独立摘要/TL;DR，读者无需滚动即可获取主结论 |
|                            | A2 结构导航     | 层级分明（H1/H2/H3），长文有清晰小标题路标       |
|                            | A3 视觉重点     | 精准使用加粗/斜体强调核心洞察，信噪比高          |
| **Structuring 信息结构化** | B1 密集信息解构 | 复杂数据用表格/嵌套列表呈现，一目了然            |
|                            | B2 对比对齐     | 方案A vs B / 历史 vs 现状使用表格，维度横向可比  |
|                            | B3 一致性与渲染 | 格式统一，Markdown 渲染完美                      |
| **Editorial 编辑清晰度**   | C1 论证链可视化 | 逻辑链条可视（主张→证据→结论），引用锚点清晰     |
|                            | C2 风险与行动   | 独立板块列出风险/局限性及下一步建议              |

**评分计算**：

```plain
score = Σ(8项得分) / 40    # 归一化到 [0, 1]
```

**反刷分机制**：空表格、无意义重复列表、为格式而格式 → 直接判 1 分。

------

### 3) 引用规范性（Grounding）

**目标**：评估报告的引用覆盖率和引用真实性——回答「关键事实都有出处吗？引用是真的吗？」

**评估流程**：

1. 从对话轨迹中提取 User Query、Evidence（工具调用与返回）、最终报告
2. LLM 审计员识别报告中的所有「关键事实句」（含数字/日期/财务指标/确定性陈述）
3. 检查每个关键事实句句末是否有引用标记 `[n]`
4. 检查引用是否在 References 中有合法条目（有效 URL 或完整的 no-url 记录）
5. 检查引用内容与 Evidence 是否一致（检测虚假引用）

**输出字段**：

- `total_key_facts`：关键事实句总数
- `cited_key_facts`：句末有引用的关键事实句数
- `fake_count`：引用内容与证据明显矛盾的数量
- `missing_count`：缺少引用的关键事实句数
- `invalid_reference_nums`：不合规的引用编号

**评分计算**：

```plain
citation_coverage = cited_key_facts / total_key_facts     # 引用覆盖率
grounding_score = 1 - fake_count / cited_key_facts        # 引用真实性
final_score = 0.5 × coverage + 0.5 × grounding            # 综合分数
```

------

### 4) 引用逻辑审计（AUDIT - Citation Integrity Audit）

**目标**：审计 AI 研究报告中的每一个引用标记 `[n]` 是否严格符合「逻辑蕴含（Logical Entailment）」原则——回答「每个引用是否被原始证据严格支撑？」

**核心理念：证据优先（Evidence-first）**。审计官必须像法官判案一样，先罗列证据，再进行逻辑推导，最后下达判决，严禁先下结论再找证据。

**三步验证流程**：

1. **提取（Extract）**：锁定报告中由 `[n]` 支撑的陈述片段（Claim）
2. **溯源（Trace）**：在 Reference 列表中找到 `[n]` 对应的原始文本，摘录核心证据句（Source Quote）
3. **比对（Compare）**：分析 Claim 是否被 Source Quote 严格支撑
   - Check: 数字/事实是否一致？
   - Check: 语气是否一致（有没有把"可能"改成"确定"）？
   - Check: 因果关系是否存在？

**判决标准（Verdict Criteria）**：

| Verdict        | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| `Supported`    | 证据充分，逻辑闭环。允许合理的概括，但禁止添加细节           |
| `Overstated`   | 夸大其词。证据只说了 A，报告却写成了 A+（如去掉限定词、强加因果） |
| `Contradicted` | 事实冲突。报告内容与证据相反                                 |
| `Hallucinated` | 无中生有。关键细节在证据中找不到，或引用编号不存在           |
| `Irrelevant`   | 引用无效。证据内容真实，但与报告所述主题无关                 |

**评分计算**（确定性打分，由 Python 代码计算，非 LLM 输出）：

```plain
integrity_score = Supported数量 / 总引用数
```

关键设计：LLM 只负责结构化输出（Claim 提取 + 证据溯源 + 逻辑分析 + 判决），分数完全由代码确定性计算，避免 LLM 自评分的不稳定性。

------

### 工具调用惩罚

在加权融合分数之外，额外施加工具调用惩罚，鼓励 Agent 积极使用工具收集数据：

| 工具调用次数 | 惩罚          |
| ------------ | ------------- |
| 0 次         | -1.0          |
| 1-2 次       | -0.5          |
| ≥3 次        | 0.0（无惩罚） |

------

## Quick Start

### 环境准备

1. 安装 AgentJet 及依赖

```bash
cd /path/to/AgentJet
bash install.sh # TODO：把这部分缩减到一个install：https://yuque.alibaba-inc.com/bayotg/wxz7sb/qdesuu33621x2yhi
```

1. 配置 `.env` 文件（API 密钥、模型路径、数据路径等）：

```bash
# .env 示例
MODEL_PATH=/path/to/Qwen3-8B
TRAIN_DATA_PATH=/path/to/train.json
VAL_DATA_PATH=/path/to/val.json
TRAIN_REF_ANS_PATH=/path/to/train_ref_answers.json
VAL_REF_ANS_PATH=/path/to/val_ref_answers.json
CKPT_SAVE_PATH=/path/to/checkpoints
OPENJUDGE_API_KEY=your_api_key
RM_API_KEY=your_api_key
```

1. 启动 EnvService（金融工具服务）

### 单机调试模式

```bash
bash tutorial/example_deep_finance/deep_finance_single.sh
```

该脚本以 `--backbone="debug"` 模式运行，适合验证工作流和调试。

### 多机训练模式

```bash
# 在 PAI-DLC 或多机环境中提交
bash tutorial/example_deep_finance/deep_finance.sh
```

该脚本会：

1. 从 YAML 模板动态生成配置文件
2. 在 Master 节点启动 Ray Head + 训练任务
3. Worker 节点自动加入 Ray 集群

### 关键参数说明

| 参数                          | 默认值 | 说明                                  |
| ----------------------------- | ------ | ------------------------------------- |
| `NUM_REPEAT`                  | 4      | Group size，每个 query rollout 的次数 |
| `NUM_STEPS`                   | 6      | 每个样本的最大交互轮数                |
| `TRAIN_BATCH_SIZE`            | 32     | 训练 batch size                       |
| `RM_WEIGHT`                   | 0.5    | 分析充分性权重                        |
| `PRESENTATION_QUALITY_WEIGHT` | 0.25   | 呈现质量权重                          |
| `GROUNDING_WEIGHT`            | 0.25   | 引用规范性权重                        |
| `EBTU_WEIGHT`                 | 0.0    | 证据溯源权重（可选启用）              |
| `AUDIT_WEIGHT`                | 0.0    | 引用逻辑审计权重（可选启用）          |

------

## 实验结果


![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843906200-9dd35ac4-f71e-40dc-b130-f03e3e6bae6a.png)

![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843940824-4e3637d7-a16e-4994-8878-242effc2c0d7.png)![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843950142-09def779-5521-41f0-a457-a7715a819cc7.png)


