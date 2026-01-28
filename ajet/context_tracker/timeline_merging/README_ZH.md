
# AgentJet Timeline

在多智能体的复杂 LLM Agents 交互过程中，我们称一个 Agent 在任务过程中，反复调用 LLM 产生的 Token 轨迹为一条 Timeline

Timeline 包含以下要素：
- Text 文本 message 列表
    - 提示：在多数qwen模型中，message以 <|im_start|> 开始，以 <|im_end|> 结束，具体取决于模型的 tokenizer 和 chat_template
- Token 序列 message 列表
    - 提示：在多数qwen模型中，message以 <|im_start|> 对应的Token ID开始，以 <|im_end|> 所对应的 Token 结束，具体取决于模型的 tokenizer
- Author 列表
    - 提示：标识每条 message 的生产者。一般有 "llm" 和 "env" 两种。
- Token LogProb Message 列表
    - 提示：记录每个Token生成时的对数概率。如果一个Token的生产者不是 "llm"，则值为 `INVALID_LOG_PROB_VALUE` （根据需求，设置为 0 或者 `np.inf`）
- Loss Mask Message 列表
    - 提示：loss_mask 的每一位都和 Token 一一对应
    - loss_mask=1 代表该Token参与 loss计算，也通常同时代表了该Token是LLM生成的Token
    - loss_mask=0 代表不参与loss计算，在大多数情况下，代表该Token源于用户输入，tokenizer 和 chat_template 的补充，环境反馈等。


## 多轮对话+多智能体情况下，时间线的相互交织

在多轮对话+多智能体的情况下，提取出干净、整齐的时间线并不容易：

- 为方便使用，兼容绝大多数Agentic框架和并行LLM调用，AgentJet 只感知用户（或Agent框架）发出的标准OpenAI格式的 LLM 请求，不需要用户提供LLM请求之间的因果关系。

- 一些 Agentic 框架（如 Langchain 等），为了提高任务的成功率，会自动为用户执行重试操作。例如在LLM工具调用参数不合法时，Agent框架会将报错信息作为临时context拼接在请求中，
达到预期结果后，又将这些临时context剔除。如果不加处理，在这个过程中产生的样本会大幅降低 RL 算法的效率。

- 动态记忆机制的应用：您可以使用[ReMe](https://github.com/agentscope-ai/ReMe)等项目，为Agent提供长短期记忆的能力，大幅提高Agent在个人助理任务中的表现。
当Agent决定更新存在于历史context中的知识时，会让时间线产生分叉点。

- 当环境中存在多个智能体，且当前任务为局部可观测环境时（例如，智能体个体的context中存储了不能相互窥探秘密；或每个智能体通过context offload的技巧，主动屏蔽了一些信息从而更好地专注于当前的任务），
自然会产生多条时间线，每条时间线隶属于一个智能体。

- 当一段 Token 被解码成文本，然后再从文本被tokenizer重新编码回token序列时，有时候并不能精准地转化成的最开始的Token序列（这种漂移在不同模型中发生的概率也不同）。
这种Token漂移需要精细化的处理从而 (1)提高训练效率，(2)稳定训练。


在AgentJet系统中，我们采取的方法是“时间线合并”：

**AgentJet 在epsiode结束时自主地辨识不同时间线的差异，并且根据用户预设，寻找可以合并的时间线并自动地完成合并。进而减少相互重叠的冗余样本数量，提高训练效率。**

## 时间线合并算法

当一个 episode 开始时，AgentJet会初始化一个 context tracker 对象，捕获所有 llm 请求，每次 llm 请求从<|im_start|>开始，到<|im_end|>或者token数量溢出为止。每次 llm 请求在合并前都被视为一条独立的初始时间线。在一个 episode 中，可以采集 m 个 agent 的 n条初始时间线

$\text{Timelines} = \lbrace
T_1\left(M_\text{1}, m_\text{1}, a_\text{1}\right),
T_2\left(M_\text{2}, m_\text{2}, a_\text{2}\right),
\dots,
T_n\left(M_\text{n}, m_\text{n}, a_\text{n}\right)
\rbrace$

其中：
- $T_i$ 代表第 $i$ 条（未合并）时间线。$T_i = [T_{i}^{[1]}, T_{i}^{[2]}, \dots, T_{i}^{[|T_{i}|]}]$。
    - 最后一项 $T_{i}^{[|T_{i}|]} = m_\text{i}$ ：总是这一次 llm 请求的输出。
    - 前 $|T_{i}|-1$ 项：总是这个llm请求的输入 $M_\text{i}$。
- $a_\text{i} \in \lbrace A_1, \dots, A_m \rbrace$ 代表 agent 的名称ID。值得一提的是，当用户的workflow没有提供agent名称时，则将 $\lbrace
T_1, T_2, \dots, T_n\rbrace$ 视为源于同一个agent（default agent）的timeline。
- $M_\text{i}$ 和 $m_\text{i}$ 分别代表输入 message 列表和输出 message。每条 message 具备 Text，Token，Loss Mask 三元组。

当 episode 结束时，对比所有时间线。如果两条时间线满足以下条件时：

- 条件1: $|T_{i}| <= |T_{j}|$
- 条件2: $T_{i}$ 和 $T_{j}$ 的前 $|T_{i}|$ 个 message 的 Token 序列全等。即 $\text{Token}(T_{i}^{[k]}) = \text{Token}(T_{j}^{[k]}), \forall k \in \left[1, |T_{i}| \right]$。

则将两条时间线合并：

- $T_{i}$ 是**被吸收**的短时间线；$T_{j}$ 是需要更新的长时间线。
- 如果有一组全等 message 满足 $\text{Author}(T_{i}^{[k]}) = \text{llm}$ 且 $\text{Author}(T_{j}^{[k]}) \neq \text{llm}$，则：
    - $\text{Author}(T_{j}^{[k]}) = \text{llm}$
    - $\text{Token}(T_{j}^{[k]}) = \text{Token}(T_{i}^{[k]})$
    - $\text{TokenLogProb}(T_{j}^{[k]}) = \text{TokenLogProb}(T_{i}^{[k]})$

    ```python
    def toggle_author_and_mask(
        source_timeline: List[ExtendedMessage], # the longer timeline
        target_timeline: List[ExtendedMessage], # the shorter timeline
    ) -> List[ExtendedMessage]:
        for k in range(len(target_timeline)):
            if target_timeline[k].author == "llm" and source_timeline[k].author != "llm":
                source_timeline[k].author = target_timeline[k].author
                source_timeline[k].token_arr = target_timeline[k].token_arr
                source_timeline[k].token_logprob_arr = target_timeline[k].token_logprob_arr
                assert source_timeline[k].need_training
        return source_timeline  # merged timeline
    ```

备注：Loss Mask 会根据 $\text{Author}(\cdot)$ 列表进行精细的后期计算，因此合并时间线时不需要关注。

## 更宽松的合并条件，换更快的训练速度

### 宽松Token匹配

在实践中我们发现，当一段 Token 被解码成文本，然后再从文本被tokenizer重新编码回token序列时，有时候并不能精准地转化成的最开始的Token序列。

因此，现实中经常会发生的情况是：
-  $\text{Author}(T_{i}^{[k]}) = \text{llm}$
-  $\text{Author}(T_{j}^{[k]}) \neq \text{llm}$
-  $\text{Text}(T_{j}^{[k]}) = \text{Text}(T_{i}^{[k]})$
-  $\text{Token}(T_{j}^{[k]}) \neq \text{Token}(T_{i}^{[k]})$

即一个文本序列完全相等，但在 vllm 内部的Tokenizer转换中，产生了两种 Token 序列的变体。
在这种情况下，您可以通过调整
```yaml
ajet.context_tracker.timeline_merging_policy.timeline_compare_level = "text" / "token"  #（default text）
```
控制AgentJet的行为。


| 合并策略 | 合并条件 | 适用场景 | 优点 | 缺点 |
|---------|---------|---------|------|------|
| **token** | 要求 $\text{Token}(T_{i}^{[k]}) = \text{Token}(T_{j}^{[k]})$ | Token 序列必须完全一致才能合并 | 严格匹配，训练数据精确度高 | 由于 tokenizer 的编解码漂移，可能导致无法合并本应合并的时间线，训练效率降低 |
| **text** | 只要求 $\text{Text}(T_{i}^{[k]}) = \text{Text}(T_{j}^{[k]})$ | 文本内容相同即可合并，容忍 Token 序列差异 | 更宽松的合并条件，提高合并率和训练效率，减少冗余样本 | 可能合并 Token 表示略有不同的样本，但在实践中影响很小 |

**推荐配置：**
- 默认使用 `"text"` 策略，可以有效处理 tokenizer 编解码过程中的 Token 漂移问题。
- 在需要严格保证 训推一致性 时使用 `"token"` 策略。

### 宽松 Tool 匹配

大部分模型的tokenizer chat template会将需要使用tool清单安排在最开头（system prompt）。
当Agent的工具清单发生微调，但其他context没有变化时，您可以通过调整
```yaml
ajet.context_tracker.timeline_merging_policy.ignore_tools = True / False #（default True）
```
控制AgentJet的行为。

| 合并策略 | 合并条件 | 适用场景 | 优点 | 缺点 |
|---------|---------|---------|------|------|
| **True** | 忽略工具清单差异，只要其他 context 相同即可合并 | Agent 工具清单动态变化，但核心对话逻辑不变 | 大幅提高合并率，减少因工具清单变化导致的冗余样本，提升训练效率 | 可能合并工具环境略有差异的样本，但在大多数场景下影响有限 |
| **False** | 严格比较工具清单，工具清单必须完全一致才能合并 | 工具调用对训练至关重要，需要精确匹配工具配置 | 保证时间线的工具环境完全一致，训练数据严格对齐 | 当工具清单发生微小变化时，无法合并context相同的时间线，训练效率降低 |

**推荐配置：**
- 使用 `True` 策略，可以有效减少冗余样本。
- 在需要严格保证 训推一致性 时使用 `False` 策略。当 Agent 工具大幅度、低频变化（如动态加载工具、工具版本更新等）时，也建议使用 `False`。

## 其他时间线管理选项

### 自动 Re-tokenization 漂移修复

AgentJet默认情况下会自动根据 vLLM 引擎返回的 Token ID 进行 Re-tokenization 漂移修复。这会多消耗一点点CPU算力。

```yaml
ajet.context_tracker.fix_retokenization_drift = True #（default True）
```

关于 Re-tokenization 漂移现象，您可以关注 https://github.com/vllm-project/vllm/pull/22587 了解详细信息。

### 检测时间线分歧点

在单Agent+多轮对话场景中，如果您对训练效率非常关注，希望详细诊断您使用的 Agentic 框架在什么时刻、出于何种原因，让时间线产生了分叉，可以开启

```yaml
ajet.context_tracker.detect_timeline_snap = False #（default False）
```

启动时间线分歧点实时检测功能。这会消耗CPU算力，拖慢训练进程。仅建议在调试模式（`--backbone=debug`）时使用。


