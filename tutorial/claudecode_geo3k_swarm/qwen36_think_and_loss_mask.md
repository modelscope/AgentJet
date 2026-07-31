# Qwen3.6 `<think>` 注入行为与 loss mask 头部错位

本文以「代码 → 结果」的形式，列举 Qwen3.6 chat template 中 `<think>` 块的
注入规则，以及它如何导致训练时两条 loss mask 计算路径（override 与 generated）
在头部出现错位。所有结果均由 `Qwen3___6-35B-A3B` 的 tokenizer 实测得到。

## 关键 token id 速查

| token | id | 说明 |
|-------|-----|------|
| `<|im_start|>` | 248045 | 消息起始 |
| `assistant` | 74455 | 角色名 |
| `\n`（单换行） | 198 | 单个换行 |
| `<think>` | 248068 | 思考块起始 |
| `</think>` | 248069 | 思考块结束 |
| `\n\n`（双换行） | 271 | **两个换行合并成的单一 token** |
| `<|im_end|>` (eos) | 248046 | 消息结束 |

> 核心陷阱：单换行(198)与双换行(271)是**不同的 token**。当生成提示末尾的
> 换行与紧随其后的内容换行相邻时，tokenizer 会把它们**合并**成 271。

## 1. 生成提示 `add_generation_prompt=True` —— 由 `enable_thinking` 控制

模板第 147-153 行：

```jinja
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}
```

实测（`ajet_apply_chat_template` 不传该 flag，即 None）：

```text
enable_thinking=None   →  ...assistant'\n<think>\n'
                          ids  = [248045, 74455, 198, 248068, 198]
                          toks = ['<|im_start|>','assistant','\n','<think>','\n']
enable_thinking=True    →  ...assistant'\n<think>\n'            （同上，5 token）
enable_thinking=False   →  ...assistant'\n<think>\n\n</think>\n\n'
                          ids  = [248045, 74455, 198, 248068, 271, 248069, 271]
```

要点：默认（None）生成提示 = 5 token `<|im_start|>assistant\n<think>\n`，
末尾是**单换行 198**；`enable_thinking=False` 才注入空闭合块（用双换行 271）。

## 2. 历史 assistant 消息渲染 —— 由 `preserve_thinking` 或消息位置控制

模板第 100-104 行：

```jinja
{%- if (preserve_thinking is defined and preserve_thinking is true)
       or (loop.index0 > ns.last_query_index) %}
    {{- '...\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '...\n' + content }}
{%- endif %}
```

### 2A. assistant 是最后一条（在最后 user query 之后 → `loop.index0 > last_query_index`）

```text
输入 content = '<think>\nR\n</think>\n\nAns'
preserve_thinking=None   →  '\n<|im_start|>assistant\n<think>\nR\n</think>\n\nAns<|im_end|>\n'
preserve_thinking=True   →  '\n<|im_start|>assistant\n<think>\nR\n</think>\n\nAns<|im_end|>\n'
preserve_thinking=False  →  '\n<|im_start|>assistant\n<think>\nR\n</think>\n\nAns<|im_end|>\n'
```
→ 位置条件已满足，无论 flag 如何都**保留** think 块。

### 2B. assistant 在中间（后面还有 user query → `loop.index0 <= last_query_index`）

```text
preserve_thinking=None   →  '\n<|im_start|>assistant\nAns<|im_end|>\n'                       # 剥离
preserve_thinking=True   →  '\n<|im_start|>assistant\n<think>\nR\n</think>\n\nAns<|im_end|>\n'  # 保留
preserve_thinking=False  →  '\n<|im_start|>assistant\nAns<|im_end|>\n'                       # 剥离
```
→ 仅 `preserve_thinking=True` 保留；默认(None)/False 都**剥离** think 块。

## 3. 两条 loss mask 路径在头部的错位

训练时头部屏蔽由两处独立计算，`get_loss_mask` 内部断言二者必须完全一致：

- **generated mask**（`get_loss_mask`）用 3-token `blackout_token_combo`
  = `<|im_start|>assistant\n` = `[248045, 74455, 198]`
- **override mask**（`replace_token_ids`）用 5-token `generation_prompt_token`
  = `<|im_start|>assistant\n<think>\n` = `[248045, 74455, 198, 248068, 198]`

在真实 rollout 里，LLM 消息是最后一条（第 2A 节 → 保留 think 块），头部随
**模型输出是否含 `<think>...</think>`** 分成两种：

### 3A. 模型输出**含** think 标签（`reasoning_content` 非空）

```text
头部6 token: [248045, 74455, 198, 248068, 198, 19290]
           = ['<|im_start|>','assistant','\n','<think>','\n','reason']
generated(combo3) 精确匹配 [248045,74455,198]        → 命中@0 → 屏蔽前 3 个
override(combo5)  精确匹配 [...,248068,198]           → 命中@0 → 屏蔽前 5 个
```
→ **override=5、generated=3，第 3/4 位（`<think>`,`\n`）错位**。这正是原始报错日志。

### 3B. 模型输出**不含** think 标签（`reasoning_content` 为空）

```text
头部6 token: [248045, 74455, 198, 248068, 271, 248069]
           = ['<|im_start|>','assistant','\n','<think>','\n\n','</think>']
generated(combo3) 精确匹配 [248045,74455,198]        → 命中@0 → 屏蔽前 3 个
override(combo5)  精确匹配 [...,248068,198(单换行)]   → 找不到(位置4是271双换行)
                  → find_sublist_indices 返回 -1，-1+5 = 屏蔽前 4 个
```
→ 空 reasoning 时 `<think>\n` 的换行与 `\n</think>` 的换行**合并成 271**，
5-token 序列不再逐字出现，override 退化成 4；generated 仍是 3。

## 4. 根因总结

| 因素 | 说明 |
|------|------|
| 长度不一致 | combo(3) vs generation_prompt(5)，`<think>\n` 差 2 token |
| enable_thinking 默认开放 | 生成提示尾部固定 `<think>\n`（单换行 198），模型接着填 |
| 换行合并 | 空 reasoning 时 `\n`+`\n` → 271，破坏 5-token 逐字匹配 |
| 精确匹配脆弱 | `find_sublist_indices` 依赖逐字命中，换行变体一动就失配 |

## 5. 可选修复方向

1. **对齐两条路径**：让 `blackout_token_combo` 也用探测法得到的 5-token 生成提示，
   并改用「稳定前缀(去掉尾换行) + 1」匹配，兼容 198/271 两种换行变体。
2. **固定 template 行为**：`ajet_apply_chat_template` 显式传 `enable_thinking=False`，
   让生成提示与历史渲染都用固定的 `<think>\n\n</think>\n\n`，头部 tokenization 稳定。
   代价：会改变是否在 think 块上算 loss 的语义。
