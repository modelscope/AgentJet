
# AgentJet Timeline

在多智能体的复杂 LLM Agents 交互过程中，我们称一个 Agent 在任务过程中，反复调用 LLM 产生的 Token 轨迹为一条 Timeline

Timeline 包含以下要素：

- Text 文本 message 列表
    - 提示：在多数qwen模型中，message以 <|im_start|> 开始，以 <|im_end|> 结束，具体取决于模型的 tokenizer 和 chat_template
- Token 序列 message 列表
    - 提示：在多数qwen模型中，message以 <|im_start|> 对应的Token ID开始，以 <|im_end|> 所对应的 Token 结束，具体取决于模型的 tokenizer
- Loss Mask Message 列表
    - 提示：loss_mask 的每一位都和 Token 一一对应
    - loss_mask=1 代表该Token参与 loss计算，也通常同时代表了该Token是LLM生成的Token
    - loss_mask=0 代表不参与loss计算，在大多数情况下，代表该Token源于用户输入，tokenizer 和 chat_template 的补充，环境反馈等。


Timeline


<!--

uv pip install -e /mnt/data_cpfs/taoshuchang.tsc/deepresearch/RM-Gallery -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -e /mnt/data_cpfs/taoshuchang.tsc/deepresearch/OpenJudge -i https://mirrors.aliyun.com/pypi/simple/
uv pip install openai==1.109.1 -i https://mirrors.aliyun.com/pypi/simple/ -->
