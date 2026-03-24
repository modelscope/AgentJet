# Generate an agent / agent loop with AgentJet Swarm and train it with one key

Use prompt below in opencode or claudecode to generate a one-key-to-tune agent

=============================

English prompt to be tranlated ...

=============================

你的任务：
- 编写一个学习"谁是卧底"任务的智能体，通过强化学习和监督学习相结合的方式训练，游戏规则如下：
  - 游戏共有 N 名玩家，其中大多数人是**平民**，少数人是**卧底**
  - 游戏开始时，每位平民会收到同一个**平民词**，每位卧底会收到一个与平民词相近但不同的**卧底词**（例如平民词为"苹果"，卧底词为"梨"）
  - 每轮游戏中，所有玩家依次对自己拿到的词进行**口头描述**，描述必须真实反映自己的词，但不能直接说出词语本身，也不能过于明显地暴露自己的身份
  - 全部玩家描述完毕后，进入**投票环节**，所有玩家投票选出自己认为最可疑的卧底，得票最多的玩家被淘汰出局
  - 游戏持续多轮，直到满足以下任一结束条件：
    - **平民获胜**：所有卧底均被淘汰
    - **卧底获胜**：卧底人数 ≥ 平民人数（卧底在数量上取得优势）
  - 智能体需要通过大量对局训练掌握两种核心能力：
    - **描述策略学习**：学会根据自己的词语和当前局势，生成既不暴露身份、又能让同阵营玩家认同的最优描述
    - **推理决策学习**：学会根据历史对话、其他玩家的描述模式和行为特征，准确识别卧底并做出最优投票决策
  - 训练目标：最大化智能体在不同角色（平民/卧底）下的游戏胜率，通过自对弈和奖励机制不断优化策略
- 我希望使用基础模型 `/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct`
- 使用 8 GPU 训练
- Batch Size 16
- 我目前没有数据集，你需要帮助我 mock 少量游戏对局数据以供测试和初始训练
- 使用OpenAI SDK，灵活使用Tools
- 代码中不得出现中文

你的 skill（首先读取该 SKILL 文件，获取必要知识）：
./ajet/copilot/write-swarm-client/SKILL.md

- 追加要求：
  - optional 0. (agent_roll) team A 平民 共享一个7B模型, team B卧底使用qwen-max （DASHSCOPE_API_KEY已经在环境变量中），
                              每个episode随机分配每个所有人的ID和名字（随机生成一个长长的随机姓名名字清单），胜者奖励 1，败者奖励 0
  - optional 1. (agent_roll_adv) 对抗式训练，team A 平民 共享一个7B模型（swarm server 1）， team B卧底共享另一个7B模型（swarm server 2），
                              每个episode随机分配每个所有人的ID和名字（随机生成一个长长的随机姓名名字清单），胜者奖励 1，败者奖励 0

- 追加要求：
    agent_roll： 使用4个显卡
    agent_roll_adv：swarm server 1 和 swarm server 2 分别使用4个显卡（一共8个显卡）

- 追加要求：使用 tmux + uv 的 .venv 调试，直到所有Bug都已经排除 & 训练正常开始。你可以使用 `spy-swarm-server`, `spy-swarm-server-2`, `spy-swarm-client` 三个 tmux session

    - 当前调试阶段：
        调试 agent_roll 【执行调试】
        调试 agent_roll_adv 【跳过调试】
