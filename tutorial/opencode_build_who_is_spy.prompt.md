# Generate an agent / agent loop with AgentJet Swarm and train it with one key

Use prompt below in opencode or claudecode to generate a one-key-to-tune agent

=============================

Your task:
- Write an agent that learns the "Who is the Spy" task, trained via a combination of reinforcement learning and supervised learning. Game rules are as follows:
  - The game has N players in total, most of whom are **civilians**, and a few are **spies**
  - At the start of the game, each civilian receives the same **civilian word**, and each spy receives a **spy word** that is similar but different from the civilian word (e.g., civilian word is "apple", spy word is "pear")
  - In each round, all players take turns giving a **verbal description** of the word they were given. The description must truthfully reflect their own word, but must not state the word itself directly, nor reveal their identity too obviously
  - After all players finish describing, the game enters the **voting phase**, where all players vote for whom they consider the most suspicious spy. The player with the most votes is eliminated
  - The game continues for multiple rounds until one of the following end conditions is met:
    - **Civilians win**: all spies have been eliminated
    - **Spies win**: the number of spies ≥ the number of civilians (spies gain a numerical advantage)
  - The agent needs to master two core capabilities through extensive self-play training:
    - **Description strategy learning**: learn to generate the optimal description based on its own word and the current situation — one that neither reveals its identity nor fails to win recognition from teammates
    - **Reasoning and decision learning**: learn to accurately identify spies and make optimal voting decisions based on dialogue history, other players' description patterns, and behavioral features
  - Training objective: maximize the agent's win rate across different roles (civilian/spy), continuously optimizing strategy via self-play and reward mechanisms
- I want to use the base model `/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct`
- Train with 8 GPUs
- Batch size 16
- I currently have no dataset; you need to help me mock a small amount of game episode data for testing and initial training
- Use the OpenAI SDK, flexibly using Tools
- No Chinese characters are allowed in the code

Your skill (read this SKILL file first to acquire the necessary knowledge):
./ajet/copilot/write-swarm-client/SKILL.md

- Additional requirements:
  - optional 0. (agent_roll) team A civilians share a single 7B model; team B spies use qwen-max (DASHSCOPE_API_KEY is already in the environment variables).
                              For each episode, randomly assign everyone's ID and name (randomly generate a long list of random full names). Winners get reward 1, losers get reward 0
  - optional 1. (agent_roll_adv) adversarial training: team A civilians share one 7B model (swarm server 1), team B spies share another 7B model (swarm server 2).
                              For each episode, randomly assign everyone's ID and name (randomly generate a long list of random full names). Winners get reward 1, losers get reward 0

- Additional requirements:
    agent_roll: use 4 GPUs
    agent_roll_adv: swarm server 1 and swarm server 2 each use 4 GPUs (8 GPUs in total)

- Additional requirements: debug using tmux + uv's .venv until all bugs are cleared and training starts normally. You may use the three tmux sessions `spy-swarm-server`, `spy-swarm-server-2`, `spy-swarm-client`

    - Current debugging stage:
        Debug agent_roll 【execute debugging】
        Debug agent_roll_adv 【skip debugging】

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
