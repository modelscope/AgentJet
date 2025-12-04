# 配置指南

本页对 AgentScope Tuner 的配置文件进行详细说明。

## 总览
AgentScope Tuner 使用 YAML 格式的配置文件来设置数据、训练算法、奖励、日志以及其他运行时行为。默认配置文件位于 `astuner/default_config/astune_default.yaml`。

一个典型的配置文件包含一个根节点 `astuner`，进一步被划分为若干部分：

- **基础信息**
  - `project_name`, `experiment_name`, `experiment_dir`：用于标识实验及其保存位置。
  - `backbone`：选择训练后端，例如 `debug`、`trinity` 或 `verl`。
- **数据与奖励**
  - `task_reader`：如何加载训练 / 验证样本（EnvService、本地文件、HuggingFace 数据集等）。
  - `task_judge`：如何评估 Agent 并计算奖励（自定义评测器或基于 LLM 的自动打分器）。
  - `data`：Prompt / Response 的长度和 batch 大小等。
- **模型**
  - `model`：要训练的基础模型从哪里加载（本地路径或 HuggingFace 仓库）。
- **Rollout 与交互配置**
  - `rollout`：Agent 如何与环境交互（协议、采样参数、最大步数等）。
  - `context_tracker`：如何管理对话 / 历史信息。
- **训练配置**
  - `trainer_common`：全局训练超参数（epoch、checkpoint、优化算法参数、损失、FSDP、logger 等）。
  - `debug`：当 `backbone: debug` 时使用的额外调试配置。

你可以从默认 YAML 开始，只修改与你的使用场景相关的部分。文末 *附录* 提供了一个 **完整配置示例** 供参考。

## 模型
要训练一个 Agent，首先需要指定待训练的模型，以及训练过程中使用的一些环境变量。

在配置文件中，你可以设置从某个位置加载模型参数的路径：

```yaml
astuner:
  # ...
  
  # 待训练的模型
  model:
    path: path/to/model
    
  # ...

```

你可以通过以下方式指定模型来源：

+ 本地文件：指向包含 Transformers 格式模型的本地目录，例如 `/mnt/data/models/Qwen2.5-14B-Instruct`
+ HuggingFace 仓库：指向某个 HuggingFace 仓库，例如 `Qwen/Qwen2.5-14B-Instruct`。模型会自动下载并加载。



另外，如果在训练中使用 LLM-as-a-Judge，则需要配置一些必要的环境变量：

```bash
# the API key of DashScope, which provides the remove LLM calling
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
```

## 数据
数据相关的配置主要包括两部分：`task_reader` 和 `task_judge`。

### Task Reader
`task_reader` 用于定义如何读取训练集和验证集。目前支持以下三种类型：

```yaml
astuner:
  task_reader:
    # options:
    #   env_service: read dataset from EnvService
    #   dataset_file: read dataset from local file
    #   huggingface_dat_repo# read dataset from huggingface repo
    type: env_service

    # 1. env_service reader config
    env_service:
      env_type: "appworld"
      env_url: "http://127.0.0.1:8080"
      env_action_preference: code
      training_split: train
      validation_split: dev

    # 2. dataset_file reader config
    dataset_file:
      training:
        file_path: "xxxx.jsonl"
      validation:
        file_path: "xxxx.jsonl"

    # 3. huggingface_dat_repo reader config
    huggingface_dat_repo:
      dataset_path: "gsm8k"
      training_split: "train"
      validation_split: "validation"
```

+ `env_service`：从 EnvService 中读取数据，适用于需要与 EnvService 交互的任务。
    - `env_type`：环境类型，需要与 EnvService 中初始化的环境类型保持一致（例如 `appworld`）。
    - `env_url`：EnvService 的服务地址（例如 `http://127.0.0.1:8080`）。
    - `env_action_preference`：偏好的 Action 形式，可选 `code`、`text` 或 `box`。
    - `training_split`：在环境中用于训练的数据切分名称。
    - `validation_split`：在环境中用于验证的数据切分名称。
+ `dataset_file`：从本地文件中读取数据，通常为 JSONL 格式。
    - `training.file_path`：训练数据集的本地路径。
    - `validation.file_path`：验证数据集的本地路径。
+ `huggingface_dat_repo`：直接从 HuggingFace 仓库中读取数据集。
    - `dataset_path`：HuggingFace 上的数据集仓库名（例如 `gsm8k`）。
    - `training_split`：用于训练的数据集切分名称。
    - `validation_split`：用于验证的数据集切分名称。

### Task Judge
`task_judge` 用于评估 Agent 的表现并计算奖励。

```yaml
astuner:
  task_judge:
    # options: 'customized_protocal', 'rubrics_auto_grader'
    judge_type: customized_protocal
    # the package path to judge (reward) function
    judge_protocol: astuner.task_judge.env_service_as_judge->EnvServiceJudge
    # LLM, which may be used by judge
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512

    # rubrics_auto_grader config
    rubrics_auto_grader: # only active when `judge_type: rubrics_auto_grader`
      model_name: qwen-max
      grader_mode: pointwise
      # ...
```

+ `judge_type`：评测方式。
    - `customized_protocal`：使用自定义 Python 类进行打分。需要通过 `judge_protocol` 指定类路径（例如 `package.module->ClassName`）。
    - `rubrics_auto_grader`：使用基于 LLM 的自动打分。
+ `alien_llm_model`：评测时可能用到的辅助 LLM 模型。

## 训练配置
### 后端
AgentScope Tuner 支持三种训练后端：**trinity**、**verl**，以及一个额外的 **debug** 模式。

要配置所使用的后端，可以修改：

```yaml
astuner:
  # debug or trinity or verl
  backbone: trinity
```

### Rollout
`rollout` 配置控制 Agent 在与环境进行交互采样过程中的行为。

```yaml
astuner:
  rollout:
    use_agentscope_protocol: True
    agentscope_learn_protocol: tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol
    max_env_worker: 128
    temperature: 0.9
    top_p: 1.0
    name: vllm
    n_vllm_engine: 2
    num_repeat: 4
```

+ `use_agentscope_protocol`：是否使用 AgentScope 定义的交互协议。
+ `agentscope_learn_protocol`：指定具体的交互协议实现类。
+ `temperature` / `top_p`：采样参数。
+ `name`：推理引擎名称（例如 `vllm`）。
+ `n_vllm_engine`：使用的 vLLM 引擎数量（仅在 backbone 为 trinity 时生效）。

### Context Tracker
当且仅当 `rollout.use_agentscope_protocol=False` 时，才会使用 `context_tracker`，可独立于 AgentScope 管理对话。

```yaml
astuner:
  context_tracker:
    context_tracker_type: "linear"
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
    # ...
```

- `context_tracker_type`：上下文管理策略，例如 `linear`。
- `alien_llm_model` / `alien_llm_response_length`：用于上下文管理的辅助 LLM 及其最大回复长度。

其他策略（例如 `auto_context_cm`、`sliding_window_cm`、`linear_think_cm`）请依照实际情况启用。

### 通用参数
`trainer_common` 包含训练流程控制的通用参数：

```yaml
astuner:
  trainer_common:
    total_epochs: 50
    save_freq: 20
    test_freq: 20
    val_before_train: False
    val_pass_n: 4
    nnodes: 1
    n_gpus_per_node: 8
    mini_batch_num: 1
    ulysses_sequence_parallel_size: 1
    fsdp_config:
      param_offload: True
      optimizer_offload: True
```

+ `total_epochs`：训练总 epoch 数。
+ `save_freq`：保存模型 checkpoint 的频率（以 step 计）。
+ `test_freq`：执行验证 / 测试的频率（以 step 计）。
+ `val_before_train`：是否在训练开始前先执行一次验证。
+ `val_pass_n`：验证阶段每个问题的采样数量（Pass@N）。
+ `nnodes` / `n_gpus_per_node`：分布式训练配置，用于指定节点数以及每个节点的 GPU 数量。
+ `mini_batch_num`：梯度累积的 mini-batch 数量。
+ `ulysses_sequence_parallel_size`：Ulysses attention 的序列并行大小。
+ `fsdp_config`：FSDP（Fully Sharded Data Parallel）配置。
    - `param_offload`：是否将模型参数 offload 到 CPU 以节省 GPU 显存。
    - `optimizer_offload`：是否将优化器状态 offload 到 CPU。

### 优化算法
优化算法及其超参数主要在 `algorithm`、`optim` 和根配置中进行设置：

```yaml
astuner:
  trainer_common:
    algorithm:
      adv_estimator: grpo
      use_kl_in_reward: False
    optim:
      lr: 1e-6
    use_kl_loss: True
    kl_loss_coef: 0.002
    kl_loss_type: low_var_kl
```

+ `optim`：
    - `lr`：学习率（Learning Rate）。
+ `algorithm`：
    - `adv_estimator`：Advantage 计算方法，例如 `grpo`（Group Relative Policy Optimization）。
    - `use_kl_in_reward`：是否在奖励中加入 KL 散度项。
+ `use_kl_loss`：是否在损失计算中加入 KL 约束。
+ `kl_loss_coef`：KL 损失系数。
+ `kl_loss_type`：KL 损失的计算方式，例如 `low_var_kl`。

### 调试模式
当 `backbone` 设置为 `debug` 时所使用的配置。

```yaml
astuner:
  debug:
    debug_max_parallel: 16
    debug_first_n_tasks: 2
    debug_vllm_port: 18000
    debug_vllm_seed: 12345
    debug_tensor_parallel_size: 4
```

该模式和配置可用于：

- **限制任务数与并发数**：在少量任务和较小并发下快速验证训练流程。
- **固定随机性**：通过 `debug_vllm_seed` 帮助复现问题。

## 日志与训练监控
### 配置 Logger
AgentScope Tuner 支持多种日志后端，可通过 `trainer_common.logger` 列表进行配置：

+ `console`：将日志输出到标准输出，方便快速查看训练进度。
+ `wandb`：对接 wandb 平台，提供可视化训练曲线和指标监控。

```yaml
astuner:
  trainer_common:
    logger:
      - console
      - wandb
```

### 日志结构
所有实验输出都会保存在 `./launcher_record/{experiment_name}` 目录下：

+ **Logs：** 训练过程生成的日志与错误信息。
+ **Metrics：**
    - 如果启用了 `console`，日志中会包含训练指标。
    - 如果启用了 `wandb`，训练指标、日志以及其他相关数据也会同步到云端。
+ **Checkpoint：** 训练得到的模型 checkpoint。


## 附录：完整配置示例

```yaml
# ------------------ main configuration ------------------
astuner:
  project_name: "astuner_default_project"
  experiment_name: "read_yaml_name"
  experiment_dir: "auto"  # {exp-dir}/{experiment_name}
  backbone: debug # `debug` or `trinity` or `verl`


  model:
    # which model should be trained
    path: /path/to/model/such/as/Qwen/Qwen2___5-14B-Instruct

  data:
    # max number of tokens for prompt
    max_prompt_length: 3000
    # max number of tokens for response
    max_response_length: 15000
    # how many tasks per training batch
    train_batch_size: 32
    # [Hint]: The final number of samples per update will be: N_{sample} = (data.train_batch_size * rollout.num_repeat * rollout.multi_turn.expected_steps)


  rollout:
    # activate AgentScope learn protocol
    use_agentscope_protocol: True

    # the path to the workflow class
    agentscope_learn_protocol: tutorial.example_appworld.appworld->ExampleAgentScopeLearnProtocol

    # whether or not to disable all tool calls
    agentscope_disable_toolcalls: False

    # maximum number of parallel environments / simulate workers
    max_env_worker: 128

    # step reward gamma (experimental, do not change)
    gamma: 1.0

    # monitor LLM's abormal behaviors during rollout
    compute_madness_checklist:
      - "nonsense"
    # send signal to terminate context tracing when LLM is losing control
    agent_madness_termination: True # terminate_after_gone_mad
    # punish the LLM when it is detected as lost control
    agent_madness_reward: -1.0

    # max response length in one turn
    max_response_length_in_one_turn: 4096

    # max token length allowed for the model during rollout
    max_model_len: 18000

    multi_turn:
      # how many samples should be collected for each task run
      max_sample_per_task: 30
      # limit the maximum steps for each task
      max_steps: 30
      # the expected steps for each task, used to calculate the training batch size for trinity
      expected_steps: 1

    # TP size for rollout engine
    tensor_model_parallel_size: 1

    # the number of vllm engines, number of gpus for infer is `n_vllm_engine*tensor_model_parallel_size`, this argument is NOT effective when NOT using trinity
    n_vllm_engine: 2

    # how many sequences are allowed to be processed in parallel by each vllm engine
    max_num_seqs: 10

    # the usage of infer engine, options: (vllm, sglang)
    name: vllm

    # how many times a task should be repeated
    num_repeat: 4

    # rollout kwargs
    temperature: 0.9
    top_p: 1.0

    # validation kwargs
    val_kwargs:
      temperature: 0.0
      top_k: -1
      top_p: 1.0
      do_sample: False
      num_repeat: 1


  task_reader:
    # the type of task_reader
    # options:
    #   env_service: read dataset from EnvService
    #   dataset_file: read dataset from local file
    #   huggingface_dat_repo# read dataset from huggingface repo
    type: env_service # `env_service` or `dataset_file` or `huggingface_dat_repo`
    # when `type == dataset_file`
    dataset_file:
      training:
        file_path: "/path/to/training/data.jsonl"
      validation:
        file_path: "/path/to/validation/data.jsonl"
    # when `type == env_service`
    env_service:
      # the type of env, must be init in EnvService first
      env_type: "appworld"
      # the url of the service
      env_url: "http://127.0.0.1:8080"
      # code, text, box
      env_action_preference: code
      # the name of training split in this environment
      training_split: train
      # the name of validation split in this environment
      validation_split: dev
    # when `type == huggingface_dat_repo`
    huggingface_dat_repo:
      # the repo name
      dataset_path: "gsm8k"
      # the name of training split
      training_split: "train"
      # the name of validation split
      validation_split: "validation"
  
  # task judge. it provide rewards for agent training
  task_judge:
    # options: 'customized_protocal', 'rubrics_auto_grader'
    judge_type: customized_protocal  
    # the package path to judge (reward) function
    judge_protocol: astuner.task_judge.env_service_as_judge->EnvServiceJudge

    # the helper LLM model used for LLM-AS-Judge
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
    
    # when `judge_type == rubrics_auto_grader`
    rubrics_auto_grader:
      model_name: qwen-max
      # reward mode
      grader_mode: pointwise
      # the language of prompts, tasks, llm outputs
      language: en
      # the range of score
      min_score: 0
      max_score: 1
      success_threshold: 0.7
      sampling_mode: all_samples
      generate_number: 1
      max_epochs: 2
      max_retries: 3
      aggregation_mode: keep_all
      grader_name: "auto_grader"
      num_reference_samples: 20
      query_field: main_query
      answer_field: final_answer
      reference_field: answer
      input_data_type: dataset_file # `env_service` or `dataset_file` or `huggingface_dat_repo`
      dataset_file:
        training:
          file_path: "tutorial/example_rm_auto_grader/rubrics_train.jsonl"


  # when backbone is `debug`, debug related configurations
  debug:
    debug_max_parallel: 16
    debug_first_n_tasks: 2
    debug_vllm_port: 18000
    debug_vllm_seed: 12345
    debug_tensor_parallel_size: 4
    

  # trainer common configurations
  trainer_common:
    # validate before the first epoch
    val_before_train: False
    # the rollout size in validation phase
    val_pass_n: 4
    # the frequency (step) of checkpoint saving
    save_freq: 20
    # the frequency (step) of test phase
    test_freq: 20
    # totol epochs to train
    total_epochs: 50
    # the number of nodes in clusters
    nnodes: 1
    # the number of gpus in each node
    n_gpus_per_node: 8
    # loggers that are enabled
    # options:
    #   console: log in the console
    #   wandb: log with wandb
    logger:
      - console
      - wandb
    # optimization algorithms
    algorithm:
      task_norm_patch: False
      adv_estimator: grpo
      use_kl_in_reward: False
    mini_batch_num: 1
    # FSDP config
    fsdp_config:
      # offload param to save gpu memory
      param_offload: True
      # offload optimizer to save gpu memory
      optimizer_offload: True
    # optimizer config
    optim:
      # learning rate
      lr: 1e-6
    # use KL loss in training
    use_kl_loss: True
    # KL loss coefficient
    kl_loss_coef: 0.002
    # type of KL loss
    kl_loss_type: low_var_kl
    ulysses_sequence_parallel_size: 1


  # context tracker protocol is valid ONLY when `use_agentscope_protocol=False`
  context_tracker:
    context_tracker_type: "linear"
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
    max_env_len: 4096


  # DO NOT EDIT, FOR ROBOT TESTING PURPOSE ONLY. NOT FOR HUMAN.
  execute_test: False        # DO NOT EDIT, FOR ROBOT TESTING PURPOSE ONLY. NOT FOR HUMAN.
  execute_testing_lambda: "" # DO NOT EDIT, FOR ROBOT TESTING PURPOSE ONLY. NOT FOR HUMAN.

```
