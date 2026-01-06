# 配置指南

本页对 AgentScope Tuner 的配置文件进行详细说明。

---

## 总览

AgentScope Tuner 使用 YAML 格式的配置文件来设置数据、训练算法、奖励、日志以及其他运行时行为。

!!! info "默认配置文件"
    默认配置文件位于 `ajet/default_config/astune_default.yaml`。

一个典型的配置文件包含一个根节点 `ajet`，进一步被划分为若干部分：

| 类别 | 配置项 | 说明 |
|------|--------|------|
| **基础信息** | `project_name`, `experiment_name`, `experiment_dir`, `backbone` | 标识实验及其保存位置，选择训练后端 |
| **数据与奖励** | `task_reader`, `task_judge`, `data` | 数据加载、评估和 batch 配置 |
| **模型** | `model` | 要训练的基础模型路径 |
| **Rollout** | `rollout`, `context_tracker` | 智能体交互配置和历史管理 |
| **训练** | `trainer_common`, `debug` | 训练超参数和调试配置 |

!!! tip "配置建议"
    您可以从默认 YAML 开始，只修改与您的使用场景相关的部分。文末*附录*提供了一个**完整配置示例**供参考。

---

## 模型配置

要训练一个智能体，首先需要指定待训练的模型：

```yaml title="model 配置"
ajet:
  model:
    path: path/to/model
```

=== "本地文件"

    指向包含 Transformers 格式模型的本地目录：

    ```yaml
    ajet:
      model:
        path: /mnt/data/models/Qwen2.5-14B-Instruct
    ```

=== "HuggingFace 仓库"

    指向 HuggingFace 仓库，模型会自动下载并加载：

    ```yaml
    ajet:
      model:
        path: Qwen/Qwen2.5-14B-Instruct
    ```

!!! warning "LLM-as-a-Judge 环境变量"
    如果在训练中使用 LLM-as-a-Judge，需要配置必要的环境变量：

    ```bash
    export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
    export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
    ```

---

## 数据配置

数据相关的配置主要包括两部分：`task_reader` 和 `task_judge`。

### Task Reader

`task_reader` 用于定义如何读取训练集和验证集。支持多种 reader 类型：

=== "EnvService"

    从 EnvService 中读取数据：

    ```yaml
    ajet:
      task_reader:
        type: env_service
        env_service:
          env_type: "appworld"
          env_url: "http://127.0.0.1:8080"
          env_action_preference: code
          training_split: train
          validation_split: dev
    ```

=== "JSONL 文件"

    从本地 JSONL 文件读取：

    ```yaml
    ajet:
      task_reader:
        type: jsonl_dataset_file
        jsonl_dataset_file:
          training:
            file_path: "xxxx.jsonl"
          validation:
            file_path: "xxxx.jsonl"
    ```

=== "HuggingFace"

    从 HuggingFace 仓库读取：

    ```yaml
    ajet:
      task_reader:
        type: huggingface_dat_repo
        huggingface_dat_repo:
          dataset_path: "gsm8k"
          training_split: "train"
          validation_split: "validation"
    ```

=== "数据生成"

    从文档自动生成任务：

    ```yaml
    ajet:
      task_reader:
        type: data_generation
        data_generation:
          document_reader:
            document_path:
              - 'dataset/document/your-document.pdf'
            languages:
              - eng
          query_reader:
            type: dataset_file
            dataset_file:
              training:
                file_path: 'dataset/jsonl/your-queries.jsonl'
          task_num: 10
          llm_model: qwen-long
    ```

### Task Judge

`task_judge` 用于评估智能体的表现并计算奖励：

```yaml title="task_judge 配置"
ajet:
  task_judge:
    judge_type: customized_protocol  # 或 'rubrics_auto_grader'
    judge_protocol: ajet.task_judge.env_service_as_judge->EnvServiceJudge
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
```

| 配置项 | 说明 |
|--------|------|
| `judge_type` | 评测方式：`customized_protocol` 或 `rubrics_auto_grader` |
| `judge_protocol` | 自定义评测器的类路径（格式：`package.module->ClassName`） |
| `alien_llm_model` | 评测时可能用到的辅助 LLM 模型 |

---

## 训练配置

### 后端选择

AgentScope Tuner 支持三种训练后端：

| 后端 | 说明 |
|------|------|
| **trinity** | 默认选项。通用、灵活且可扩展的大模型强化微调框架 |
| **verl** | Volcano engine reinforcement learning for LLMs |
| **debug** | 允许用户设置断点并调试代码 |

```yaml
ajet:
  backbone: trinity  # debug 或 trinity 或 verl
```

### Rollout 配置

`rollout` 配置控制智能体在与环境进行交互采样过程中的行为：

```yaml title="rollout 配置"
ajet:
  rollout:
    agentscope_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
    max_env_worker: 128
    temperature: 0.9
    top_p: 1.0
    name: vllm
    n_vllm_engine: 2
    num_repeat: 4
```

| 配置项 | 说明 |
|--------|------|
| `agentscope_workflow` | 具体的交互协议实现类 |
| `temperature` / `top_p` | 采样参数 |
| `name` | 推理引擎名称（例如 `vllm`） |
| `n_vllm_engine` | vLLM 引擎数量（仅 trinity 后端有效） |

### 通用训练参数

`trainer_common` 包含训练流程控制的通用参数：

```yaml title="trainer_common 配置"
ajet:
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

??? note "参数详解"
    | 配置项 | 说明 |
    |--------|------|
    | `total_epochs` | 训练总 epoch 数 |
    | `save_freq` | 保存模型 checkpoint 的频率（以 step 计） |
    | `test_freq` | 执行验证/测试的频率（以 step 计） |
    | `val_before_train` | 是否在训练开始前先执行一次验证 |
    | `val_pass_n` | 验证阶段每个问题的采样数量（Pass@N） |
    | `nnodes` / `n_gpus_per_node` | 分布式训练配置 |
    | `mini_batch_num` | 梯度累积的 mini-batch 数量 |
    | `fsdp_config` | FSDP 配置，控制参数和优化器 offload |

### 优化算法

优化算法及其超参数主要在 `algorithm`、`optim` 中设置：

```yaml title="优化算法配置"
ajet:
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

| 配置项 | 说明 |
|--------|------|
| `adv_estimator` | Advantage 计算方法（例如 `grpo`） |
| `lr` | 学习率 |
| `use_kl_loss` | 是否在损失计算中加入 KL 约束 |
| `kl_loss_coef` | KL 损失系数 |

### 调试模式

当 `backbone` 设置为 `debug` 时使用的配置：

```yaml title="debug 配置"
ajet:
  debug:
    debug_max_parallel: 16
    debug_first_n_tasks: 2
    debug_vllm_port: 18000
    debug_vllm_seed: 12345
    debug_tensor_parallel_size: 4
```

!!! tip "调试模式用途"
    - **限制任务数与并发数**：在少量任务和较小并发下快速验证训练流程
    - **固定随机性**：通过 `debug_vllm_seed` 帮助复现问题

---

## 日志与训练监控

### 配置 Logger

AgentScope Tuner 支持多种日志后端：

| 后端 | 说明 |
|------|------|
| `console` | 将日志输出到标准输出 |
| `wandb` | 对接 Weights & Biases 平台 |
| `swanlab` | 使用 SwanLab 进行日志记录 |

```yaml
ajet:
  trainer_common:
    logger: swanlab
```

### 日志结构

所有实验输出都会保存在 `./launcher_record/{experiment_name}` 目录下：

| 类型 | 说明 |
|------|------|
| **Logs** | 训练过程生成的日志与错误信息 |
| **Metrics** | 具体输出位置取决于所选的 logger 后端 |
| **Checkpoint** | 训练得到的模型 checkpoint |

---

## 附录：完整配置示例

??? example "完整配置文件"
    ```yaml
    ajet:
      project_name: "astuner_default_project"
      experiment_name: "read_yaml_name"
      experiment_dir: "auto"
      backbone: debug

      model:
        path: /path/to/model/such/as/Qwen/Qwen2___5-14B-Instruct

      data:
        max_prompt_length: 3000
        max_response_length: 15000
        train_batch_size: 32

      rollout:
        agentscope_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
        agentscope_disable_toolcalls: False
        max_env_worker: 128
        gamma: 1.0
        compute_madness_checklist:
          - "nonsense"
        agent_madness_termination: True
        agent_madness_reward: -1.0
        max_response_length_in_one_turn: 4096
        max_model_len: 18000
        multi_turn:
          max_sample_per_task: 30
          max_steps: 30
          expected_steps: 1
        tensor_model_parallel_size: 1
        n_vllm_engine: 2
        max_num_seqs: 10
        name: vllm
        num_repeat: 4
        temperature: 0.9
        top_p: 1.0
        val_kwargs:
          temperature: 0.0
          top_k: -1
          top_p: 1.0
          do_sample: False
          num_repeat: 1

      task_reader:
        type: env_service
        jsonl_dataset_file:
          training:
            file_path: "/path/to/training/data.jsonl"
          validation:
            file_path: "/path/to/validation/data.jsonl"
        env_service:
          env_type: "appworld"
          env_url: "http://127.0.0.1:8080"
          env_action_preference: code
          training_split: train
          validation_split: dev
        huggingface_dat_repo:
          dataset_path: "gsm8k"
          training_split: "train"
          validation_split: "validation"

      task_judge:
        judge_type: customized_protocol
        judge_protocol: ajet.task_judge.env_service_as_judge->EnvServiceJudge
        alien_llm_model: qwen3-235b-a22b-instruct-2507
        alien_llm_response_length: 512

      debug:
        debug_max_parallel: 16
        debug_first_n_tasks: 2
        debug_vllm_port: 18000
        debug_vllm_seed: 12345
        debug_tensor_parallel_size: 4

      trainer_common:
        val_before_train: False
        val_pass_n: 4
        save_freq: 20
        test_freq: 20
        total_epochs: 50
        nnodes: 1
        n_gpus_per_node: 8
        logger: swanlab
        algorithm:
          adv_estimator: grpo
          use_kl_in_reward: False
        mini_batch_num: 1
        fsdp_config:
          param_offload: True
          optimizer_offload: True
        optim:
          lr: 1e-6
        use_kl_loss: True
        kl_loss_coef: 0.002
        kl_loss_type: low_var_kl
        ulysses_sequence_parallel_size: 1
        checkpoint_base_dir: ./saved_checkpoints

      context_tracker:
        context_tracker_type: "linear"
        alien_llm_model: qwen3-235b-a22b-instruct-2507
        alien_llm_response_length: 512
        max_env_len: 4096
    ```
