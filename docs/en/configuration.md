# Configuration

This page provides a detailed description of the configuration files.

## Overview
AgentScope Tuner uses YAML-format configuration files to set up the data, algorithms, reward, logging and other runtime behaviors. The default config is located at `astuner/default_config/astune_default.yaml`.

At a high level, a typical config contains a single root section `astuner`, which is further divided into several logical parts:

- **Basic metadata**
  - `project_name`, `experiment_name`, `experiment_dir`: Identify where experiments are stored.
  - `backbone`: Select the training backend, e.g. `debug`, `trinity`, or `verl`.
- **Data & Reward**
  - `task_reader`: How to load training / validation samples (EnvService, local file, HuggingFace dataset, etc.).
  - `task_judge`: How to evaluate the agent and compute rewards (custom judge or LLM-based auto-grader).
  - `data`: Prompt / response length and batch sizes.
- **Model**
  - `model`: Where to load the base model to be trained (local path or HuggingFace repo).
- **Rollout & interaction**
  - `rollout`: How the agent interacts with the environment (protocol, sampling parameters, max steps, etc.).
  - `context_tracker`: How the conversation / history is managed.
- **Training loop**
  - `trainer_common`: Global training hyperparameters (epochs, checkpoints, algorithms, KL loss, FSDP, logger, etc.).
  - `debug`: Extra debug-only settings used when `backbone: debug`.

You can start from the default YAML and only modify the sections relevant to your use case. The **full configuration example** is provided in the *Appendix* at the end of this page as a reference.

## Model Registry
To train an agent, we first need to specify the model to be trained and some environment variables used during the training process.

In the configuration file, you can set the path to load model parameters from a specific location.

```yaml
astuner:
  # ...

  # model to be trained
  model:
    path: path/to/model

  # ...

```

We can use

+ Local file: Points to a local directory containing a model in transformers format, e.g., `/mnt/data/models/Qwen2.5-14B-Instruct`
+ HuggingFace repo: Points to a HuggingFace repo, e.g., `Qwen/Qwen2.5-14B-Instruct`. The model will be automatically downloaded to local and loaded.



Additionally, if LLM-as-a-Judge is used in training, some necessary environment variables need to be configured:

```bash
# the API key of DashScope, which provides the remove LLM calling
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
```

## Data
Configuring data mainly involves two parts: `task_reader` and `task_judge`.

### Task Reader
`task_reader` defines how to read training and validation data. Currently, the following three types are supported.

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

+ `env_service`: Read data from EnvService, for tasks that need to interact with the EnvService.
    - `env_type`: Environment type, must be consistent with the type initialized in EnvService (e.g., `appworld`).
    - `env_url`: The service address of EnvService (e.g., `http://127.0.0.1:8080`).
    - `env_action_preference`: The preferred format of Action, options: `code`, `text`, or `box`.
    - `training_split`: The name of the dataset split used for training in the environment.
    - `validation_split`: The name of the dataset split used for validation in the environment.
+ `dataset_file`: Read data from local files, usually in JSONL format.
    - `training.file_path`: Local file path of the training dataset.
    - `validation.file_path`: Local file path of the validation dataset.
+ `huggingface_dat_repo`: Read datasets directly from HuggingFace repositories.
    - `dataset_path`: The dataset repository name on HuggingFace (e.g., `gsm8k`).
    - `training_split`: The dataset split name for training.
    - `validation_split`: The dataset split name for validation.

### Task Judge
`task_judge` is used to evaluate Agent performance and calculate rewards.

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

+ `judge_type`: Judge type.
    - `customized_protocal`: Use a custom Python class for scoring. Need to specify the class path via `judge_protocol` (e.g., `package.module->ClassName`).
    - `rubrics_auto_grader`: Use LLM-based automatic grading.
+ `alien_llm_model`: Auxiliary LLM model that may be used during judgment.

## Training Configuration
### Backend
AgentScope Tuner supports three training backends, including **trinity** and **verl**, as well as an extra **debug** mode.

+ **trinity**: The default option. A general-purpose, flexible and scalable framework designed for reinforcement fine-tuning of LLMs.
+ **verl**: Volcano engine reinforcement learning for LLMs.
+ **debug**: A backend that allows users to set breakpoint and debug code.

To configure the backend to be used, you can modify

```yaml
astuner:
  # debug or trinity or verl
  backbone: trinity
```

### Rollout
The rollout section controls the behavior of the Agent during the interaction sampling process with the environment.

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

+ `use_agentscope_protocol`: Whether to use the interaction protocol defined by AgentScope.
+ `agentscope_learn_protocol`: Specify the specific interaction protocol implementation class.
+ `temperature` / `top_p`: Sampling parameters.
+ `name`: Inference engine name (e.g., `vllm`).
+ `n_vllm_engine`: The number of vLLM engines to use (effective when backbone is trinity).

### Context Tracker
The `context_tracker` section is used **only when** `rollout.use_agentscope_protocol=False`. It controls how the conversation history is managed and summarized.

```yaml
astuner:
  context_tracker:
    context_tracker_type: "linear"
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
    # ...
```

- `context_tracker_type`: Strategy for context management, e.g. `linear`.
- `alien_llm_model` / `alien_llm_response_length`: Auxiliary LLM and its max response length used in context tracking.

Other commented sub-sections (such as `auto_context_cm`, `sliding_window_cm`, `linear_think_cm`) can be enabled and customized in advanced use cases.

### Common Parameters
`trainer_common` contains common parameters for training flow control:

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

+ `total_epochs`: Total number of training epochs.
+ `save_freq`: Frequency of saving model checkpoints (in steps).
+ `test_freq`: Frequency of validation/testing (in steps).
+ `val_before_train`: Whether to perform a validation before training starts.
+ `val_pass_n`: Number of samples per question in validation phase (Pass@N).
+ `nnodes` / `n_gpus_per_node`: Distributed training configuration, specifying the number of nodes and GPUs per node.
+ `mini_batch_num`: Number of mini-batches for accumulation.
+ `ulysses_sequence_parallel_size`: Sequence parallel size for Ulysses attention.
+ `fsdp_config`: FSDP (Fully Sharded Data Parallel) configuration.
    - `param_offload`: Whether to offload model parameters to CPU to save GPU memory.
    - `optimizer_offload`: Whether to offload optimizer state to CPU.

### Optimization Algorithms
Optimization algorithms and hyperparameters are mainly set in `algorithm`, `optim`, and root configuration:

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

+ `optim`:
    - `lr`: Learning Rate.
+ `algorithm`:
    - `adv_estimator`: Advantage estimator, e.g., `grpo` (Group Relative Policy Optimization).
    - `use_kl_in_reward`: Whether to include KL divergence as part of the reward.
+ `use_kl_loss`: Whether to include KL divergence constraint in loss calculation.
+ `kl_loss_coef`: Coefficient for KL loss.
+ `kl_loss_type`: Calculation method of KL loss, e.g., `low_var_kl`.

### Debug Mode
When `backbone` is set to `debug`, the `debug` section controls the behavior of the debug backend.

```yaml
astuner:
  debug:
    debug_max_parallel: 16
    debug_first_n_tasks: 2
    debug_vllm_port: 18000
    debug_vllm_seed: 12345
    debug_tensor_parallel_size: 4
```

Typical usages include:

- **Limiting tasks and concurrency**: quickly verify the training pipeline on a few tasks with small parallelism.
- **Fixing randomness**: `debug_vllm_seed` helps reproduce issues.

## Logging & Monitoring
### Configure Logger
AgentScope Tuner supports multiple logging backends, configured via the `trainer_common.logger` list:

+ `console`: Standard output logs for quick progress checking.
+ `wandb`: Integrated with wandb experiment tracking platform, providing visualized training curves and metric monitoring.

```yaml
astuner:
  trainer_common:
    logger:
      - console
      - wandb
```

### Log Structure
All experiment outputs will be saved in `./launcher_record/{experiment_name}`:

+ **Logs:** Logs and error messages generated by the launcher and trainer.
+ **Metrics:**
    - If `console` is enabled, the log will contain training metrics.
    - If `wandb` is enabled, training metrics, logs, and other related data will also be synced to the cloud or specified local directory.
+ **Checkpoint:** Checkpoints of the trained models.


## Appendix: Full configuration example

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
