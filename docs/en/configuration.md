# Configuration

This page provides a detailed description of the configuration files for AgentScope Tuner.

---

## Overview

AgentScope Tuner uses YAML-format configuration files to set up data, algorithms, rewards, logging, and other runtime behaviors.

!!! info "Default Configuration"
    The default config is located at `ajet/default_config/astune_default.yaml`.

At a high level, a typical config contains a single root section `ajet`, which is divided into several logical parts:

<div class="key-features" markdown>

- <img src="https://api.iconify.design/lucide:clipboard-list.svg" class="inline-icon" />&nbsp;**Basic Metadata** — Project name, experiment name, experiment directory, and backbone selection
    - `project_name`, `experiment_name`, `experiment_dir`
    - `backbone`: Select training backend (`debug`, `trinity`, or `verl`)

- <img src="https://api.iconify.design/lucide:bar-chart-2.svg" class="inline-icon" />&nbsp;**Data & Reward** — How to load data and evaluate agents
    - `task_reader`: Load training/validation samples
    - `task_judge`: Evaluate agents and compute rewards
    - `data`: Prompt/response length and batch sizes

- <img src="https://api.iconify.design/lucide:bot.svg" class="inline-icon" />&nbsp;**Model & Rollout** — Model configuration and agent interaction
    - `model`: Base model to train
    - `rollout`: Agent-environment interaction settings
    - `context_tracker`: Conversation/history management

</div>

---

## Model Configuration

### Specifying the Model

```yaml title="config.yaml"
ajet:
  model:
    path: path/to/model
```

| Source Type | Example |
|-------------|---------|
| **Local file** | `/mnt/data/models/Qwen2.5-14B-Instruct` |
| **HuggingFace repo** | `Qwen/Qwen2.5-14B-Instruct` (auto-downloaded) |

### Environment Variables for LLM-as-Judge

If using LLM-as-a-Judge, configure these environment variables:

```bash
# DashScope API key for remote LLM calling
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
```

---

## Data Configuration

### Task Reader

`task_reader` defines how to read training and validation data.

=== "EnvService"

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

=== "JSONL File"

    ```yaml
    ajet:
      task_reader:
        type: jsonl_dataset_file
        jsonl_dataset_file:
          training:
            file_path: "data/train.jsonl"
          validation:
            file_path: "data/val.jsonl"
    ```

=== "HuggingFace"

    ```yaml
    ajet:
      task_reader:
        type: huggingface_dat_repo
        huggingface_dat_repo:
          dataset_path: "gsm8k"
          training_split: "train"
          validation_split: "validation"
    ```

### Task Judge

`task_judge` evaluates agent performance and calculates rewards.

```yaml title="config.yaml"
ajet:
  task_judge:
    judge_type: customized_protocol  # or 'rubrics_auto_grader'
    judge_protocol: ajet.task_judge.env_service_as_judge->EnvServiceJudge
    alien_llm_model: qwen3-235b-a22b-instruct-2507
    alien_llm_response_length: 512
```

| Option | Description |
|--------|-------------|
| `customized_protocol` | Use a custom Python class for scoring |
| `rubrics_auto_grader` | Use LLM-based automatic grading |

---

## Training Configuration

### Backend Selection

AgentScope Tuner supports three training backends:

| Backend | Description |
|---------|-------------|
| **trinity** | Default. Flexible and scalable framework for RL fine-tuning |
| **verl** | Volcano Engine reinforcement learning for LLMs |
| **debug** | Allows breakpoint debugging in IDEs |

```yaml title="config.yaml"
ajet:
  backbone: trinity  # debug, trinity, or verl
```

### Rollout Configuration

Controls agent behavior during environment interaction:

```yaml title="config.yaml"
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

| Parameter | Description |
|-----------|-------------|
| `agentscope_workflow` | Path to workflow implementation class |
| `temperature` / `top_p` | Sampling parameters |
| `name` | Inference engine (e.g., `vllm`) |
| `n_vllm_engine` | Number of vLLM engines (Trinity only) |

### Common Training Parameters

```yaml title="config.yaml"
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
    fsdp_config:
      param_offload: True
      optimizer_offload: True
```

| Parameter | Description |
|-----------|-------------|
| `total_epochs` | Total training epochs |
| `save_freq` | Checkpoint save frequency (steps) |
| `test_freq` | Validation frequency (steps) |
| `nnodes` / `n_gpus_per_node` | Distributed training setup |
| `fsdp_config` | FSDP memory optimization |

### Optimization Algorithms

```yaml title="config.yaml"
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

| Parameter | Description |
|-----------|-------------|
| `adv_estimator` | Advantage estimator (e.g., `grpo`) |
| `lr` | Learning rate |
| `use_kl_loss` | Include KL divergence in loss |
| `kl_loss_coef` | KL loss coefficient |

---

## Debug Mode

When `backbone: debug`, additional settings are available:

```yaml title="config.yaml"
ajet:
  debug:
    debug_max_parallel: 16
    debug_first_n_tasks: 2
    debug_vllm_port: 18000
    debug_vllm_seed: 12345
    debug_tensor_parallel_size: 4
```

!!! tip "Debug Mode Use Cases"
    - **Limiting tasks**: Quickly verify the pipeline on a few tasks
    - **Fixing randomness**: `debug_vllm_seed` helps reproduce issues
    - **Reduced parallelism**: Easier to debug with smaller concurrency

---

## Logging & Monitoring

### Logger Selection

```yaml title="config.yaml"
ajet:
  trainer_common:
    logger: swanlab  # console, wandb, or swanlab
```

| Logger | Description |
|--------|-------------|
| `console` | Standard output for quick progress checking |
| `wandb` | Weights & Biases experiment tracking |
| `swanlab` | SwanLab logging |

### Output Structure

All experiment outputs are saved in `./launcher_record/{experiment_name}`:

| Directory | Contents |
|-----------|----------|
| **Logs** | Logs and error messages |
| **Metrics** | Training metrics (depends on logger) |
| **Checkpoint** | Model checkpoints |

---

## Full Configuration Example

??? example "Complete Configuration Template"
    ```yaml title="config.yaml"
    ajet:
      project_name: "astuner_default_project"
      experiment_name: "read_yaml_name"
      experiment_dir: "auto"
      backbone: debug

      model:
        path: /path/to/model/Qwen2.5-14B-Instruct

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
        env_service:
          env_type: "appworld"
          env_url: "http://127.0.0.1:8080"
          env_action_preference: code
          training_split: train
          validation_split: dev

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

---

## Next Steps

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">See all configurations applied in a real training example.</p></a>
<a href="../visualization/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-line.svg" class="card-icon card-icon-general" alt=""><h3>Visualization</h3></div><p class="card-desc">Monitor and visualize your training progress.</p></a>
</div>
