# LoRA Training

Use Low-Rank Adaptation (LoRA) to efficiently fine-tune large language models with reduced computational overhead.

## Overview

LoRA reduces the number of trainable parameters by decomposing weight updates into low-rank matrices, making training faster and more memory-efficient while maintaining model quality.

## Quick Start

### Configuration

Add the `lora` section to your YAML config:

```yaml title="your_config.yaml"
ajet:
  model:
    path: /path/to/your/model

  lora:
    lora_rank: 32
    lora_alpha: 32
    target_modules: all-linear
    load_format: safetensors
```

### Start Training

```bash
ajet --conf your_config.yaml --backbone='verl'
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lora_rank` | Rank of the low-rank matrices | `32` |
| `lora_alpha` | Scaling factor for LoRA weights | `32` |
| `target_modules` | Which modules to apply LoRA to | `all-linear` |
| `load_format` | Format to load LoRA weights | `safetensors` |

### Parameter Details

- **`lora_rank`**: Higher values allow more expressive adaptations but increase trainable parameters. Typical values: 8-64.
- **`lora_alpha`**: Scales LoRA contributions. Often set equal to `lora_rank`.
- **`target_modules`**: `all-linear` applies LoRA to all linear layers. You can also specify explicit module names.
- **`load_format`**: Supports `safetensors` (recommended, safe) or `pt` (PyTorch).

## Example Configurations

### Math Agent with LoRA

```yaml title="math_agent_lora.yaml"
ajet:
  project_name: math_agent_lora
  task_reader:
    type: huggingface_dat_repo
    huggingface_dat_repo:
      dataset_path: '/path/to/gsm8k'
      training_split: "train"
      validation_split: "test"

  task_judge:
    judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge

  model:
    path: /path/to/Qwen2.5-7B-Instruct

  rollout:
    user_workflow: "tutorial.example_math_agent.math_agent->ExampleMathLearn"
    temperature: 1.0
    max_env_worker: 64
    num_repeat: 6

  trainer_common:
    save_freq: 100
    test_freq: 100
    total_epochs: 100
    logger: swanlab
    val_before_train: true
    optim:
      lr: 3e-05

  lora:
    lora_rank: 32
    lora_alpha: 32
    target_modules: all-linear
    load_format: safetensors
```

## Benchmarking LoRA

Pre-configured LoRA benchmarks are available in `tests/bench/`:

- `benchmark_mathlora` - Math reasoning tasks
- `benchmark_countdownlora` - Countdown game tasks
- `benchmark_frozenlakelora` - FrozenLake tasks
- `benchmark_learn2asklora` - Learning to ask tasks
- `benchmark_appworldlora` - AppWorld tasks

Run a benchmark:

```bash
python -m pytest tests/bench/benchmark_mathlora/execute_benchmark_mathlora.py
```

## LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tune |
|--------|------|----------------|
| Trainable params | ~0.1-1% | 100% |
| GPU memory | Low | High |
| Training speed | Fast | Slow |
| Model quality | Comparable | Excellent |
| Catastrophic forgetting | Less risk | Higher risk |

## Saving and Loading

LoRA weights are saved separately and can be merged back into the base model:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
lora_model = PeftModel.from_pretrained(base_model, "lora_checkpoint_path")
merged_model = lora_model.merge_and_unload()
```

## Next Steps

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator.svg" class="card-icon card-icon-general" alt=""><h3>Math Agent</h3></div><p class="card-desc">Train a tool-using math reasoning agent.</p></a>
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-general" alt=""><h3>Tune First Agent</h3></div><p class="card-desc">Get started with AgentJet training.</p></a>
<a href="../configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-general" alt=""><h3>Configuration</h3></div><p class="card-desc">Deep dive into config options.</p></a>
</div>