# Trinity Configuration Guide 🛠️

## How to Modify Trinity Configuration in ASTuner

1. 🎯 **Recommended Method**: In most cases, you do not need to directly adjust Trinity parameters. Simply refer to and modify the upper-level `astune/default_config/astune_default.yaml` configuration file, and ASTuner will **automatically** handle parameter mapping for you.

2. ⚙️ **Special Cases**: Some Trinity tuning parameters are not yet mapped in ASTuner. You can refer to Trinity’s documentation and modify them in the following format:

```yaml
trinity:
  algorithm:
    algorithm_type: multi_step_grpo
```

3. 🚫 **Never Edit**:
   - Never edit `astune/default_config/trinity/trinity_launcher.yaml`
   - Never edit `astune/default_config/trinity/trinity_default.yaml`

## Configuration Mapping Modification 🔄

Some ASTune configurations overlap with Trinity.
You can configure mappings via the `astune/default_config/trinity/config_auto_convertion_trinity.json` file.

## Trinity Hyperparameter Quick Guide 📊

Trinity adopts a typical producer (explorer)-consumer (trainer) architecture:
- 🏭 **Producer**: Uses VLLM to generate samples
- 🧠 **Consumer**: Consumes samples to update the model
Both operate on different runtime schedules.

### Explorer Core Parameters 🔍

- `buffer.batchsize`: The minimum unit for reading task data from the dataset. Each read increments the explorer step count by 1.
- `repeat_times`: The number of repetitions per task, also the group size (G) in GRPO.
- `engine_num`: Number of VLLM engines.
- `tensor_parallel_size`: Number of GPUs occupied by each VLLM engine.
- `engine_num * tensor_parallel_size`: Total number of GPUs used by the explorer.
- `eval_interval`: Evaluation interval (in explorer steps).

### Trainer Core Parameters 🏋️

- `buffer.train_batch_size`: The minimum unit consumed from the explorer’s production queue. Each read triggers one optimization step.
- `trainer.save_interval`: Parameter save interval (in trainer steps).

### Explorer-Trainer Coordination Parameters 🤝

- `sync_interval`: Synchronization interval.
- `sync_offset`: Synchronization offset.
- `sync_style`: Synchronization method.

### Runtime Instance Analysis 📈

**Supply Side**: Explorer runs 89 steps × batch size (8) × repeat times (4) × tasks per round (≈1) = 2,848 samples.

meanwhile

**Demand Side**: Trainer runs 10 steps × training batch size (264) = 2,640 samples.

### Training Memory Control 💾

Same as VERL, control training memory with the following parameters:
- `trainer.max_token_len_per_gpu`
- `ulysses_sequence_parallel_size`
