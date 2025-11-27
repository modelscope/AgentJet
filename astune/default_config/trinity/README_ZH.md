# Trinity 配置指南 🛠️

## 如何修改 Trinity 配置

1. 🎯 **推荐方式**：在大多数情况下，您无需直接调整 Trinity 参数，只需参考并修改上层的 `astune/default_config/astune_default.yaml` 配置文件即可，
   ASTuner 会**自动**帮您完成参数映射。

2. ⚙️ **特殊情况**：部分 Trinity 调优参数目前尚未在 ASTuner 中建立映射，您可以参考 Trinity 的文档，然后通过以下形式进行修改：

```yaml
trinity:
  algorithm:
    algorithm_type: multi_step_grpo
```

3. 🚫 **永远不要编辑**：
   - 永远不要编辑 `astune/default_config/trinity/trinity_launcher.yaml`
   - 永远不要编辑 `astune/default_config/trinity/trinity_default.yaml`

## 配置映射修改 🔄

某些 ASTune 配置与 Trinity 存在重叠，
可通过 `astune/default_config/trinity/config_auto_convertion_trinity.json` 文件进行映射配置

## Trinity 超参数简明指南 📊

Trinity 采用典型的生产者（探索器）-消费者（训练器）架构：
- 🏭 **生产者**：使用 VLLM 生成样本
- 🧠 **消费者**：消耗样本更新模型
两者具有不同的运行时序

### 探索器核心参数 🔍

- `buffer.batchsize`：从数据集读取任务数据的最小单位，每次读取视为探索器步数 +1
- `repeat_times`：每个任务重复次数，也是 GRPO 中 G（分组）的大小
- `engine_num`：VLLM 引擎数量
- `tensor_parallel_size`：每个 VLLM 引擎占用的显卡数量
- `engine_num * tensor_parallel_size`：探索器使用的总显卡数量
- `eval_interval`：评估间隔（以探索器步数为单位）

### 训练器核心参数 🏋️

- `buffer.train_batch_size`：从探索器生产队列中消费的最小单位，每次读取后执行一次优化步骤
- `trainer.save_interval`：参数保存间隔（以训练器步数为单位）

### 探索器-训练器协同参数 🤝

- `sync_interval`：同步间隔
- `sync_offset`：同步偏移
- `sync_style`：同步方式

### 运行实例分析 📈

**供给端**：探索器运行 89 步 × 批次大小(8) × 重复次数(4) × 每轮任务(≈1) = 2848 个样本

与此同时，在另一边

**消费端**：训练器运行 10 步 × 训练批次大小(264) = 2640 个样本

### 训练显存控制 💾

与 VERL 相同，通过以下参数控制训练显存：
- `trainer.max_token_len_per_gpu`
- `ulysses_sequence_parallel_size`