# 训练可视化

通过可视化指标监控训练进度，对于理解模型行为、以及更高效地调参至关重要。AgentScope Tuner 支持多种可视化后端，可用于实时跟踪训练曲线、奖励趋势以及其他关键指标。

---

### 1. 支持的可视化工具

AgentScope Tuner 支持以下可视化后端：

* **SwanLab**（推荐）：面向 AI 研究的现代化实验跟踪平台。[了解 SwanLab](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)
* **WandB**：Weights & Biases 实验跟踪平台
* **TensorBoard**：TensorFlow 提供的传统可视化工具
* **Console**：将日志以纯文本形式输出到标准输出

---

### 2. SwanLab 快速开始

#### 2.1 配置 SwanLab

只需在 YAML 配置中将 logger 后端设置为 `swanlab`：

```yaml
astuner:
  trainer_common:
    logger: swanlab
```

#### 2.2 启动训练

像平时一样启动训练：

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

#### 2.3 查看训练曲线

训练启动后，SwanLab 会自动：

1. 记录关键指标（reward、success rate、loss 等）
2. 生成实时训练曲线
3. 提供可视化的 Web Dashboard

你可以通过训练日志中打印的 URL 打开 SwanLab 面板，或直接访问 SwanLab 的 Web 界面查看你的实验记录。

---

### 3. 理解训练曲线

#### 3.1 需要重点关注的指标

训练过程中通常会跟踪以下指标：

* **Reward**：每个 episode 的平均奖励，反映任务表现
* **Success Rate**：任务成功完成的比例
* **Loss**：策略优化算法的训练损失
* **Response Length**：模型回复的平均长度
* **KL Divergence**：当前策略与参考策略之间的 KL 散度

#### 3.2 如何解读曲线

**Reward 曲线：**

![Example Training Curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

一条典型的 reward 曲线通常包含：

* **初始阶段**：模型仍在探索，reward 可能较低或波动较大
* **学习阶段**：随着策略变好，reward 逐步上升
* **收敛阶段**：当模型接近最优表现时，reward 开始趋于平稳（平台期）

**建议重点观察：**

* **持续上升趋势**：通常意味着学习有效
* **平台期**：可能表示已经收敛，或需要调整超参数以进一步提升
* **突然下降**：可能是训练不稳定、或出现过拟合等问题信号

---

### 4. 最佳实践

#### 4.1 对比多次实验

通过多次实验对比不同超参数设置的效果，并在 SwanLab 或 WandB 中将曲线并排比较，能更高效地定位有效配置。

#### 4.2 设置合适的日志频率

在日志细节与训练开销之间取得平衡：

```yaml
astuner:
  trainer_common:
    # 每 N step 记录一次日志
    log_freq: 1
```

#### 4.3 在关键位置保存 Checkpoint

配置 checkpoint 保存策略，以保留峰值表现阶段的模型：

```yaml
astuner:
  trainer_common:
    save_freq: 100  # 每 100 step 保存一次
```

---

### 5. 了解更多

关于可视化与监控的更多信息：

* [SwanLab Documentation](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)
* [Configuration Guide](./configuration.md#logging--monitoring)
* [Beast-Logger Usage](./beast_logger.md) - token-level 调试可视化
