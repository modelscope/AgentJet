# 训练可视化

通过可视化指标监控训练进度，对于理解模型行为、以及更高效地调参至关重要。AgentScope Tuner 支持多种可视化后端，可用于实时跟踪训练曲线、奖励趋势以及其他关键指标。

---

## 支持的可视化工具

<div class="card-grid">
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-line.svg" class="card-icon card-icon-agent" alt=""><h3>SwanLab</h3></div><p class="card-desc">面向 AI 研究的现代化实验跟踪平台（推荐）。</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-areaspline.svg" class="card-icon card-icon-tool" alt=""><h3>WandB</h3></div><p class="card-desc">Weights & Biases 实验跟踪平台。</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-bar.svg" class="card-icon card-icon-data" alt=""><h3>TensorBoard</h3></div><p class="card-desc">TensorFlow 提供的传统可视化工具。</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:console.svg" class="card-icon card-icon-general" alt=""><h3>Console</h3></div><p class="card-desc">将日志以纯文本形式输出到标准输出。</p></div>
</div>

---

## SwanLab 快速开始

### 配置 SwanLab

只需在 YAML 配置中将 logger 后端设置为 `swanlab`：

```yaml
astuner:
  trainer_common:
    logger: swanlab
```

### 启动训练

像平时一样启动训练：

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

### 查看训练曲线

!!! success "自动记录"
    训练启动后，SwanLab 会自动：
    
    1. 记录关键指标（reward、success rate、loss 等）
    2. 生成实时训练曲线
    3. 提供可视化的 Web Dashboard

您可以通过训练日志中打印的 URL 打开 SwanLab 面板，或直接访问 SwanLab 的 Web 界面查看您的实验记录。

---

## 理解训练曲线

### 需要重点关注的指标

| 指标 | 说明 |
|------|------|
| **Reward** | 每个 episode 的平均奖励，反映任务表现 |
| **Success Rate** | 任务成功完成的比例 |
| **Loss** | 策略优化算法的训练损失 |
| **Response Length** | 模型回复的平均长度 |
| **KL Divergence** | 当前策略与参考策略之间的 KL 散度 |

### 如何解读曲线

**Reward 曲线示例：**

<div align="center">
<img width="600" alt="Example Training Curve" src="https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png"/>
</div>

!!! info "典型的 Reward 曲线阶段"
    | 阶段 | 特征 |
    |------|------|
    | **初始阶段** | 模型仍在探索，reward 可能较低或波动较大 |
    | **学习阶段** | 随着策略变好，reward 逐步上升 |
    | **收敛阶段** | 当模型接近最优表现时，reward 开始趋于平稳（平台期） |

**建议重点观察：**

- <img src="https://api.iconify.design/lucide:trending-up.svg" class="inline-icon" /> **持续上升趋势**：通常意味着学习有效
- <img src="https://api.iconify.design/lucide:minus.svg" class="inline-icon" /> **平台期**：可能表示已经收敛，或需要调整超参数以进一步提升
- <img src="https://api.iconify.design/lucide:trending-down.svg" class="inline-icon" /> **突然下降**：可能是训练不稳定、或出现过拟合等问题信号

---

## 最佳实践

### 对比多次实验

通过多次实验对比不同超参数设置的效果，并在 SwanLab 或 WandB 中将曲线并排比较，能更高效地定位有效配置。

### 设置合适的日志频率

在日志细节与训练开销之间取得平衡：

```yaml
astuner:
  trainer_common:
    log_freq: 1  # 每 N step 记录一次日志
```

### 在关键位置保存 Checkpoint

配置 checkpoint 保存策略，以保留峰值表现阶段的模型：

```yaml
astuner:
  trainer_common:
    save_freq: 100  # 每 100 step 保存一次
```

---

## 了解更多

<div class="card-grid">
<a href="https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:book-open-page-variant.svg" class="card-icon card-icon-agent" alt=""><h3>SwanLab 文档</h3></div><p class="card-desc">SwanLab 官方文档和教程。</p></a>
<a href="./configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-tool" alt=""><h3>配置指南</h3></div><p class="card-desc">完整的配置选项参考。</p></a>
<a href="./beast_logger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:bug.svg" class="card-icon card-icon-data" alt=""><h3>Beast-Logger</h3></div><p class="card-desc">Token 级调试可视化工具。</p></a>
</div>
