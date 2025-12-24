# Training Visualization

Monitoring training progress through visualized metrics is essential for understanding model behavior and tuning hyperparameters effectively. AgentScope Tuner supports multiple visualization backends to track training curves, reward trends, and other key metrics in real-time.

---

### 1. Supported Visualization Tools

AgentScope Tuner supports the following visualization backends:

- **SwanLab** (Recommended): A modern experiment tracking platform designed for AI research. [Learn more about SwanLab](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)
- **WandB**: Weights & Biases experiment tracking platform
- **TensorBoard**: Traditional visualization toolkit from TensorFlow
- **Console**: Simple text-based logging to standard output

---

### 2. Quick Start with SwanLab

#### 2.1 Configure SwanLab

Simply set the logger backend to `swanlab` in your YAML configuration:

```yaml
astuner:
  trainer_common:
    logger: swanlab
```

#### 2.2 Start Training

Launch your training as usual:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

#### 2.3 View Training Curves

Once training starts, SwanLab will automatically:

1. Track key metrics (reward, success rate, loss, etc.)
2. Generate real-time training curves
3. Provide a web dashboard for visualization

You can access the SwanLab dashboard through the URL printed in the training logs, or visit the SwanLab web interface to view your experiments.

---

### 3. Understanding Training Curves

#### 3.1 Key Metrics to Monitor

The following metrics are typically tracked during training:

- **Reward**: The average reward per episode, indicating task performance
- **Success Rate**: Percentage of successfully completed tasks
- **Loss**: Training loss from the policy optimization algorithm
- **Response Length**: Average length of model responses
- **KL Divergence**: Divergence between the current policy and the reference policy

#### 3.2 Interpreting the Curves

**Reward Curve:**

![Example Training Curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

A typical reward curve shows:

- **Initial Phase**: Reward may be low or unstable as the model explores
- **Learning Phase**: Reward gradually increases as the model learns better strategies
- **Convergence**: Reward plateaus when the model reaches optimal performance

**What to look for:**

- **Rising trend**: Indicates successful learning
- **Plateaus**: May indicate convergence or need for hyperparameter adjustment
- **Sudden drops**: Could signal instability or overfitting

---

### 4. Best Practices

#### 4.1 Monitor Multiple Runs

Compare different hyperparameter settings by running multiple experiments and comparing their curves side-by-side in SwanLab or WandB.

#### 4.2 Set Appropriate Logging Frequency

Balance between logging detail and training overhead:

```yaml
astuner:
  trainer_common:
    # Log every N steps
    log_freq: 1
```

#### 4.3 Save Checkpoints at Key Points

Configure checkpoint saving to preserve models at peak performance:

```yaml
astuner:
  trainer_common:
    save_freq: 100  # Save every 100 steps
```

---

### 5. Learn More

For more detailed information about visualization and monitoring:

- [SwanLab Documentation](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)
- [Configuration Guide](./configuration.md#logging--monitoring)
- [Beast-Logger Usage](./beast_logger.md) - Token-level debugging visualization
