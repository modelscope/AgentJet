# Training Visualization

Monitoring training progress through visualized metrics is essential for understanding model behavior and tuning hyperparameters effectively.

---

## Supported Visualization Tools

<div class="card-grid">
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:chart-line.svg" class="card-icon card-icon-agent" alt=""><h3>SwanLab ⭐</h3></div><p class="card-desc">Modern experiment tracking platform designed for AI research. Recommended.</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/simple-icons:weightsandbiases.svg" class="card-icon card-icon-general" alt=""><h3>WandB</h3></div><p class="card-desc">Weights & Biases experiment tracking platform.</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:console.svg" class="card-icon card-icon-tool" alt=""><h3>Console</h3></div><p class="card-desc">Simple text-based logging to standard output.</p></div>
</div>

---

## Quick Start with SwanLab

### Step 1: Configure SwanLab

Simply set the logger backend to `swanlab` in your YAML configuration:

```yaml title="config.yaml"
ajet:
  trainer_common:
    logger: swanlab
```

### Step 2: Start Training

Launch your training as usual:

```bash
ajet --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

### Step 3: View Training Curves

!!! success "Automatic Tracking"
    Once training starts, SwanLab will automatically:

    1. Track key metrics (reward, success rate, loss, etc.)
    2. Generate real-time training curves
    3. Provide a web dashboard for visualization

You can access the SwanLab dashboard through the URL printed in the training logs.

---

## Understanding Training Curves

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| **Reward** | Average reward per episode, indicating task performance |
| **Success Rate** | Percentage of successfully completed tasks |
| **Loss** | Training loss from the policy optimization algorithm |
| **Response Length** | Average length of model responses |
| **KL Divergence** | Divergence between current and reference policy |

### Interpreting the Curves

**Example Training Curve:**

![Example Training Curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

A typical reward curve shows:

| Phase | Description |
|-------|-------------|
| **Initial** | Reward may be low or unstable as the model explores |
| **Learning** | Reward gradually increases as the model learns better strategies |
| **Convergence** | Reward plateaus when the model reaches optimal performance |

!!! tip "What to Look For"
    - **Rising trend**: Indicates successful learning
    - **Plateaus**: May indicate convergence or need for hyperparameter adjustment
    - **Sudden drops**: Could signal instability or overfitting

---

## Best Practices

### Monitor Multiple Runs

Compare different hyperparameter settings by running multiple experiments and comparing their curves side-by-side.

### Set Appropriate Logging Frequency

Balance between logging detail and training overhead:

```yaml title="config.yaml"
ajet:
  trainer_common:
    log_freq: 1  # Log every N steps
```

### Save Checkpoints at Key Points

Configure checkpoint saving to preserve models at peak performance:

```yaml title="config.yaml"
ajet:
  trainer_common:
    save_freq: 100  # Save every 100 steps
```

---

## Next Steps

<div class="card-grid">
<a href="../beast_logger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:bug.svg" class="card-icon card-icon-tool" alt=""><h3>Beast Logger</h3></div><p class="card-desc">Token-level debugging and visualization.</p></a>
<a href="../data_generation/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:auto-fix.svg" class="card-icon card-icon-data" alt=""><h3>Data Generation</h3></div><p class="card-desc">Auto-generate training data from documents.</p></a>
<a href="https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:book-open-variant.svg" class="card-icon card-icon-general" alt=""><h3>SwanLab Docs</h3></div><p class="card-desc">Official SwanLab documentation.</p></a>
</div>
