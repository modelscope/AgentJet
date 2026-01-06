# Quick Start

AgentScope Tuner provides a complete feature set for tuning agents. You can try starting training an agent right away:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

---

## Minimum Example

Let's begin with the simplest example: a math agent with a tool call.

<div class="workflow-single">
<div class="workflow-header">Getting Started Flow</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Set up Environment</strong>

Check out the [installation guide](./installation.md) to set up the training environment.</li>
<li><strong>Define Your Workflow</strong>

Write an Agent class (e.g., `MathToolWorkflow`) that inherits from the base Workflow class.</li>
<li><strong>Configure and Run</strong>

Use the `AstunerJob` API to configure and start training.</li>
</ol>
</div>
</div>

### Code Example

```python title="train_math_agent.py"
from agentscope_tuner import AstunerJob
from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow

model_path = "YOUR_MODEL_PATH"
job = AstunerJob(n_gpu=8, algorithm='grpo', model=model_path)
job.set_workflow(MathToolWorkflow)
job.set_data(type="hf", dataset_path='openai/gsm8k')

# [Optional] Save yaml file for manual adjustment
# job.dump_job_as_yaml('saved_experiments/math.yaml')

# [Optional] Load yaml file from manual adjustment
# job.load_job_from_yaml('saved_experiments/math.yaml')

# Start training
tuned_model = job.tune()
```

!!! tip "CLI Alternative"
    The code above is equivalent to running in terminal:
    ```bash
    astuner --conf ./saved_experiments/math.yaml
    ```

---

## Explore Examples

Explore our rich library of examples to kickstart your journey:

<div class="card-grid">
<a href="./example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="./example_app_world/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:application.svg" class="card-icon card-icon-agent" alt=""><h3>AppWorld Agent</h3></div><p class="card-desc">Creating an AppWorld agent using AgentScope and training it.</p></a>
<a href="./example_werewolves/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:wolf.svg" class="card-icon card-icon-multimodal" alt=""><h3>Werewolves Game</h3></div><p class="card-desc">Developing Werewolves RPG agents and training them.</p></a>
<a href="./example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor.</p></a>
<a href="./example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing and solving a countdown game with RL.</p></a>
<a href="./example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle.</p></a>
</div>

---

---

## Next Steps

<div class="card-grid">
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-agent" alt=""><h3>Tune Your First Agent</h3></div><p class="card-desc">Complete step-by-step guide to building your own agent from scratch.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent Example</h3></div><p class="card-desc">Detailed walkthrough of the Math Agent training example.</p></a>
</div>
