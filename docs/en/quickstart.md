# Quick Start

AgentScope Tuner provides a complete feature set for tuning agents. You can try starting training an agent right away:

```bash
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```


### Minimum Example

Let's begin with the simplest example: a math agent with a tool call.

- First, please check out the [installation guide](./installation.md) to set up the training environment.
- Then, tune your first model using the minimum example below (suppose you have written an Agent called `MathToolWorkflow`).
  ```python
  from agentscope_tuner import AstunerJob
  from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow
  model_path = "YOUR_MODEL_PATH"
  job = AstunerJob(n_gpu=8, algorithm='grpo', model=model_path)
  job.set_workflow(MathToolWorkflow)
  job.set_data(type="hf", dataset_path='openai/gsm8k')
  # [Optional: Save yaml file for manual adjustment]  job.dump_job_as_yaml('saved_experiments/math.yaml')
  # [Optional: Load yaml file from manual adjustment] job.load_job_from_yaml('saved_experiments/math.yaml')
  tuned_model = job.tune()  # Equivalent to `astuner --conf ./saved_experiments/math.yaml` in the terminal
  ```

### Explore Examples

Explore our rich library of examples to kickstart your journey:

- 🔢 [**Training a math agent that can write python code**](./example_math_agent.md).
- 📱 [**Creating an AppWorld agent using AgentScope and training it**](./example_app_world.md).
- 🐺 [**Developing Werewolves RPG agents and training them**](./example_werewolves.md).
- 👩🏻‍⚕️ [**Learning to ask questions like a doctor**](./example_learning_to_ask.md).
- 🎴 [**Writing a countdown game using AgentScope and solving it**](./example_countdown.md).
- 🚶 [**Solving a frozen lake walking puzzle using ASTuner**](./example_frozenlake.md).


### Tune Your First Agent From Scratch

Begin to build your own agent and tune it following our document:

- 📚 [**Tune Your First Agent**](./tune_your_first_agent.md).
