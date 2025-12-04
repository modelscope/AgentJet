# Tracing-Feedback Training

ASTune allows you to recycle the chat logs generated during an Agent's execution and continuously improve the Agent through iterative training, which we call **Tracing-Feedback Training**. It provides features

+ Loading tracing log from agentscope studio database
+ Converting log into formatted data
+ Filtering high-quality samples with custom rubrics/filters
+ Packing samples into datasets for iterative training


In the next section, we will demonstrate how to improve an Agent using Tracing-Feedback Training.

> **AgentScope & Studio Version Compatibility**
>
> It is recommended to use matched versions:
>
> + AgentScope (v1.0.7)
> + Studio (23eb7c0b1185486d1baca36aea0ce8b85ea9de48)
>

## Setup

To use tracing logs for training, you are expected to already have an agent built with **agentscope** running in **agentscope-studio** for some time (usually in production), which means you have

1. Written your agent with [agentscope](https://github.com/agentscope-ai/agentscope).
2. Enabled tracing module following [the doc](https://doc.agentscope.io/tutorial/task_tracing.html).
3. Deployed your agent and collected the database.

By default, agentscope-studio will store the tracing logs in  
`~/AgentScope-Studio/database.sqlite`, containing all recorded dialogues between the user and the agent.



We have prepared a demo agent in `tutorials/example_feedback_tracing/agent_deployed.py`. You can simulate the tracing log with it and get the database file.

## Start Tracing-Feedback Training

Once we have the log (`database.sqlite`), we can train a new Agent with Tracing-Feedback Training.

1. Set the `astuner.task_reader.type` parameter to `tracing` in the configuration file to enable tracing-feedback mode.
2. Configure the `astuner.task_reader.feedback_tracing` section with the database path and filtering options.
3. Configure other training parameters and Rewards as you would in a normal training workflow.

```yaml
astuner:
  # ...
  
  task_reader:
    # use tracing log as tasks
    type: tracing
    feedback_tracing:
      # path to the database
      base_url: ./tutorial/example_feedback_tracing/database.sqlite
      # path where the module will write cache
      train_output_path: ./tutorial/example_feedback_tracing/tasks.jsonl
      # the model used in filters
      alien_llm_model: qwen3-235b-a22b-instruct-2507
      alien_llm_response_length: 2048
      # filters
      filters:
        # the default filter llm_evaluate
        - type: llm_evaluate
          enabled: true
          params:
            # define your rubrics to drop any bad-quality tasks
            custom_rubrics: |
              1. Check the answer and drop the task if it does not answer or answer is wrong.
              2. Consider a response is invalid if it does not wrap the final answer in \boxed{}.
            # LLM temperature
            temperature: 0.5
            # print debug log
            print_reason: false
            max_thread: 16
```

When everything is ready, start the training with `launcher.py`.

```bash
# this launch the demo
python launcher.py --conf tutorial/example_feedback_tracing/example_feedback_tracing.yaml --backbone='trinity' --with-ray
```

After training, we can now deploy the new Agent into production and collect new logs. This workflow enables continuous improvement through iterative tracing-feedback training.

## Customize

### Filter

The module provides Filter to select high-quality samples from logs for training. Users are allowed to customize the specific rubrics of their own tasks.

To write rubrics, edit the configuration file:

```yaml
astuner:
  # ...
  
  task_reader:
    # ...
    feedback_tracing:
      # ...
      filters:
        - type: llm_evaluate
          enabled: true # enable the filter
          params:
            # define your rubrics
            custom_rubrics: |
              1. Check the answer and drop the task if it does not answer or answer is wrong.
              2. Consider a response is invalid if it does not wrap the final answer in \boxed{}.
            temperature: 0.5
            print_reason: false
            max_thread: 16
```

