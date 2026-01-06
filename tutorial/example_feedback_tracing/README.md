# Training a New Agent from Tracing Logs

AgentJet allows you to recycle the chat logs generated during an Agent’s execution and continuously improve the Agent through iterative training.
This document demonstrates how to train an Agent using tracing log feedback.

## 1. Preparing the Data

To use tracing logs for training, you must already have an Agent built with **agentscope** running in **agentscope-studio** for some time.

In this example, we implement a math-problem-solving agent in `agent_deployed.py`.
To demonstrate the workflow, we will first simulate the data-collection process.

1. Install [agentscope-studio](https://github.com/agentscope-ai/agentscope-studio).
2. Start agentscope-studio with the default port settings.
3. Run `agent_deployed.py` and simulate user–agent conversations.

After several rounds of interaction, studio will store the tracing logs in
`~/AgentScope-Studio/database.sqlite`, containing all recorded dialogues between the user and the agent.

> **AgentScope & Studio Version Compatibility**
>
> It is recommended to use matched versions:
>
> * AgentScope (v1.0.7)
> * Studio (23eb7c0b1185486d1baca36aea0ce8b85ea9de48)

## 2. Starting Trace-Feedback Training

Once you have the tracing log (`database.sqlite`), you can use the trace-feedback training module to train a new Agent.

1. Set the `task_reader` parameter to `tracing` in the configuration file to enable trace-feedback mode.
2. Configure the `tracing` section with the database path and filtering options.
3. Configure other training parameters and Rewards as you would in a normal training workflow.

An example database and configuration file are provided under
`example_feedback_tracing/`.

When everything is ready, start the training with:

```bash
ajet --conf tutorial/example_feedback_tracing/example_feedback_tracing.yaml --backbone='trinity' --with-ray
```

## 3. Deploying the New Agent

You can now deploy the newly trained Agent into production, enabling continuous improvement through iterative trace-feedback training.
