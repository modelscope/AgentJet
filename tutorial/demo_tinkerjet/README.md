# TinkerScript


TinkerScript is an experimental component of AgentJet,
allowing users to
- run, debug and train **full-weight** LLM model behind user-defined LLM workflows in **machines without GPU**.

Similar to Tinker & Open-Tinker, the basic idea behind TinkerScript is to:
- use remote (or cloud) GPU machine(s) as computation media.

However, TinkerScript goes even further on this path:

- Users only need to write and run their agents in a big `while` loop (e.g., in their laptop), and provide samples generated in this process.

- TinkerScript will take care of everything else.

- TinkerScript trains **full-weight** LLM model instead of lora.

- Upon the termination of the training session, user can call `download_tuned_model` to download tuned LLM(s).


# Core Training Code

The core code at user-side is as simple as:

```python

# step 1: ... write user-defined `execute_agent`
# step 2: ... init `tinkerjet_remote` to handshake with remote GPU server
# step 3: ... define hyper-parameters `NUM_EPOCH`, `GRPO_N`
# step 4: ... spawn `dataset` from dataset file

# step 5: rock & roll
## rollout
def rollout(task):
    try:
        api_baseurl_key = tinkerjet_remote.begin_episode()
        workflow_output = execute_agent(task, api_baseurl_key)
        tinkerjet_remote.end_episode(workflow_output)
        return workflow_output.reward
    except Exception as e:
        print(f"Episode abandoned")
        return 0.0
## Main Training loop
for epoch in range(NUM_EPOCH):
    for task in dataset.get_training_tasks():
        for i in range(GRPO_N):
            reward = rollout(task)
            print(f"{epoch}-{task}-run:{i}-{reward}")

# step 6: get trained model and shutdown
tuned_model_checkpoint = tinkerjet_remote.download_tuned_model()
tinkerjet_remote.close()

```

# Limitation

- Users are only limited to use OpenAI `baseurl` + `apikey` to build applications. Features such as `tuner.as_agentscope_model` is no longer available.

- AgentJet are not able to explicitly distinguish different agents in multi-agent scenario.
  But **do not worry**, AgentJet will still try its best to recognize shards of llm timelines and merge them behind the curtain, automatically.

- TinkerScript does not support prompt tuning.
