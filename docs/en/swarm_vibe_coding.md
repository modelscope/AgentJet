# Vibe Coding with AgentJet Swarm

AgentJet Swarm client is so simple that even LLMs can tune model using its APIs.

Here is an example:

```txt
Your task:

- Write an intelligent agent that learns the CountDown task (You are an agent specialized in solving countdown number puzzles. Given a target number and a list of source numbers, find a way to reach the target number using basic arithmetic operations (+, -, *, /). Each source number can only be used once.)
- I hope to use the base model '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'
- Train using 8 GPUs
- Batch Size 16
- I currently do not have a dataset, you need to help me mock a small amount of data for testing

Your skills (First read the SKILL file to acquire necessary knowledge):
ajet/copilot/write-swarm-client/SKILL.md
```

## Instruction

Copy and paste the prompt above into opencode or claude-code, and then hit `ajet-swarm start` and `python /path/to/ai/generated/agent_roll.py`,
and wait for the training to finish.

## Generated training code structure.

```bash
tutorial/opencode_build_countdown_agent
├── agent_roll.py
├── agent_run.py
├── countdown_dataset
│   ├── examples.json
│   └── train.jsonl
├── generate_countdown_dataset.py
├── __init__.py
└── readme.md

2 directories, 10 files
```

## Reference result:

<div align="center">
<img width="600" alt="image" src="https://img.alicdn.com/imgextra/i2/O1CN01u5JHH521QRGeQAFsL_!!6000000006979-2-tps-1200-600.png"/>
</div>




