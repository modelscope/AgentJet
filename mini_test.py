from ajet import AgentJetJob
from tutorial.example_math_agent.math_agent_simplify import MathToolWorkflow

model_path = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-1___5B-Instruct"
job = AgentJetJob(backbone="verl", n_gpu=2, algorithm="grpo", model=model_path)
job.set_workflow(MathToolWorkflow, ensure_reward_in_workflow=True)
job.set_data(type="hf", dataset_path="openai/gsm8k")
# [Optional] job.dump_job_as_yaml('./saved_experiments/math.yaml')   # Save yaml file for manual adjustment
# [Optional] job.load_job_from_yaml('./saved_experiments/math.yaml') # Load yaml file from manual adjustment

# Equivalent to `ajet --conf ./saved_experiments/math.yaml` in the terminal
tuned_model = job.tune()
