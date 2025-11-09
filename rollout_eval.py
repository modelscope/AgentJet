import os
import sys
import hydra
import json
import time
import pickle

from openai import OpenAI
from types import SimpleNamespace
from beast_logger import print_dict
from agentopia.schema.task import Task
from agentopia.parallel_env import ParallelEnvManager
from agentopia.backbone_verl.trainer import BeyondAgentRayPPOTrainer
from agentopia.utils.process_dataset import create_rl_dataset, create_rl_sampler

class TokenAndProb:

    def __init__(self, t):
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        self.token_id = int(t.token.split('token_id:')[-1])
        self.logprob = t.logprob
        try:
            self.decoded_string = bytes(t.bytes).decode('utf-8')
        except:
            self.decoded_string = '<cannot decode>' + str(t.bytes)

class ChatCompletionScheduler():

    def __init__(self, url, config):
        from transformers import AutoTokenizer
        self.url = url
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.actor_rollout_ref.model.path)
        self.chat_scheduler = SimpleNamespace(
            model_name="dummy-model-name",
            weighted_addresses="dummy-weighted-addresses",
            completion_callback=SimpleNamespace(tokenizer=self.tokenizer),
        )

    def submit_chat_completions(self, messages, sampling_params, request_id):
        client = OpenAI(
            base_url=self.url,
            api_key="token-abc123",
        )
        response_length_eps = 6 # 减少几个token给lm_start等special token的后续处理留余地
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.config.actor_rollout_ref.rollout.response_length - response_length_eps,
            # min_tokens=1,   # 必须至少输出1个token： OpenAI API不支持min_tokens参数
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            repetition_penalty=1.0,
            top_p=self.config.actor_rollout_ref.rollout.top_p
        )
        sampling_params["temperature"] = self.config.actor_rollout_ref.rollout.val_kwargs.temperature
        sampling_params["top_k"] = self.config.actor_rollout_ref.rollout.val_kwargs.top_k
        sampling_params["top_p"] = self.config.actor_rollout_ref.rollout.val_kwargs.top_p
        sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})
        assert sampling_params["temperature"] == 0
        assert sampling_params["n"] == 1
        assert sampling_params["top_p"] == 1.0
        completion = client.chat.completions.create(
            model=self.config.actor_rollout_ref.model.path,
            messages=messages,
            extra_body=sampling_params
        )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message: message["content"] = ""
        t = {"role": message["role"], "request_id":completion.id, "content": message['content'], "tokens": [TokenAndProb(t) for t in completion.choices[0].logprobs.content]}
        messages.append(t)
        return messages

def objdump(obj, file="eval_norm_pass_4_newnew.tmp"):
   with open(file, "wb+") as f:
      pickle.dump(obj, f)
   return

def objload(file="eval_norm_pass_4_newnew.tmp"):
   import os
   if not os.path.exists(file):
      return
   with open(file, "rb") as f:
      return pickle.load(f)

def run(config, hf_modelpath):
    # --------- fast adjustment for debugging ---------
    max_parallel = 256
    n_task = 200
    pass_n = 8

    # --------- init ---------
    async_rollout_manager = ChatCompletionScheduler(config=config, url="http://localhost:18000/v1")
    print(f"Using tokenizer: {async_rollout_manager.tokenizer}")
    parallel_env = ParallelEnvManager(
        config=config,
        async_rollout_manager=async_rollout_manager,
        max_parallel=max_parallel,
        max_llm_retries=3,
        llm_mode="remote",
        tokenizer=async_rollout_manager.tokenizer
    )

    appworld_dataset_base = os.path.dirname(config.data.val_files)
    dev_dataset_path = os.path.join(appworld_dataset_base, "dev.parquet")
    test_normal_dataset_path = os.path.join(appworld_dataset_base, "test_normal.parquet")
    test_chanllenge_dataset_path = os.path.join(appworld_dataset_base, "test_challenge.parquet")
    test_dev_dataset = create_rl_dataset(dev_dataset_path, config.data, tokenizer=async_rollout_manager.tokenizer, processor=None, is_train=False, env_config=config.env_service)
    test_normal_dataset = create_rl_dataset(test_normal_dataset_path, config.data, tokenizer=async_rollout_manager.tokenizer, processor=None, is_train=False, env_config=config.env_service)

    tasks = [
        Task(
            task_id=str(dat['extras']['task_id']),
            query=dat['raw_prompt'],
            env_type=config.env_service.env_type
        ) for dat in test_normal_dataset
    ]
    tasks = tasks[:n_task]
    # repeat pass_n times
    tasks = tasks * pass_n

    cmts = parallel_env.rollout(tasks=tasks, mode="validate", epoch='1')
    task_results = {}
    for _cmt in cmts:
        reward = _cmt.reward_structure.raw_reward
        task_id = _cmt.task_id
        if task_id not in task_results:
            task_results[task_id] = {}
            task_results[task_id]['reward_arr'] = []
            task_results[task_id]['tag_arr'] = []
        if reward >= 1:
            _cmt.tag = "success"
        elif reward == 0:
            _cmt.tag = "failure"
        else:
            _cmt.tag = "half_success"
        task_results[task_id]['tag_arr'] += [_cmt.tag]
        task_results[task_id]['reward_arr'] += [_cmt.reward_structure.raw_reward]
        task_results[task_id]['scenario'] = task_id.split('_')[0]

    task_scenario = [task_id.split('_')[0] for task_id in task_results.keys()]
    set_scenarios = set(task_scenario)
    num_scenarios = len(set_scenarios)

    repeated_success_tasks = 0
    num_all_success_tasks = 0   # n 次实验中全部success的任务数
    num_pass_n_tasks = 0    # n 次实验中至少有一次success的任务数
    for task_id, task_outcomes in task_results.items():
        # 计算 num_all_success_tasks  # n 次实验中全部success的任务数
        # 计算 num_pass_n_tasks   # n 次实验中至少有一次success的任务数
        assert len(task_outcomes['tag_arr']) == pass_n
        if all(tag == "success" for tag in task_outcomes['tag_arr']):
            num_all_success_tasks += 1
        if any(tag == "success" for tag in task_outcomes['tag_arr']):
            num_pass_n_tasks += 1
        repeated_success_tasks += task_outcomes['tag_arr'].count("success")

    num_all_success_scenarios = 0  # 如果一个 scenario 的所有 task 都在 n 次实验中全部 success，则 num_all_success_scenarios +1
    num_pass_n_scenarios = 0  # 如果一个 scenario 的所有 task 都在 n 次实验中至少有一次 success，则 num_pass_n_scenarios +1
    repeated_num_pass_1_scenarios = 0   # 按顺序排列，如果一个 scenario 的所有 task 都在第 x 次实验中 success，则 repeated_num_pass_1_scenarios +1
    for scenario in set_scenarios:
        scenario_task_results = {task_id: task_outcomes for task_id, task_outcomes in task_results.items() if task_outcomes['scenario'] == scenario}
        # num_all_success_scenarios
        if all(all(tag == "success" for tag in task_outcomes['tag_arr']) for task_outcomes in scenario_task_results.values()):
            num_all_success_scenarios += 1
        # num_pass_n_scenarios
        if all(any(tag == "success" for tag in task_outcomes['tag_arr']) for task_outcomes in scenario_task_results.values()):
            num_pass_n_scenarios += 1
        # num_pass_1_scenarios
        for x in range(pass_n):
            if all(task_outcomes['tag_arr'][x]=='success' for task_outcomes in scenario_task_results.values()):
                repeated_num_pass_1_scenarios += 1
    target_dataset_name = "test_normal_dataset"
    num_tasks = len(task_results)
    rewards = [ _cmt.reward_structure.raw_reward for _cmt in cmts ]
    val_metrics = {
        "target dataset name": target_dataset_name,
        "pass_n": pass_n,

        "total_tasks": len(task_results),
        "num_all_success_tasks": num_all_success_tasks,
        f"num_pass_n_tasks(pass@{pass_n})": num_pass_n_tasks,

        "num_scenarios": num_scenarios,
        "num_all_success_scenarios": num_all_success_scenarios,
        f"num_pass_n_scenarios(pass@{pass_n})": num_pass_n_scenarios,

        "TGC@1":                           repeated_success_tasks / (num_tasks * pass_n),
        f"TGC@{pass_n}":                         num_pass_n_tasks / num_tasks,
        f"TGC@{pass_n}-all-pass":           num_all_success_tasks / num_tasks,
        f"SGC@1":                   repeated_num_pass_1_scenarios / (num_scenarios * pass_n),
        f"SGC@{pass_n}":                     num_pass_n_scenarios / num_scenarios,
        f"SGC@{pass_n}-all-pass":       num_all_success_scenarios / num_scenarios,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
    }
    print("Generated batch output")
    try:
        obj = objload()
        if obj is None: obj = {}
    except:
        obj = {}
    obj[config.trainer.experiment_name] = {}
    obj[hf_modelpath] = val_metrics
    objdump(obj)
    print_dict(val_metrics)


@hydra.main(config_path="config", config_name="beyond_agent_dataflow", version_base=None)
def main(config):
    from omegaconf import OmegaConf
    OmegaConf.resolve(config)
    run(config, config.trainer.hfmodelpath)
    print('all eval task done')

if __name__ == "__main__":
    main()