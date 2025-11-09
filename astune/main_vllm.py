import os
import sys
import hydra

from openai import OpenAI
from types import SimpleNamespace
from astune.schema.task import Task
from beast_logger import register_logger
from astune.parallel_env import ParallelEnvManager

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.astune.model.path)
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
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.config.astune.rollout.max_response_length_in_one_turn,
            temperature=self.config.astune.rollout.temperature,
            top_p=self.config.astune.rollout.top_p
        )
        sampling_params["temperature"] = self.config.astune.rollout.val_kwargs.temperature
        sampling_params["top_k"] = self.config.astune.rollout.val_kwargs.top_k
        sampling_params["top_p"] = self.config.astune.rollout.val_kwargs.top_p
        sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})

        completion = client.chat.completions.create(
            model=self.config.astune.model.path,
            messages=messages,
            extra_body=sampling_params
        )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message: message["content"] = ""
        t = {"role": message["role"], "request_id":completion.id, "content": message['content'], "tokens": [TokenAndProb(t) for t in completion.choices[0].logprobs.content]}
        messages.append(t)

        return messages


def run(config):
    # --------- fast adjustment for debugging ---------
    max_parallel = config.astune.debug.debug_max_parallel
    n_task = config.astune.debug.debug_first_n_tasks
    vllm_port = config.astune.debug.debug_vllm_port

    # --------- init ---------
    async_rollout_manager = ChatCompletionScheduler(config=config, url=f"http://localhost:{vllm_port}/v1")
    parallel_env = ParallelEnvManager(
        config=config,
        async_rollout_manager=async_rollout_manager,
        max_parallel=max_parallel,
        max_llm_retries=3,
        llm_mode="remote",
        tokenizer=async_rollout_manager.tokenizer
    )

    from astune.task_reader.task_reader_base import TaskReaderRouter
    task_reader = TaskReaderRouter(config)
    tasks = task_reader.get_validation_tasks()
    print(tasks[:2])
    cmt = parallel_env.rollout(tasks=tasks[:n_task], mode="sample", epoch='1') # "sample" or "validate"
    gen_batch_output = parallel_env.to_dataproto(cmt)
    print("Generated batch output")


@hydra.main(config_path="astune/default_config", config_name="astune_default", version_base=None)
def main(config):
    from omegaconf import OmegaConf
    OmegaConf.resolve(config)
    print('*' * 20)

    def companion_launch():
        from astune.utils.smart_daemon import LaunchCommandWhenAbsent
        import torch
        print("Launching companion process for async LLM server...")
        model_path = config.astune.model.path
        tensor_parallel_size = config.astune.debug.debug_tensor_parallel_size
        n_avail_gpus = torch.cuda.device_count()
        if tensor_parallel_size > n_avail_gpus:
            print(f"Warning: tensor_parallel_size {tensor_parallel_size} is greater than available GPUs {n_avail_gpus}. Setting tensor_parallel_size to {n_avail_gpus}.")
            tensor_parallel_size = n_avail_gpus
        gpu_memory_utilization = config.actor_rollout_ref.rollout.gpu_memory_utilization
        max_num_seqs = config.actor_rollout_ref.rollout.max_num_seqs
        max_model_len = config.astune.rollout.max_model_len
        seed = config.astune.debug.debug_vllm_seed
        vllm_port = config.astune.debug.debug_vllm_port
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable, "-m",
                f"vllm.entrypoints.cli.main",
                f"serve", f"{model_path}",
                f"--tensor-parallel-size", f"{tensor_parallel_size}",
                f"--dtype", f"auto",
                f"--enforce-eager",
                f"--gpu-memory-utilization", f"{gpu_memory_utilization}",
                f"--disable-custom-all-reduce",
                f"--max-num-seqs", f"{max_num_seqs}",
                f"--max-model-len", f"{max_model_len}",
                f"--load-format", "auto",
                f"--enable-chunked-prefill",
                f"--enable-prefix-caching",
                f"--seed", f"{seed}",
                f"--port", f"{vllm_port}",
            ],
            dir='./',
            tag="external_vllm_server"
        )
        companion.launch(launch_wait_time=1800, success_std_string="Application startup complete", env_dict={**os.environ})
    companion_launch()

    run(config)

if __name__ == "__main__":
    main()