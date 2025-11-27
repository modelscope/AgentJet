import atexit
import os
import sys
from types import SimpleNamespace

import hydra
from beast_logger import register_logger
from openai import OpenAI

from astuner.backbone.common_warm_up import warm_up_process
from astuner.schema.task import Task
from astuner.task_rollout.native_parallel_worker import ParallelEnvManager
from astuner.utils.sms_agent import send_train_message


class TokenAndProb:
    def __init__(self, t):
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        self.token_id = int(t.token.split("token_id:")[-1])
        self.logprob = t.logprob
        try:
            self.decoded_string = bytes(t.bytes).decode("utf-8")
        except:
            self.decoded_string = "<cannot decode>" + str(t.bytes)


class ChatCompletionScheduler:
    def __init__(self, url, config):
        from transformers import AutoTokenizer

        self.url = url
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.astuner.model.path)
        self.chat_scheduler = SimpleNamespace(
            model_name="dummy-model-name",
            weighted_addresses="dummy-weighted-addresses",
            completion_callback=SimpleNamespace(tokenizer=self.tokenizer),
        )

    def submit_chat_completions(self, messages, sampling_params, request_id, tools=[]):
        client = OpenAI(
            base_url=self.url,
            api_key="token-abc123",
        )
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.config.astuner.rollout.max_response_length_in_one_turn,
        )
        sampling_params["temperature"] = self.config.astuner.rollout.val_kwargs.temperature
        sampling_params["top_k"] = self.config.astuner.rollout.val_kwargs.top_k
        sampling_params["top_p"] = self.config.astuner.rollout.val_kwargs.top_p

        sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})

        if tools:
            completion = client.chat.completions.create(
                model=self.config.astuner.model.path,
                messages=messages,
                tools=tools,
                extra_body=sampling_params,
            )
        else:
            completion = client.chat.completions.create(
                model=self.config.astuner.model.path,
                messages=messages,
                extra_body=sampling_params,
            )

        message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

        if "content" not in message:
            message["content"] = ""

        messages.append(
            {
                "role": message["role"],
                "request_id": completion.id,
                "content": message["content"],
                "tool_calls": message.get("tool_calls", None),
                "tokens": [TokenAndProb(t) for t in completion.choices[0].logprobs.content],  # type: ignore
            }
        )
        return messages


def run(config):
    # --------- fast adjustment for debugging ---------
    warm_up_process(config)
    max_parallel = config.astuner.debug.debug_max_parallel
    n_task = config.astuner.debug.debug_first_n_tasks
    vllm_port = config.astuner.debug.debug_vllm_port

    # --------- init ---------
    async_rollout_manager = ChatCompletionScheduler(
        config=config, url=f"http://localhost:{vllm_port}/v1"
    )
    parallel_env = ParallelEnvManager(
        config=config,
        async_rollout_manager=async_rollout_manager,
        max_parallel=max_parallel,
        max_llm_retries=3,
        llm_mode="remote",
        tokenizer=async_rollout_manager.tokenizer,
    )

    from astuner.task_reader import TaskReaderRouter

    task_reader = TaskReaderRouter(config)
    tasks = task_reader.get_validation_tasks()
    print(tasks[:2])
    cmt = parallel_env.rollout(
        tasks=tasks[:n_task], mode="sample", epoch="1"
    )  # "sample" or "validate"
    gen_batch_output = parallel_env.to_dataproto(cmt)
    print("Generated batch output")


@hydra.main(
    config_path="astuner/default_config",
    config_name="astune_default",
    version_base=None,
)
def main(config):
    from omegaconf import OmegaConf

    OmegaConf.resolve(config)
    print("*" * 20)

    def companion_launch():
        import torch

        from astuner.utils.smart_daemon import LaunchCommandWhenAbsent

        print("Launching companion process for async LLM server...")
        model_path = config.astuner.model.path
        tensor_parallel_size = config.astuner.debug.debug_tensor_parallel_size
        n_avail_gpus = torch.cuda.device_count()
        if tensor_parallel_size > n_avail_gpus:
            print(
                f"Warning: tensor_parallel_size {tensor_parallel_size} is greater than available GPUs {n_avail_gpus}. Setting tensor_parallel_size to {n_avail_gpus}."
            )
            tensor_parallel_size = n_avail_gpus
        gpu_memory_utilization = config.actor_rollout_ref.rollout.gpu_memory_utilization
        max_num_seqs = config.actor_rollout_ref.rollout.max_num_seqs
        max_model_len = config.astuner.rollout.max_model_len
        seed = config.astuner.debug.debug_vllm_seed
        vllm_port = config.astuner.debug.debug_vllm_port
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                "-m",
                f"vllm.entrypoints.cli.main",
                f"serve",
                f"{model_path}",
                f"--tensor-parallel-size",
                f"{tensor_parallel_size}",
                f"--dtype",
                f"auto",
                f"--enforce-eager",
                f"--gpu-memory-utilization",
                f"{gpu_memory_utilization}",
                f"--disable-custom-all-reduce",
                f"--max-num-seqs",
                f"{max_num_seqs}",
                f"--max-model-len",
                f"{max_model_len}",
                f"--load-format",
                "auto",
                f"--enable-chunked-prefill",
                f"--enable-auto-tool-choice",
                f"--tool-call-parser",
                "hermes",
                f"--enable-prefix-caching",
                f"--seed",
                f"{seed}",
                f"--port",
                f"{vllm_port}",
            ],
            dir="./",
            tag="external_vllm_server",
        )
        companion.launch(
            launch_wait_time=1800,
            success_std_string="Application startup complete",
            env_dict={**os.environ},
        )

    companion_launch()

    run(config)


if __name__ == "__main__":
    atexit.register(lambda: send_train_message("0000"))
    main()
