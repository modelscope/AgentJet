import re
import requests
from textwrap import dedent
from ajet.tuner_lib.weight_tuner.experimental.as_tinkerscript_client import TinkerScriptClient
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet import WorkflowOutput
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff


TINKERJET_URL = "http://localhost:10086" # Change to your tinkerjet remote url
NUM_EPOCH = 100
GRPO_N = 4  # grpo group size


class WeightUpdatedHalfway(Exception):
    """Raised when the remote side starts updating model weights halfway through an episode."""


def main():

    # Handshake with tinkerjet remote, then send training param to tinkerjet remote (such as model to be trained, algorithm, etc)
    tinkerjet_remote = TinkerScriptClient(TINKERJET_URL)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = "/mnt/data_cpfs/qingxu.fu/dataset/openai/gsm8k/main"
            )
        )
    )

    # Define rollout
    def rollout(task):
        # Q: Can I run episodes in parallel?
        # A: Yes, wrap `rollout` in a thread or process pool.
        api_baseurl_key = tinkerjet_remote.begin_episode()
        workflow_output = execute_agent(task, api_baseurl_key)
        tinkerjet_remote.end_episode(workflow_output)
        return workflow_output.reward

    # Main Training loop
    for epoch in range(NUM_EPOCH):
        for task in dataset.get_training_tasks():
            try:
                for i in range(GRPO_N):
                    reward = rollout(task)
                    print(f"{epoch}-{task}-run:{i}-{reward}")
            except WeightUpdatedHalfway as e:
                print(f"The remote side has gone into the LLM model weight update phrase halfway through an episode."
                      f"This is **normal**."
                      f"The remote no longer need this task anymore, so let's go to next task.")

    # Get tuned model from tinkerjet remote
    return None




@retry_with_backoff(max_retry=2)
def execute_agent(task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    # Prepare base_url, api_key
    base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
    # Read dataset item
    query, reference_answer = (task.main_query, task.metadata["answer"])
    # Prepare messages
    messages = [
        { "role": "system", "content": dedent("""You are an agent specialized in solving math problems. Please solve the math problem given to you.
           You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.""") },
        { "role": "user", "content": query }
    ]
    # Use raw http requests (non-streaming) to get response
    response = requests.post( f"{base_url}/chat/completions", json = { "model": "fill_whatever_model", "messages": messages, },
                               headers = { "Authorization": f"Bearer {api_key}" } )
    print(response.json())
    final_answer = response.json()['choices'][0]['message']['content']
    # Compute reward
    reference_answer = reference_answer.split("####")[-1].strip()
    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, final_answer)
    if match: is_success = match.group(1) == reference_answer
    else: is_success = False
    raw_reward = 1.0 if is_success else 0.0
    # Return
    return WorkflowOutput(reward=raw_reward, metadata={"final_answer": final_answer})


if __name__ == "__main__":
    main()
