import re
import threading
import requests
from loguru import logger
from textwrap import dedent
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.weight_tuner.experimental.as_tinkerscript_client import TinkerScriptClient
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet import WorkflowOutput
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff
from concurrent.futures import ThreadPoolExecutor

# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # grpo group size
LOCAL_NUM_EPOCH = 10000
LOCAL_MAX_PARALLEL = 32
LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/dataset/openai/gsm8k/main"
REMOTE_TINKERJET_URL = "http://localhost:10086" # Change to your tinkerscript remote url

# --------- configurations that take effect remotely -------------
REMOTE_ALLOCATE_GPU_PER_NODE = 4
REMOTE_TRAIN_MODEL_01 = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'



class WeightUpdatedHalfway(Exception):
    """Raised when the remote side starts updating model weights halfway through an episode."""


def main():

    # Handshake with tinkerscript remote, then send training param to tinkerscript remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = LOCAL_DATASET_PATH
            )
        )
    )

    # Hand shake with remote tinkerscript server
    tinkerscript_remote = TinkerScriptClient(REMOTE_TINKERJET_URL)
    tinkerscript_remote.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_TRAIN_MODEL_01,
            grpo_n=LOCAL_GRPO_N,
        )
    )

    # tinkerscript_remote = connect_to_tinkerscript_server(sync_train_config=False, start_engine=False)
    submit_sem = threading.BoundedSemaphore(LOCAL_MAX_PARALLEL)

    # Define rollout
    def rollout(task):
        try:
            group_reward = []
            for i in range(LOCAL_GRPO_N):
                # begin episode
                episode_uuid, api_baseurl_key = tinkerscript_remote.begin_episode()
                # execute agent
                workflow_output = execute_agent(task, api_baseurl_key)
                # report output back to tinkerscript remote
                tinkerscript_remote.end_episode(episode_uuid, workflow_output)
                # collect reward
                group_reward.append(workflow_output.reward)
            print(f"Group reward mean & std: {sum(group_reward)/len(group_reward)} +/- { (max(group_reward)-min(group_reward))/2 }")
        except Exception as e:
            logger.exception("Exception during rollout:", e)
        finally:
            submit_sem.release()

    # Main Training loop
    with ThreadPoolExecutor(max_workers=LOCAL_MAX_PARALLEL) as executor:
        for epoch in range(LOCAL_NUM_EPOCH):
            for task in dataset.get_training_tasks():
                print(f"Submitting task for epoch {epoch}")
                submit_sem.acquire()
                executor.submit(rollout, task)


    # model_path = tinkerscript_remote.download_latest_model(path='./tinkerscript_saved_model')

    # Get tuned model from tinkerscript remote
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
    final_answer = response.json()['choices'][0]['message']['content']
    print(final_answer)
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
