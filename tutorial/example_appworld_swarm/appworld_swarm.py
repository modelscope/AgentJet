from typing import Any, Tuple

from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion

from ajet import WorkflowOutput
from ajet.schema.task import Task
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.env_service_client.env_client_ng import EnvClient
from ajet.utils.message_utils import is_token_overflow_message


class AppworldGymWrapper:
    """Mirror of ajet.task_rollout.resource_keeper.BaseGymEnv for swarm-mode clients.

    The swarm runner does not build a `gym_env` for us, so we wrap `EnvClient`
    directly to keep the `step()/evaluate()` surface that the agent loop expects.
    """

    def __init__(self, env_client: EnvClient, episode_uuid: str):
        self.env_client = env_client
        self.episode_uuid = episode_uuid

    def step(self, action: dict) -> Tuple[Any, float, bool, dict]:
        env_output = self.env_client.step(
            instance_id=self.episode_uuid,
            action=action,
        )
        obs: Any = ""
        reward: float = 0
        info: dict = {}
        if isinstance(env_output["state"], list):
            obs = env_output["state"]
            reward = env_output["reward"]
            info = env_output["info"]
        else:
            if ("content" not in env_output["state"]) and ("error" in env_output["state"]):
                obs = f"[Error from environment: {env_output['error']}]"
            elif env_output["state"].get("content", "") == "":
                obs = "Warning: the environment does not provide any feedback, please provide valid input and try again."
            else:
                obs = env_output["state"]["content"]
        terminate = env_output["is_terminated"]
        return obs, reward, terminate, info

    def evaluate(self, params=None):
        return self.env_client.evaluate(self.episode_uuid, params=params or {"sparse": False})


class ExampleAgentScopeWorkflow:
    """Swarm-mode appworld workflow.

    Unlike the in-process workflow (which receives a fully initialized
    `WorkflowTask` with `gym_env` populated by the framework), the swarm
    client is responsible for the env_service instance lifecycle and reward
    evaluation locally.
    """

    def __init__(
        self,
        env_url: str = "http://127.0.0.1:8080",
        env_type: str = "appworld",
        max_steps: int = 25,
    ):
        self.env_url = env_url
        self.env_type = env_type
        self.max_steps = max_steps

    async def execute(self, task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey) -> WorkflowOutput:
        episode_uuid = api_baseurl_key.episode_uuid
        env_client = EnvClient(base_url=self.env_url)

        try:
            create_response = env_client.create_instance(
                env_type=self.env_type,
                task_id=task.task_id,
                instance_id=episode_uuid,
                params={},
            )
            state_message = create_response["state"]
            if isinstance(state_message, dict):
                raw_init_messages = [state_message]
            elif isinstance(state_message, list):
                raw_init_messages = state_message
            else:
                raise ValueError(
                    f"state_message should be dict or list, got {type(state_message)}"
                )

            if len(raw_init_messages) >= 2:
                first_msg, init_messages = raw_init_messages[0], raw_init_messages[1:]
            else:
                first_msg = {"content": "You're a helpful assistant."}
                init_messages = []

            interaction_message = [
                {
                    "content": first_msg.get("content", "You're a helpful assistant."),
                    "role": "system",
                }
            ]
            for msg in init_messages:
                interaction_message.append(
                    {
                        "content": msg.get("content", ""),
                        "role": msg.get("role", "user"),
                    }
                )

            client = api_baseurl_key.as_raw_openai_sdk_client()
            env = AppworldGymWrapper(env_client, episode_uuid)
            step = 0
            for step in range(self.max_steps):
                reply_message: ChatCompletion = await client.chat.completions.create(
                    model="ajet-model",
                    messages=interaction_message,
                )
                reply_content = reply_message.choices[0].message.content
                # AgentJet signals prompt overflow via a synthetic assistant message; further turns will only push the prompt further past max_model_len,
                if is_token_overflow_message(reply_content):
                    logger.warning(f"[appworld_swarm] token overflow detected at step={step} (task_id={task.task_id}); aborting rollout.")
                    break
                obs, _, terminate, _ = env.step(
                    action={"content": reply_content, "role": "assistant"}
                )
                interaction_message.extend(
                    [
                        {
                            "content": reply_message.choices[0].message.content,
                            "role": "assistant",
                        },
                        {
                            "content": obs,
                            "role": "user",
                        }
                    ]
                )
                if terminate:
                    break

            try:
                raw_reward = env.evaluate(params={"sparse": False})
            except Exception:
                logger.exception("Evaluation failed; defaulting raw_reward=0.0")
                raw_reward = 0.0

            # mirror EnvServiceJudge.compute_reward
            if raw_reward >= 1:
                is_success = True
                final_reward = 1.0 + raw_reward * 0.5
            else:
                is_success = False
                final_reward = 0.0 + raw_reward * 0.5

            return WorkflowOutput(
                reward=final_reward,
                is_success=is_success,
                metadata={"total_step": step},
            )
        except Exception:
            logger.bind(exception=True).exception(
                f"Error during appworld swarm episode (task_id={task.task_id})."
            )
            return WorkflowOutput(reward=0.0, is_success=False, metadata={"total_step": 0})
        finally:
            try:
                env_client.release_instance(episode_uuid)
            except Exception:
                logger.exception("Failed to release env instance")
