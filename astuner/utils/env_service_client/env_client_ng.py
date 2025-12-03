# env_client.py

import os
import random
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from loguru import logger

LOG_PATH = os.environ.get(
    "CLIENT_LOG_PATH", os.path.join(tempfile.gettempdir(), "app_logs", "error.out")
)


def retry_call(
    fn: Callable,
    max_retry: int = 3,
    min_backoff: float = 3.0,
    max_backoff: float = 10.0,
    fail_return: Any = None,
    err_prefix: str = "",
    instance_id: str | None = "",
    action_name: str = "",
):
    for i in range(max_retry):
        try:
            res = fn()
            if i > 0:
                logger.info(
                    f"{err_prefix} {action_name} [instance={instance_id}] succeed at try {i+1}/{max_retry}"
                )
            return res
        except Exception as e:
            logger.info(
                f"{err_prefix} {action_name} [instance={instance_id}] retry {i+1}/{max_retry} failed: {e}"
            )
            if i + 1 == max_retry:
                logger.exception(
                    f"{err_prefix} {action_name} [instance={instance_id}] max retries exceeded, fallback used."
                )
                raise RuntimeError("Env Service Timeout")
            wait = random.uniform(min_backoff, max_backoff)
            time.sleep(wait)
    return fail_return


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 30.0

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str | None = None,
        instance_id: str | None = None,
        messages: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # if env_type in map_env_type: env_type = map_env_type[env_type]
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.exception(
                f"[{endpoint}] _make_request failed (instance={instance_id}): {e}, data: {data}"
            )
            raise Exception(f"Request failed: {str(e)}, data: {data}")

    def get_env_profile(
        self,
        env_type: str,
        split: str = "train",
        params: Optional[dict] = None,
        max_retry: int = 1,
    ) -> List[str]:
        def call():
            # resolved_env_type = map_env_type.get(env_type, env_type)
            response = self._make_request(
                endpoint="/get_env_profile",
                env_type=env_type,
                params={"split": split, **(params or {})},
            )
            if "data" in response:
                return response["data"]
            elif "task_ids" in response:
                return response["task_ids"]
            else:
                return []

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=[],
            err_prefix="[get_env_profile]",
            action_name="get_env_profile",
        )

    def get_tools_info(
        self,
        instance_id: str,
        messages: Dict = {},
        params: Dict = {},
        max_retry: int = 3,
    ) -> Any:
        def call():
            response = self._make_request(
                endpoint="get_info",
                instance_id=instance_id,
                messages=messages,
                params=params,
            )
            return response.get("data", None)

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=None,
            err_prefix="[get_tools_info]",
            instance_id=instance_id,
            action_name="get_tools_info",
        )

    def create_instance(
        self,
        env_type: str,
        task_id: str,
        instance_id: Optional[str] = None,
        params: Optional[Dict] = None,
        max_retry: int = 3,
    ) -> dict:
        fallback = {
            "state": [
                {
                    "role": "system",
                    "content": "create query failed, this is a empty task.",
                },
                {
                    "role": "user",
                    "content": "create failed, this is a empty task,please close this task.",
                },
            ],
            "reward": 0,
            "is_terminated": False,
            "info": {
                "instance_id": instance_id or "",
                "task_id": task_id or "",
            },
        }

        def call():
            # if env_type in map_env_type: env_type = map_env_type[env_type]
            r = self._make_request(
                endpoint="create",
                env_type=env_type,
                task_id=task_id,
                instance_id=instance_id,
                params=params,
            )
            return r["data"]

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix="[create_instance]",
            instance_id=instance_id,
            action_name="create_instance",
        )

    def step(
        self,
        instance_id: str,
        action: Dict = {},
        params: Dict = {},
        max_retry: int = 3,
    ) -> dict:
        fallback = {
            "state": [
                {
                    "role": "assistant",
                    "content": "Step failed (timeout or exception),please retry",
                }
            ],
            "reward": 0,
            "is_terminated": False,
            "info": {"instance_id": instance_id or "", "task_id": ""},
        }

        def call():
            resp = self._make_request(
                endpoint="step",
                instance_id=instance_id,
                messages=action,
                params=params,
            )
            return resp["data"]

        res = retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix="[step]",
            instance_id=instance_id,
            action_name="step",
        )
        res["state"] = res["state"][0]
        return res

    def evaluate(
        self,
        instance_id: str,
        messages: Dict = {},
        params: Dict = {},
        max_retry: int = 3,
    ) -> float:
        def call():
            resp = self._make_request(
                endpoint="evaluate",
                instance_id=instance_id,
                messages=messages,
                params=params,
            )
            return resp.get("data", 0.0)

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=0.0,
            err_prefix="[evaluate]",
            instance_id=instance_id,
            action_name="evaluate",
        )

    def release_instance(self, instance_id: str, max_retry: int = 3) -> bool:
        def call():
            resp = self._make_request(endpoint="release", instance_id=instance_id)
            return resp.get("success", False)

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=False,
            err_prefix="[release_instance]",
            instance_id=instance_id,
            action_name="release_instance",
        )


# Usage example
def main():
    client = EnvClient()
    env_type = "appworld"

    # Get the task list
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")

    # Create an instance
    task_id = task_ids[0] if task_ids else None
    if not task_id:
        print("Task list is empty; cannot create an instance!")
        return
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response.get("state", [])
    print(f"Created instance {instance_id} with query: {query}")

    # Execute an action
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # Evaluate
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # Release the instance
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
