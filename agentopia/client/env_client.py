# env_client.py
from typing import Dict, List, Any

import requests


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 300.0

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict:
        """统一的请求处理方法"""
        url = f"{self.base_url}/{endpoint}"
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
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}, data: {data}")

    def get_task_ids(
        self, env_type: str, split: str = "train", params: dict | None = None
    ) -> List[str]:
        """获取任务ID列表"""
        payload: dict = {"env_type": env_type}
        if params:
            payload["params"] = params
        response = self._make_request(
            endpoint="get_task_ids", env_type=env_type, params={"split": split}
        )
        return response["data"]

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """获取环境信息"""
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> dict:
        """创建环境实例"""
        response = self._make_request(
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> dict:
        """执行环境步骤"""
        response = self._make_request(
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """评估环境实例"""
        response = self._make_request(
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """释放环境实例"""
        response = self._make_request(endpoint="release", instance_id=instance_id)
        return response["success"]


# 使用示例
def main():
    client = EnvClient()

    env_type = "appworld"
    # 获取任务列表
    task_ids = client.get_task_ids(env_type)
    print(f"Available tasks: {task_ids}")

    # 创建实例
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # 执行动作
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # 评估
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # 释放实例
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
