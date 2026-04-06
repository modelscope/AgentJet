"""
AppWorld React Agent Implementation

This module implements a React-style agent for learning to interact with AppWorld environment.
The agent uses the OpenAI SDK to communicate with the model and performs multi-turn interactions
with the AppWorld environment.
"""

import os
import re
from typing import Dict, Any
from openai import OpenAI
from textwrap import dedent

from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.env_service_client.env_client_ng import EnvClient
from ajet.utils.retry import retry_with_backoff


# React Agent System Prompt
REACT_SYSTEM_PROMPT = dedent("""
You are an autonomous agent tasked with completing tasks in the AppWorld environment.

You will interact with various applications and services to accomplish user goals.
You should follow the ReAct (Reasoning and Acting) paradigm:

1. **Thought**: Think about what you need to do next
2. **Action**: Take an action in the environment
3. **Observation**: Observe the result of your action
4. **Repeat** until the task is complete

When you want to take an action, format it as:
Action: <your action here>

When you want to think, format it as:
Thought: <your reasoning here>

When you believe the task is complete, respond with:
DONE

Be strategic, efficient, and adaptive in your approach.
""").strip()


def _parse_agent_response(response: str) -> tuple[str, str]:
    """
    Parse the agent's response to extract thought and action.

    Args:
        response: The agent's text response

    Returns:
        tuple of (thought, action)
    """
    thought = ""
    action = ""

    # Extract thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|DONE|$))", response, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r"Action:\s*(.+?)(?=\n(?:Thought:|DONE|$))", response, re.DOTALL | re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()

    return thought, action


def _is_done(response: str) -> bool:
    """Check if the agent indicates the task is complete."""
    return "DONE" in response.upper()


@retry_with_backoff(max_retry=3)
def _execute_agent(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
    env_service_url: str = "http://localhost:8000",
    max_steps: int = 30
) -> WorkflowOutput:
    """
    Execute the React agent for a single episode.

    Args:
        task: The task containing environment configuration
        api_baseurl_key: OpenAI API credentials
        env_service_url: URL of the environment service
        max_steps: Maximum number of interaction steps

    Returns:
        WorkflowOutput with reward and metadata
    """
    # Initialize OpenAI client
    base_url, api_key = api_baseurl_key.base_url, api_baseurl_key.api_key
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Initialize environment client
    env_client = EnvClient(base_url=env_service_url)

    # Create environment instance
    env_type = task.env_type
    task_id = task.task_id
    instance_id = f"{task_id}_{os.urandom(8).hex()}"

    # Create the environment instance
    create_result = env_client.create_instance(
        env_type=env_type,
        task_id=task_id,
        instance_id=instance_id
    )

    # Extract initial state
    initial_state = create_result.get("state", [])
    initial_observation = ""
    if isinstance(initial_state, list) and len(initial_state) > 0:
        initial_observation = initial_state[-1].get("content", "")
    elif isinstance(initial_state, dict):
        initial_observation = initial_state.get("content", "")

    # Initialize conversation history
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {initial_observation}"}
    ]

    # Interaction loop
    step = 0
    total_reward = 0.0
    is_terminated = False
    trajectory = []

    for step in range(max_steps):
        # Get agent's response
        response = client.chat.completions.create(
            model="agentjet-model",
            messages=messages,  # type: ignore
            temperature=0.7,
            max_tokens=2048
        )

        agent_output = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": agent_output})

        # Record trajectory
        trajectory.append({
            "step": step,
            "agent_output": agent_output
        })

        # Check if done
        if _is_done(agent_output):
            is_terminated = True
            break

        # Parse thought and action
        thought, action = _parse_agent_response(agent_output)

        # If no action extracted, prompt for action
        if not action:
            messages.append({
                "role": "user",
                "content": "Please specify an action to take."
            })
            continue

        # Execute action in environment
        step_result = env_client.step(
            instance_id=instance_id,
            action={"content": action, "role": "assistant"},
            params={}
        )

        # Extract observation
        observation_data = step_result.get("state", {})
        if isinstance(observation_data, dict):
            observation = observation_data.get("content", "")
        else:
            observation = str(observation_data)

        # Get reward and termination status
        step_reward = step_result.get("reward", 0.0)
        total_reward += step_reward
        is_terminated = step_result.get("is_terminated", False)

        # Add observation to conversation
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

        # Record trajectory
        trajectory[-1]["action"] = action
        trajectory[-1]["observation"] = observation
        trajectory[-1]["reward"] = step_reward

        if is_terminated:
            break

    # Evaluate final performance
    try:
        final_reward = env_client.evaluate(
            instance_id=instance_id,
            messages={},
            params={}
        )
    except Exception:
        final_reward = total_reward

    # Release environment instance
    try:
        env_client.release_instance(instance_id=instance_id)
    except Exception:
        pass

    return WorkflowOutput(
        reward=final_reward,
        metadata={
            "total_steps": step + 1,
            "total_reward": total_reward,
            "is_terminated": is_terminated,
            "trajectory": trajectory
        }
    )


def _compute_reward(task: Task, metadata: Dict[str, Any]) -> float:
    """
    Compute reward for the episode.

    For AppWorld, the environment service already provides the reward,
    so we just extract it from metadata.

    Args:
        task: The task object
        metadata: Metadata from agent execution

    Returns:
        The computed reward
    """
    # The reward is already computed by the environment
    return metadata.get("total_reward", 0.0)


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str,
    env_service_url: str = "http://localhost:8000"
) -> WorkflowOutput:
    """
    Main entry point for running the agent and computing reward.

    Args:
        task: The task to execute
        base_url: OpenAI API base URL
        api_key: OpenAI API key
        env_service_url: Environment service URL

    Returns:
        WorkflowOutput with reward and metadata
    """
    # Create API credentials
    api_baseurl_key = OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key)

    # Execute agent
    workflow_output = _execute_agent(
        task=task,
        api_baseurl_key=api_baseurl_key,
        env_service_url=env_service_url
    )

    # The reward is already computed by the environment
    # We can optionally add custom reward shaping here

    return workflow_output
