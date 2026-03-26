"""
Agent runner for spy game - agent_roll mode.
Team A (civilians): shared 7B model
Team B (spies): qwen-max via DashScope API
"""

import os
import random
from typing import Dict
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from tutorial.opencode_build_spy_game.game_engine import SpyGame


# Pre-generated diverse name pool for players
PLAYER_NAMES = [
    "Alexander", "Benjamin", "Christopher", "Daniel", "Elizabeth",
    "Fitzgerald", "Gabriella", "Harrison", "Isabella", "Jonathan",
    "Katherine", "Leonardo", "Margaret", "Nathaniel", "Ophelia",
    "Penelope", "Quentin", "Rosalind", "Sebastian", "Theodora",
    "Ulysses", "Victoria", "Wellington", "Xander", "Yasmine",
    "Zachary", "Adelaide", "Beatrice", "Cornelius", "Desmond",
    "Eleanor", "Frederick", "Genevieve", "Humphrey", "Imogen",
    "Jasper", "Lillian", "Maximilian", "Nicolette", "Orlando",
    "Percival", "Quintessa", "Reginald", "Seraphina", "Tristan",
    "Valentina", "Winifred", "Xavier", "Yolanda", "Zephyr"
]


def _compute_reward(game_result: Dict) -> float:
    """
    Compute reward for the trainable team (civilians using 7B model).

    Args:
        game_result: Dictionary containing game outcome

    Returns:
        Reward value: 1.0 if civilians win, 0.0 if spies win
    """
    return game_result["civilian_reward"]


def _execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey) -> Dict:
    """
    Execute one episode of the spy game.

    Args:
        task: Task containing game configuration
        api_baseurl_key: API credentials for the trainable 7B model

    Returns:
        Game result dictionary
    """
    # Extract game configuration from task
    civilian_word = task.metadata["civilian_word"]
    spy_word = task.metadata["spy_word"]
    num_players = task.metadata["num_players"]
    num_spies = task.metadata["num_spies"]

    # Get DashScope API key for opponent
    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

    # Randomly sample player names for this episode
    selected_names = random.sample(PLAYER_NAMES, num_players)

    # Randomly assign player IDs and roles
    player_indices = list(range(num_players))
    random.shuffle(player_indices)

    # First num_spies indices become spies (using qwen-max)
    # Remaining indices become civilians (using 7B model)
    player_configs = []
    spy_count = 0
    civilian_count = 0

    for i in range(num_players):
        # Use shuffled index to determine role
        is_spy = player_indices[i] < num_spies

        if is_spy:
            # Spy uses qwen-max
            config = {
                "name": selected_names[i],
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": dashscope_api_key,
                "model": "qwen-max"
            }
            spy_count += 1
        else:
            # Civilian uses trainable 7B model
            config = {
                "name": selected_names[i],
                "base_url": api_baseurl_key.base_url,
                "api_key": api_baseurl_key.api_key,
                "model": "agentjet-model"
            }
            civilian_count += 1

        player_configs.append(config)

    # Assert correct role distribution
    assert spy_count == num_spies, f"Expected {num_spies} spies, got {spy_count}"
    assert civilian_count == num_players - num_spies, f"Expected {num_players - num_spies} civilians, got {civilian_count}"

    # Assert all trainable model users have the same role (all civilians in this mode)
    trainable_base_url = api_baseurl_key.base_url
    trainable_roles = []
    for i, config in enumerate(player_configs):
        if config["base_url"] == trainable_base_url:
            # Determine role by checking which model is used
            is_civilian = config["model"] == "agentjet-model" and config["base_url"] == trainable_base_url
            trainable_roles.append(is_civilian)

    # All trainable model users must have the same role
    if trainable_roles:
        assert all(trainable_roles) or not any(trainable_roles), \
            f"All trainable model users must have the same role (all civilians or all spies), but got mixed roles"
        # In agent_run mode, all trainable model users should be civilians
        assert all(trainable_roles), \
            f"In agent_run mode, all trainable model users should be civilians"

    # Create and run game
    game = SpyGame(
        civilian_word=civilian_word,
        spy_word=spy_word,
        num_players=num_players,
        num_spies=num_spies,
        player_configs=player_configs
    )

    game_result = game.play_game()
    return game_result


def run_agent_and_compute_reward(task: Task, base_url: str, api_key: str) -> WorkflowOutput:
    """
    Main entry point for running the agent and computing reward.

    Args:
        task: Task containing game configuration
        base_url: Base URL for the trainable model
        api_key: API key for the trainable model

    Returns:
        WorkflowOutput with reward and game metadata
    """
    api_baseurl_key = OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key)

    try:
        # Execute game
        game_result = _execute_agent(task, api_baseurl_key)

        # Compute reward (1.0 if civilians win, 0.0 if spies win)
        reward = _compute_reward(game_result)

        # Return workflow output
        return WorkflowOutput(
            reward=reward,
            metadata={
                "winner": game_result["winner"],
                "total_rounds": game_result["total_rounds"],
                "civilian_word": game_result["civilian_word"],
                "spy_word": game_result["spy_word"],
                "final_alive": game_result["final_alive"]
            }
        )

    except Exception as e:
        print(f"Error during game execution: {e}")
        # Return 0 reward on failure
        return WorkflowOutput(
            reward=0.0,
            metadata={
                "error": str(e),
                "winner": "error"
            }
        )
