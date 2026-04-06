"""
Agent runner for spy game - agent_roll_adv mode (adversarial training).
Team A (civilians): shared 7B model from swarm server 1
Team B (spies): shared 7B model from swarm server 2
"""

import random
from typing import Dict, Tuple
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


def _compute_rewards(game_result: Dict) -> Tuple[float, float]:
    """
    Compute rewards for both teams in adversarial mode.

    Args:
        game_result: Dictionary containing game outcome

    Returns:
        (civilian_team_reward, spy_team_reward)
    """
    civilian_reward = game_result["civilian_reward"]
    spy_reward = game_result["spy_reward"]
    return civilian_reward, spy_reward


def _execute_agent(task: Task,
                   api_baseurl_key_civilians: OpenaiBaseUrlAndApiKey,
                   api_baseurl_key_spies: OpenaiBaseUrlAndApiKey) -> Dict:
    """
    Execute one episode of the adversarial spy game.

    Args:
        task: Task containing game configuration
        api_baseurl_key_civilians: API credentials for civilian team (swarm server 1)
        api_baseurl_key_spies: API credentials for spy team (swarm server 2)

    Returns:
        Game result dictionary
    """
    # Extract game configuration from task
    civilian_word = task.metadata["civilian_word"]
    spy_word = task.metadata["spy_word"]
    num_players = task.metadata["num_players"]
    num_spies = task.metadata["num_spies"]

    # Randomly sample player names for this episode
    selected_names = random.sample(PLAYER_NAMES, num_players)

    # Randomly assign player IDs and roles
    player_indices = list(range(num_players))
    random.shuffle(player_indices)

    # First num_spies indices become spies (using swarm server 2)
    # Remaining indices become civilians (using swarm server 1)
    player_configs = []
    spy_count = 0
    civilian_count = 0

    for i in range(num_players):
        # Use shuffled index to determine role
        is_spy = player_indices[i] < num_spies

        if is_spy:
            # Spy uses swarm server 2
            config = {
                "name": selected_names[i],
                "base_url": api_baseurl_key_spies.base_url,
                "api_key": api_baseurl_key_spies.api_key,
                "model": "agentjet-model"
            }
            spy_count += 1
        else:
            # Civilian uses swarm server 1
            config = {
                "name": selected_names[i],
                "base_url": api_baseurl_key_civilians.base_url,
                "api_key": api_baseurl_key_civilians.api_key,
                "model": "agentjet-model"
            }
            civilian_count += 1

        player_configs.append(config)

    # Assert correct role distribution
    assert spy_count == num_spies, f"Expected {num_spies} spies, got {spy_count}"
    assert civilian_count == num_players - num_spies, f"Expected {num_players - num_spies} civilians, got {civilian_count}"

    # Assert all trainable model users from each server have the same role
    # Swarm server 1 (civilians) users should all be civilians
    # Swarm server 2 (spies) users should all be spies
    civilians_base_url = api_baseurl_key_civilians.base_url
    spies_base_url = api_baseurl_key_spies.base_url

    server1_users = [i for i, cfg in enumerate(player_configs) if cfg["base_url"] == civilians_base_url]
    server2_users = [i for i, cfg in enumerate(player_configs) if cfg["base_url"] == spies_base_url]

    # Check all server 1 users are civilians (indices >= num_spies in shuffled assignment)
    for idx in server1_users:
        assert player_configs[idx]["base_url"] == civilians_base_url, \
            f"Player {idx} should use civilian server but uses {player_configs[idx]['base_url']}"

    # Check all server 2 users are spies (indices < num_spies in shuffled assignment)
    for idx in server2_users:
        assert player_configs[idx]["base_url"] == spies_base_url, \
            f"Player {idx} should use spy server but uses {player_configs[idx]['base_url']}"

    # Verify role consistency: all server1 users should be civilians, all server2 users should be spies
    assert len(server1_users) == civilian_count, \
        f"Server 1 (civilians) should have {civilian_count} users, but has {len(server1_users)}"
    assert len(server2_users) == spy_count, \
        f"Server 2 (spies) should have {spy_count} users, but has {len(server2_users)}"

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


def run_agent_and_compute_reward(
    task: Task,
    base_url_civilians: str,
    api_key_civilians: str,
    base_url_spies: str,
    api_key_spies: str
) -> Tuple[WorkflowOutput, WorkflowOutput]:
    """
    Main entry point for adversarial mode - returns two WorkflowOutputs.

    Args:
        task: Task containing game configuration
        base_url_civilians: Base URL for civilian team model (swarm server 1)
        api_key_civilians: API key for civilian team
        base_url_spies: Base URL for spy team model (swarm server 2)
        api_key_spies: API key for spy team

    Returns:
        (workflow_output_civilians, workflow_output_spies)
    """
    api_baseurl_key_civilians = OpenaiBaseUrlAndApiKey(
        base_url=base_url_civilians,
        api_key=api_key_civilians
    )
    api_baseurl_key_spies = OpenaiBaseUrlAndApiKey(
        base_url=base_url_spies,
        api_key=api_key_spies
    )

    try:
        # Execute game
        game_result = _execute_agent(task, api_baseurl_key_civilians, api_baseurl_key_spies)

        # Compute rewards for both teams
        civilian_reward, spy_reward = _compute_rewards(game_result)

        # Create separate workflow outputs for each team
        workflow_output_civilians = WorkflowOutput(
            reward=civilian_reward,
            metadata={
                "team": "civilians",
                "winner": game_result["winner"],
                "total_rounds": game_result["total_rounds"],
                "civilian_word": game_result["civilian_word"],
                "spy_word": game_result["spy_word"],
                "final_alive": game_result["final_alive"]
            }
        )

        workflow_output_spies = WorkflowOutput(
            reward=spy_reward,
            metadata={
                "team": "spies",
                "winner": game_result["winner"],
                "total_rounds": game_result["total_rounds"],
                "civilian_word": game_result["civilian_word"],
                "spy_word": game_result["spy_word"],
                "final_alive": game_result["final_alive"]
            }
        )

        return workflow_output_civilians, workflow_output_spies

    except Exception as e:
        print(f"Error during adversarial game execution: {e}")
        # Return neutral rewards on failure
        error_output_civilians = WorkflowOutput(
            reward=0.5,
            metadata={"error": str(e), "winner": "error", "team": "civilians"}
        )
        error_output_spies = WorkflowOutput(
            reward=0.5,
            metadata={"error": str(e), "winner": "error", "team": "spies"}
        )
        return error_output_civilians, error_output_spies
