"""
Test script to verify a single game works correctly.
"""

from ajet.schema.task import Task
from tutorial.opencode_build_spy_game.agent_run import run_agent_and_compute_reward

# Test with a simple game configuration
test_task = Task(
    main_query="Test spy game episode",
    metadata={
        "civilian_word": "apple",
        "spy_word": "pear",
        "num_players": 6,
        "num_spies": 1,
        "episode_id": 0
    }
)

# Use a fake base_url and api_key for testing (will be replaced by swarm server)
fake_base_url = "http://localhost:10086"
fake_api_key = "test_key"

print("Testing single game execution...")
print(f"Civilian word: {test_task.metadata['civilian_word']}")
print(f"Spy word: {test_task.metadata['spy_word']}")
print(f"Players: {test_task.metadata['num_players']}, Spies: {test_task.metadata['num_spies']}")
print("\nStarting game...\n")

try:
    result = run_agent_and_compute_reward(test_task, fake_base_url, fake_api_key)
    print(f"\n{'='*60}")
    print(f"Game Result:")
    print(f"  Winner: {result.metadata.get('winner', 'unknown')}")
    print(f"  Reward: {result.reward}")
    print(f"  Rounds: {result.metadata.get('total_rounds', 'unknown')}")
    print(f"  Survivors: {result.metadata.get('final_alive', [])}")
    print(f"{'='*60}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
