# -*- coding: utf-8 -*-
"""Mock user requests using OpenClaw CLI interface."""

import json
import subprocess
import time
import os
import random
from typing import List, Dict

GATEWAY_PORT = os.getenv("OPENCLAW_PORT", "18789")

def load_dataset(filepath: str = "extraversion_questions.json") -> List[Dict]:
    """Load personality manipulation dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_agent_name() -> str:
    """Generate a random agent name."""
    adjectives = ["happy", "quick", "bright", "clever", "bold", "calm", "eager", "gentle"]
    nouns = ["fox", "wolf", "bear", "eagle", "hawk", "lion", "tiger", "owl"]
    return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(1000, 9999)}"


def create_agent(agent_name: str) -> bool:
    """Create a new agent using OpenClaw CLI."""
    try:
        workspace = f"/root/.openclaw/workspace-{agent_name}"
        result = subprocess.run(
            ["openclaw", "agents", "add", agent_name, "--workspace", workspace, "--non-interactive"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"Created agent: {agent_name}")
            return True
        else:
            print(f"Error creating agent {agent_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error creating agent: {str(e)}")
        return False


def delete_agent(agent_name: str) -> bool:
    """Delete an agent using OpenClaw CLI."""
    try:
        result = subprocess.run(
            ["openclaw", "agents", "delete", agent_name, "--force"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"Deleted agent: {agent_name}")
            return True
        else:
            print(f"Error deleting agent {agent_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error deleting agent: {str(e)}")
        return False


def send_openclaw_message(agent_name: str, message: str) -> str:
    """Send message via OpenClaw CLI to specific agent."""
    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", agent_name, "--message", message],
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main loop to send requests from dataset."""
    print("Starting OpenClaw mock user requests")

    # Load dataset
    dataset = load_dataset()
    random.shuffle(dataset)
    print(f"Loaded {len(dataset)} questions from dataset\n")

    # Process dataset in chunks of 5
    for chunk_start in range(0, len(dataset), 5):
        chunk = dataset[chunk_start:chunk_start + 5]

        # Generate random agent name
        agent_name = generate_agent_name()
        print(f"\n=== Creating agent: {agent_name} ===\n")

        # Create agent
        if not create_agent(agent_name):
            print(f"Failed to create agent, skipping chunk")
            continue

        # Send 5 messages
        for i, item in enumerate(chunk):
            question = item.get("Question", "")
            print(f"[{agent_name}/{i+1}/5] Sending: {question[:80]}...")
            response = send_openclaw_message(agent_name, question)
            print(f"Response: {response[:200]}...\n")
            time.sleep(2)

        # Delete agent
        delete_agent(agent_name)
        print(f"\n=== Deleted agent: {agent_name} ===\n")

    print("\nAll agents processed successfully")


if __name__ == "__main__":
    main()
