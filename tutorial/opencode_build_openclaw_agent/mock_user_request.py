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

def send_openclaw_message(question: str) -> str:
    """Send message via OpenClaw CLI."""
    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--message", question],
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

    # Send requests via OpenClaw CLI
    for i, item in enumerate(dataset):
        question = item.get("Question", "")
        print(f"[{i+1}/{len(dataset)}] Sending: {question[:80]}...")

        response = send_openclaw_message(question)
        print(f"Response: {response[:200]}...\n")

        time.sleep(2)

if __name__ == "__main__":
    main()
