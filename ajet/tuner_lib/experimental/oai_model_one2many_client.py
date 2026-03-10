# -*- coding: utf-8 -*-

import os
import time
import requests
from typing import List, Dict

PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8000")

MESSAGES = [
    [{"role": "user", "content": "Hello, how are you?"}],
    [{"role": "user", "content": "Tell me a joke."}],
    [{"role": "user", "content": "What's the weather like today?"}],
    [{"role": "user", "content": "Write a short poem about coding."}],
    [{"role": "user", "content": "What is Python?"}],
    [{"role": "user", "content": "How do I learn machine learning?"}],
    [{"role": "user", "content": "Tell me about your hobbies."}],
    [{"role": "user", "content": "What's your favorite programming language?"}],
    [{"role": "user", "content": "Explain what is an API."}],
    [{"role": "user", "content": "Give me a recipe for pasta."}],
]


def send_chat_request(messages: List[Dict[str, str]], stream: bool = False) -> Dict:
    """Send a chat completion request to the proxy server."""
    payload = {
        "model": "test-model",
        "messages": messages,
        "stream": stream,
    }

    try:
        response = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    print(f"Starting client, sending requests to {PROXY_URL}")
    print("Press Ctrl+C to stop\n")

    request_count = 0

    while True:
        request_count += 1
        messages = MESSAGES[request_count % len(MESSAGES)]

        print(f"[Request {request_count}] Sending: {messages[0]['content'][:50]}...")

        result = send_chat_request(messages)

        if "error" in result:
            print(f"[Request {request_count}] Error: {result['error']}")
        else:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"[Request {request_count}] Response: {content[:100]}...")

        print()

        time.sleep(5)


if __name__ == "__main__":
    try:
        health = requests.get(f"{PROXY_URL}/health", timeout=5)
        print(f"Server health: {health.json()}\n")
    except Exception as e:
        print(f"Warning: Could not connect to server: {e}\n")

    main()
