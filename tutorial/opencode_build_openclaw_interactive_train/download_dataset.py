# -*- coding: utf-8 -*-
"""Download personality_manipulation dataset from HuggingFace."""

from datasets import load_dataset
import json

def download_and_save_dataset():
    """Download personality_manipulation dataset and save extraversion samples."""
    print("Downloading personality_manipulation dataset...")
    dataset = load_dataset("holistic-ai/personality_manipulation")

    # Filter for extraversion personality
    extraversion_data = [item for item in dataset['train'] if item['Target Personality'] == 'extraversion']

    # Save to JSON
    with open('extraversion_questions.json', 'w', encoding='utf-8') as f:
        json.dump(extraversion_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(extraversion_data)} extraversion samples to extraversion_questions.json")

    # Also save all personalities for reference
    with open('all_personalities.json', 'w', encoding='utf-8') as f:
        json.dump(list(dataset['train']), f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset['train'])} total samples to all_personalities.json")

if __name__ == "__main__":
    download_and_save_dataset()
