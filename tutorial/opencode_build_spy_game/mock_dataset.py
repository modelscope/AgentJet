import json
import random
from pathlib import Path


CIVILIAN_SPY_PAIRS = [
    ("apple", "pear"),
    ("coffee", "tea"),
    ("basketball", "football"),
    ("piano", "guitar"),
    ("rose", "tulip"),
    ("dog", "cat"),
    ("bicycle", "motorcycle"),
    ("ocean", "sea"),
    ("winter", "autumn"),
    ("sunrise", "sunset"),
    ("rice", "noodle"),
    ("book", "magazine"),
    ("violin", "cello"),
    ("lion", "tiger"),
    ("river", "lake"),
    ("mountain", "hill"),
    ("sun", "moon"),
    ("chair", "stool"),
    ("milk", "yogurt"),
    ("bread", "cake"),
    ("airplane", "helicopter"),
    ("train", "subway"),
    ("doctor", "nurse"),
    ("teacher", "professor"),
    ("pen", "pencil"),
    ("email", "letter"),
    ("computer", "laptop"),
    ("phone", "tablet"),
    ("shoes", "slippers"),
    ("hat", "cap"),
    ("sword", "knife"),
    ("bow", "crossbow"),
    ("king", "emperor"),
    ("princess", "queen"),
    ("spring", "summer"),
    ("rain", "snow"),
    ("thunder", "lightning"),
    ("diamond", "crystal"),
    ("gold", "silver"),
    ("red", "orange"),
    ("square", "rectangle"),
    ("circle", "oval"),
    ("triangle", "pyramid"),
    ("watermelon", "melon"),
    ("strawberry", "raspberry"),
    ("carrot", "radish"),
    ("potato", "sweet potato"),
    ("chicken", "duck"),
    ("beef", "pork"),
    ("shark", "whale"),
]


def generate_mock_dataset(num_samples: int = 100, output_path: str = None) -> list[dict]:
    """
    Generate mock game configuration dataset.
    Each sample contains:
    - civilian_word: word for civilians
    - spy_word: word for spies
    - num_players: total number of players (6-9)
    - num_spies: number of spies (1-2)
    """
    dataset = []

    for _ in range(num_samples):
        civilian_word, spy_word = random.choice(CIVILIAN_SPY_PAIRS)

        # Randomly swap to increase diversity
        if random.random() > 0.5:
            civilian_word, spy_word = spy_word, civilian_word

        num_players = random.randint(6, 9)
        num_spies = 1 if num_players <= 7 else random.choice([1, 2])

        dataset.append({
            "civilian_word": civilian_word,
            "spy_word": spy_word,
            "num_players": num_players,
            "num_spies": num_spies,
        })

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {output_path}")

    return dataset


if __name__ == "__main__":
    output_path = Path(__file__).parent / "mock_game_dataset.json"
    dataset = generate_mock_dataset(num_samples=200, output_path=str(output_path))
    print(f"Generated {len(dataset)} game configurations")
    print(f"Sample: {dataset[0]}")
