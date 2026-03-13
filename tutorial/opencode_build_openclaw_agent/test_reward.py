#!/usr/bin/env python3
"""Test script for on_compute_relative_reward.py using real OpenJudge API."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-xxx")


async def test_pointwise():
    """Test pointwise reward mode with real API."""
    print("\n=== Testing Pointwise Mode (real API) ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)

    valid_results = [{"question": "What are your thoughts on Paris?"}]
    all_answers = [
        {"content": "I'm so excited about Paris! It's amazing and wonderful!"},
        {"content": "Paris is a city in France."},
        {"content": "I absolutely love Paris! The energy is fantastic and vibrant!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward(valid_results, all_answers)
        print(f"Scores: {scores}")
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        assert all(isinstance(s, float) for s in scores), "All scores should be floats"
        # extraverted responses should score higher than neutral
        assert scores[0] > scores[1], f"Extraverted response should score higher than neutral: {scores}"
        assert scores[2] > scores[1], f"Extraverted response should score higher than neutral: {scores}"
        print("✓ Pointwise mode test passed")
        return True
    except Exception as e:
        print(f"✗ Pointwise mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_listwise():
    """Test listwise reward mode with real API."""
    print("\n=== Testing Listwise Mode (real API) ===")
    os.environ["REWARD_MODE"] = "listwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)

    valid_results = [{"question": "What are your thoughts on Paris?"}]
    all_answers = [
        {"content": "I'm so excited about Paris! It's amazing and wonderful!"},
        {"content": "Paris is a city in France."},
        {"content": "I absolutely love Paris! The energy is fantastic and vibrant!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward(valid_results, all_answers)
        print(f"Scores: {scores}")
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        assert all(isinstance(s, float) for s in scores), "All scores should be floats"
        # neutral response should score lowest
        assert scores[1] < scores[0] or scores[1] < scores[2], \
            f"Neutral response should score lower than at least one extraverted response: {scores}"
        print("✓ Listwise mode test passed")
        return True
    except Exception as e:
        print(f"✗ Listwise mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("Testing on_compute_relative_reward.py (real API)")
    print("=" * 50)

    results = []
    results.append(await test_pointwise())
    results.append(await test_listwise())

    print("\n" + "=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
