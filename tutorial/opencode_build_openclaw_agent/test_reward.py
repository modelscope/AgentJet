#!/usr/bin/env python3
"""Test script for on_compute_relative_reward.py using real OpenJudge API.

Tests four reward dimensions:
  1. Extraversion — enthusiastic responses score higher
  2. Relevance — on-topic responses score higher than off-topic
  3. Diversity — unique responses score higher than near-duplicates
  4. Quality gate — repetitive/degenerate responses get crushed
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-xxx")


async def test_pointwise_composite():
    """Test pointwise composite reward (extraversion + relevance + diversity)."""
    print("\n=== Testing Pointwise Composite Reward ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()  # fresh history for test isolation

    question = "What are your thoughts on Paris?"
    all_answers = [
        {"content": "I'm so excited about Paris! The Eiffel Tower at night is breathtaking and the cafes are amazing!"},
        {"content": "Paris is a city in France."},
        {"content": "I absolutely love Paris! The energy on the Champs-Élysées is fantastic and so vibrant!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Composite scores: {scores}")
        for a in all_answers:
            print(f"  ext={a.get('extraversion')}, rel={a.get('relevance')}, "
                  f"div={a.get('diversity')}, reward={a.get('reward')}  "
                  f"content={a['content'][:50]}...")

        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        assert all(isinstance(s, float) for s in scores), "All scores should be floats"
        # Extraverted + relevant responses should beat the flat neutral one
        assert scores[0] > scores[1], f"Enthusiastic on-topic should beat neutral: {scores}"
        assert scores[2] > scores[1], f"Enthusiastic on-topic should beat neutral: {scores}"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_relevance_penalty():
    """Off-topic answers should get lower composite scores than on-topic ones."""
    print("\n=== Testing Relevance Penalty ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()

    question = "What is your favorite food?"
    all_answers = [
        # On-topic, extraverted
        {"content": "Oh my gosh, I absolutely LOVE sushi! The flavors are incredible and I get so excited every time!"},
        # Off-topic, extraverted (talks about space, not food)
        {"content": "WOW space exploration is SO exciting! Rockets launching into the sky fills me with energy!!!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Scores: {scores}")
        for a in all_answers:
            print(f"  ext={a.get('extraversion')}, rel={a.get('relevance')}, "
                  f"div={a.get('diversity')}, reward={a.get('reward')}  "
                  f"content={a['content'][:50]}...")

        # Both are extraverted, but on-topic should win because of relevance
        assert scores[0] > scores[1], \
            f"On-topic extraverted should beat off-topic extraverted: {scores}"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_diversity_penalty():
    """Near-duplicate answers should get lower diversity scores."""
    print("\n=== Testing Diversity Penalty ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()

    question = "Tell me about your hobbies."
    all_answers = [
        {"content": "I love hiking in the mountains! The fresh air and stunning views make me feel so alive and energized!"},
        # Near-duplicate of answer 0
        {"content": "I love hiking in the mountains! The fresh air and stunning views make me feel so alive and energized!"},
        # Unique answer
        {"content": "Dancing is my absolute passion! Nothing beats the energy of moving to great music with friends!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Scores: {scores}")
        for a in all_answers:
            print(f"  ext={a.get('extraversion')}, rel={a.get('relevance')}, "
                  f"div={a.get('diversity')}, reward={a.get('reward')}  "
                  f"content={a['content'][:50]}...")

        # The duplicate pair should have lower diversity than the unique one
        div_duplicate = all_answers[0].get("diversity", 1.0)
        div_unique = all_answers[2].get("diversity", 0.0)
        assert div_unique > div_duplicate, \
            f"Unique response should have higher diversity ({div_unique}) than duplicate ({div_duplicate})"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_cross_request_diversity():
    """Answers that repeat historical responses should be penalized."""
    print("\n=== Testing Cross-Request Diversity ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()

    # Simulate a prior request that produced a response
    mod.record_responses_to_history([
        "I love hiking in the mountains! The fresh air and stunning views make me feel so alive!"
    ])

    question = "What do you enjoy doing on weekends?"
    all_answers = [
        # Repeats the historical response almost verbatim
        {"content": "I love hiking in the mountains! The fresh air and stunning views make me feel so alive!"},
        # Fresh, unique response
        {"content": "Weekends are for exploring new restaurants and trying exotic cuisines! I get so thrilled by new flavors!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Scores: {scores}")
        for a in all_answers:
            print(f"  ext={a.get('extraversion')}, rel={a.get('relevance')}, "
                  f"div={a.get('diversity')}, reward={a.get('reward')}  "
                  f"content={a['content'][:50]}...")

        div_stale = all_answers[0].get("diversity", 1.0)
        div_fresh = all_answers[1].get("diversity", 0.0)
        assert div_fresh > div_stale, \
            f"Fresh response should have higher diversity ({div_fresh}) than stale ({div_stale})"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_repetition_penalty():
    """Degenerate looping responses should get near-zero reward."""
    print("\n=== Testing Repetition / Degeneration Penalty ===")
    os.environ["REWARD_MODE"] = "pointwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()

    question = "Tell me about Dunfermline."

    # Build a degenerate looping response (similar to the real failure case)
    good_intro = "Hello! Dunfermline is a charming town in Fife, Scotland, with a rich history."
    loop_block = (
        "\n\n---\n\n"
        "If you have any specific questions or need more information, just "
        "let me know! I'm here to assist you in making your visit to "
        "Dunfermline a delightful experience.\n\n---\n\n"
        "Looking forward to your wonderful Dunfermline adventures!\n\n---\n\n"
        "Thank you for the opportunity to share my thoughts on Dunfermline. "
        "If you have any more questions or need assistance, feel free to "
        "reach out!"
    )
    degenerate_response = good_intro + (loop_block * 15)  # repeat the block many times

    all_answers = [
        # Degenerate looping response
        {"content": degenerate_response},
        # Clean, concise, extraverted response
        {"content": "Dunfermline is absolutely wonderful! The abbey ruins are breathtaking and the town has such vibrant energy. I love the mix of history and modern community spirit there!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Scores: {scores}")
        for a in all_answers:
            print(f"  quality={a.get('quality')}, ext={a.get('extraversion')}, "
                  f"rel={a.get('relevance')}, div={a.get('diversity')}, "
                  f"reward={a.get('reward')}  "
                  f"content={a['content'][:60]}...")

        quality_degenerate = all_answers[0].get("quality", 1.0)
        quality_clean = all_answers[1].get("quality", 0.0)
        print(f"  Quality scores: degenerate={quality_degenerate}, clean={quality_clean}")

        # The degenerate response should have much lower quality
        assert quality_clean > quality_degenerate, \
            f"Clean response quality ({quality_clean}) should exceed degenerate ({quality_degenerate})"
        # The clean response should win overall
        assert scores[1] > scores[0], \
            f"Clean response ({scores[1]}) should beat degenerate ({scores[0]})"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def test_listwise_composite():
    """Listwise mode should also produce composite rewards."""
    print("\n=== Testing Listwise Composite Reward ===")
    os.environ["REWARD_MODE"] = "listwise"

    import importlib
    import on_compute_relative_reward as mod
    importlib.reload(mod)
    mod._response_history.clear()

    question = "What are your thoughts on Paris?"
    all_answers = [
        {"content": "I'm so excited about Paris! The Eiffel Tower at night is breathtaking!"},
        {"content": "Paris is a city in France."},
        {"content": "I absolutely love Paris! The Champs-Élysées energy is fantastic!"},
    ]

    try:
        scores = await mod.on_compute_relative_reward([], all_answers, question=question)
        print(f"Scores: {scores}")
        for a in all_answers:
            print(f"  ext={a.get('extraversion')}, rel={a.get('relevance')}, "
                  f"div={a.get('diversity')}, reward={a.get('reward')}  "
                  f"content={a['content'][:50]}...")

        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        # Neutral response should score lowest
        assert scores[1] < scores[0] or scores[1] < scores[2], \
            f"Neutral response should score lower than at least one extraverted response: {scores}"
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


async def main():
    print("Testing on_compute_relative_reward.py — Composite Reward")
    print("(extraversion + relevance + diversity + quality gate)")
    print("=" * 60)

    results = []
    results.append(await test_pointwise_composite())
    results.append(await test_relevance_penalty())
    results.append(await test_diversity_penalty())
    results.append(await test_cross_request_diversity())
    results.append(await test_repetition_penalty())
    results.append(await test_listwise_composite())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    if not all(results):
        names = [
            "pointwise_composite", "relevance_penalty", "diversity_penalty",
            "cross_request_diversity", "repetition_penalty", "listwise_composite",
        ]
        for name, ok in zip(names, results):
            if not ok:
                print(f"  FAILED: {name}")
    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
