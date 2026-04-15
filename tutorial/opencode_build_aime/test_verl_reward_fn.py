# -*- coding: utf-8 -*-
"""
Test for verl_reward_fn modules.
"""

import sys
sys.path.insert(0, "/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet/tutorial/opencode_build_aime")

from verl_reward_fn.prime_math import compute_score as prime_compute_score
from verl_reward_fn.math_verify import compute_score as math_verify_compute_score


def test_prime_compute_score():
    print("Testing prime_compute_score...")

    test_cases = [
        ("\\boxed{42}", "42", True),
        ("The answer is \\boxed{123}", "123", True),
        ("\\boxed{1,000}", "1000", True),
        ("\\boxed{42}", "43", False),
        ("The answer is 42", "42", True),
        ("Answer: 3.14", "3.14", True),
    ]

    for model_output, ground_truth, expected in test_cases:
        result = prime_compute_score(model_output, ground_truth)
        score = result[0] if isinstance(result, tuple) else result
        status = "PASS" if bool(score) == expected else "FAIL"
        print(f"  [{status}] prime({repr(model_output)}, {repr(ground_truth)}) = {score}, expected {expected}")

    print()


def test_math_verify_compute_score():
    print("Testing math_verify_compute_score...")

    test_cases = [
        ("\\boxed{42}", "42", True),
        ("\\boxed{123}", "123", True),
        ("\\boxed{42}", "43", False),
    ]

    for model_output, ground_truth, expected in test_cases:
        try:
            score = math_verify_compute_score(model_output, ground_truth)
            status = "PASS" if bool(score) == expected else "FAIL"
            print(f"  [{status}] math_verify({repr(model_output)}, {repr(ground_truth)}) = {score}, expected {expected}")
        except ImportError as e:
            print(f"  [SKIP] math_verify requires math-verify package: {e}")
            break
        except Exception as e:
            print(f"  [ERROR] math_verify({repr(model_output)}, {repr(ground_truth)}): {e}")

    print()


def test_rstar2_reward_style():
    print("Testing rStar2-style reward computation (prime -> math_verify fallback)...")

    from verl_reward_fn.prime_math import compute_score as prime_compute_score
    from verl_reward_fn.math_verify import compute_score as math_verify_compute_score

    def compute_score(model_output: str, ground_truth: str) -> float:
        try:
            prime_score = prime_compute_score(model_output, ground_truth)[0]
            if prime_score:
                return 1.0
        except Exception:
            prime_score = 0.0
        try:
            math_verify_score = math_verify_compute_score(model_output, ground_truth)
            if math_verify_score:
                return 1.0
        except Exception:
            return 0.0
        return 0.0

    test_cases = [
        ("\\boxed{42}", "42", 1.0),
        ("The answer is \\boxed{123}", "123", 1.0),
        ("\\boxed{42}", "43", 0.0),
        ("Answer: 3.14", "3.14", 1.0),
    ]

    for model_output, ground_truth, expected in test_cases:
        score = compute_score(model_output, ground_truth)
        status = "PASS" if score == expected else "FAIL"
        print(f"  [{status}] compute_score({repr(model_output)}, {repr(ground_truth)}) = {score}, expected {expected}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing verl_reward_fn modules")
    print("=" * 60)
    print()

    try:
        test_prime_compute_score()
    except ImportError as e:
        print(f"prime_math tests skipped: {e}\n")
    except Exception as e:
        print(f"prime_math tests failed: {e}\n")

    try:
        test_math_verify_compute_score()
    except ImportError as e:
        print(f"math_verify tests skipped: {e}\n")
    except Exception as e:
        print(f"math_verify tests failed: {e}\n")

    try:
        test_rstar2_reward_style()
    except ImportError as e:
        print(f"rStar2 reward style tests skipped: {e}\n")
    except Exception as e:
        print(f"rStar2 reward style tests failed: {e}\n")

    print("=" * 60)
    print("Tests completed")
    print("=" * 60)
