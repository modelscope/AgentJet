"""
Single-episode debug runner for the spy game.

Plays one full game with all players backed by qwen-max via DashScope, then
prints a focused diagnostic report:

  * the per-team accumulated penalties
  * every auto-elimination (repetition rule) that fired, with the shared words
  * every leak that fired, with the offending player + word
  * a verbatim-repetition spot check that re-runs the rule's extractor on the
    descriptions we actually saw, so we can verify the rule is actually
    capable of catching what we eyeballed

Run:
    DASHSCOPE_API_KEY=... python -m tutorial.opencode_build_spy_game.agent_roll_single_episode
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

from tutorial.opencode_build_spy_game.game_engine import (
    SpyGame,
    extract_content_words,
    REPEAT_LIMIT,
    LEAK_PENALTY,
    WIN_REWARD,
    _description_leaks_word,
)


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

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-max"


def _selftest_extractor() -> None:
    """Re-run the repetition rule on the exact verbatim copies from the user's
    transcript, to prove the extractor would fire if it were called."""
    print("\n" + "=" * 70)
    print("SELFTEST: extractor on the user's round-3 verbatim copies")
    print("=" * 70)
    sentence = "It's a versatile item that pairs well with spreads for a quick and satisfying meal."
    a = extract_content_words(sentence)
    b = extract_content_words(sentence)
    overlap = sorted(a & b)
    print(f"sentence : {sentence!r}")
    print(f"content  : {sorted(a)}")
    print(f"overlap  : {overlap}  (count={len(overlap)}, limit={REPEAT_LIMIT})")
    print(f"would fire: {len(overlap) > REPEAT_LIMIT}")


def _selftest_leak() -> None:
    print("\n" + "=" * 70)
    print("SELFTEST: leak detector")
    print("=" * 70)
    cases = [
        ("bread", "I love bread for breakfast.",      True),
        ("bread", "I love breadcrumbs for breakfast.", False),
        ("bread", "I love BREAD for breakfast.",       True),
        ("bread", "Tasty bread, sliced.",              True),
        ("cake",  "It's a versatile item.",            False),
    ]
    for word, desc, expect in cases:
        got = _description_leaks_word(desc, word)
        ok = "OK " if got == expect else "FAIL"
        print(f"  [{ok}] word={word!r:10s} desc={desc!r:55s} -> {got} (expected {expect})")


def _build_player_configs(num_players: int, names: List[str]) -> List[Dict]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set; cannot run debug episode.")
    return [
        {
            "name": names[i],
            "base_url": DASHSCOPE_BASE_URL,
            "api_key": api_key,
            "model": DASHSCOPE_MODEL,
        }
        for i in range(num_players)
    ]


def _summarise(result: Dict) -> None:
    print("\n" + "#" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("#" * 70)
    print(f"winner             : {result['winner']}")
    print(f"aborted_by_role    : {result['aborted_by_role']}")
    print(f"total_rounds       : {result['total_rounds']}")
    print(f"final_alive        : {result['final_alive']}")
    print(f"civilian_reward    : {result['civilian_reward']:+.4f}")
    print(f"spy_reward         : {result['spy_reward']:+.4f}")
    print(f"team_penalties     : {result['team_penalties']}")
    print(f"reward constants   : WIN_REWARD={WIN_REWARD:+.2f} LEAK_PENALTY={LEAK_PENALTY:+.2f} REPEAT_LIMIT={REPEAT_LIMIT}")

    auto_elims = [
        e for e in result["game_history"]
        if e.get("type") == "elimination" and "Auto-eliminated" in (e.get("reason") or "")
    ]
    leaks = [
        e for e in result["game_history"]
        if e.get("type") == "elimination" and e.get("aborted")
    ]
    vote_elims = [
        e for e in result["game_history"]
        if e.get("type") == "elimination"
        and e.get("eliminated_name") is not None
        and not e.get("aborted")
        and "Auto-eliminated" not in (e.get("reason") or "")
    ]

    print(f"\nrule firings: auto-elims={len(auto_elims)}  leaks={len(leaks)}  vote-elims={len(vote_elims)}")
    for e in auto_elims:
        print(f"  AUTO-ELIM round {e['round']}: {e['eliminated_name']} ({e['eliminated_role']}) | {e['reason']}")
    for e in leaks:
        print(f"  LEAK      round {e['round']}: {e['eliminated_name']} ({e['eliminated_role']}) | {e['reason']}")

    # Cross-check: scan every description against earlier-in-round descriptions
    # using the exact same extractor the engine uses, and report any pair the
    # engine should have caught but didn't.
    print("\ncross-check (recomputing repetition overlap on stored history):")
    by_round: Dict[int, List[Dict]] = {}
    for entry in result["game_history"]:
        if entry.get("type") == "description":
            by_round.setdefault(entry["round"], []).append(entry)
    missed = 0
    for r, entries in by_round.items():
        prior: set = set()
        for e in entries:
            own = extract_content_words(e["description"])
            shared = sorted(own & prior)
            stored = e.get("repeated_words", [])
            marker = ""
            if len(shared) > REPEAT_LIMIT and not stored:
                marker = "  <-- WOULD HAVE FIRED but engine recorded none"
                missed += 1
            elif sorted(stored) != shared:
                marker = f"  <-- mismatch: engine recorded {stored}"
            print(
                f"  r{r} {e['player_name']:13s} stored_repeats={len(stored):2d}  "
                f"recompute_overlap={len(shared):2d} {shared}{marker}"
            )
            prior |= own
    if missed:
        print(f"\n!!! repetition rule missed {missed} firings -- the engine is buggy")
    else:
        print("\nall repetitions detected by recompute were also caught by the engine.")


def main(argv: List[str]) -> int:
    # Allow picking a task from the mock dataset, or default to "bread vs cake"
    # which matches the transcript the user shared.
    dataset_path = Path(__file__).with_name("mock_game_dataset.json")
    tasks = json.loads(dataset_path.read_text())
    task = next(
        (t for t in tasks if t["civilian_word"] == "bread" and t["spy_word"] == "cake"),
        tasks[0],
    )
    if len(argv) > 1:
        idx = int(argv[1])
        task = tasks[idx]

    seed = int(os.environ.get("EPISODE_SEED", "0"))
    random.seed(seed)
    print(f"task = {task}  seed = {seed}")

    _selftest_leak()
    _selftest_extractor()

    names = random.sample(PLAYER_NAMES, task["num_players"])
    configs = _build_player_configs(task["num_players"], names)

    game = SpyGame(
        civilian_word=task["civilian_word"],
        spy_word=task["spy_word"],
        num_players=task["num_players"],
        num_spies=task["num_spies"],
        player_configs=configs,
    )
    result = game.play_game()
    _summarise(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
