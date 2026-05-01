import random
import re
from typing import List, Dict, Tuple, Set, Optional
from openai import OpenAI


# How many shared nouns/adjectives with prior descriptions in the same round
# trigger immediate elimination. ">" means strict greater-than, so the player
# is safe at exactly REPEAT_LIMIT.
REPEAT_LIMIT = 2

# Reward bookkeeping (additive, accumulated per team):
#   * winning team: +1.0 base
#   * losing/draw/aborted team: 0.0 base (losing alone does not change reward)
#   * each repetition auto-elimination: -1 / remaining_same_team_player on that team
#   * each secret-word leak: LEAK_PENALTY on the leaker's team
LEAK_PENALTY = -0.8
WIN_REWARD = 1.0


def _description_leaks_word(description: str, word: str) -> bool:
    """True iff `description` contains `word` as a standalone token.

    Word-boundary match so 'bread' in 'breadcrumbs' does NOT trigger,
    but 'bread.' or 'bread,' does.
    """
    if not description or not word:
        return False
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", description.lower()) is not None

_POS_READY: bool = False  # flips to True after first successful smoke test


def _ensure_pos_tagger() -> None:
    """Verify NLTK + required data are available. Raises loudly on failure
    rather than silently disabling the rule (which previously hid the bug
    that all rewards collapsed to 0/1 because no auto-eliminations fired).
    """
    global _POS_READY
    if _POS_READY:
        return
    import nltk  # let ImportError propagate -- we want it loud
    for resource, pkg in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)
    # Smoke test: must succeed or we raise.
    nltk.pos_tag(nltk.word_tokenize("test sentence"))
    _POS_READY = True


def extract_content_words(text: str) -> Set[str]:
    """Return the lowercase set of noun and adjective tokens in `text`.

    Uses NLTK Penn Treebank tags: NN/NNS/NNP/NNPS for nouns, JJ/JJR/JJS for
    adjectives. Raises if NLTK isn't usable -- a silent empty return would
    mean the repetition rule never fires, collapsing all training rewards
    to 0/1 (the bug we just hunted down).
    """
    if not text:
        return set()
    _ensure_pos_tagger()
    import nltk
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    return {w for w, t in tagged if (t.startswith("NN") or t.startswith("JJ")) and w.isalpha()}


class SpyGamePlayer:
    """Represents a single player in the spy game."""

    def __init__(self, player_id: str, name: str, role: str, word: str,
                 base_url: str, api_key: str, model: str = "agentjet-model"):
        self.player_id = player_id
        self.name = name
        self.role = role  # "civilian" or "spy"
        self.word = word
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.is_alive = True
        self.descriptions: List[str] = []
        self.votes_received = 0

    def get_client(self) -> OpenAI:
        """Get OpenAI client for this player."""
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _format_history(self, game_history: List[Dict]) -> str:
        """Format game history grouped by round: descriptions, votes, and eliminations."""
        if not game_history:
            return ""

        rounds: Dict[int, Dict] = {}
        for entry in game_history:
            r = entry.get("round")
            if r is None:
                continue
            bucket = rounds.setdefault(
                r,
                {
                    "descriptions": [],
                    "votes": [],
                    "eliminations": [],
                    "had_elimination_phase": False,
                },
            )
            etype = entry.get("type")
            if etype == "description":
                bucket["descriptions"].append(entry)
            elif etype == "vote":
                bucket["votes"].append(entry)
            elif etype == "elimination":
                bucket["had_elimination_phase"] = True
                bucket["eliminations"].append(
                    {
                        "name": entry.get("eliminated_name"),
                        "role": entry.get("eliminated_role"),
                        "reason": entry.get("reason"),
                    }
                )

        def tag(name: str) -> str:
            return " (you)" if name == self.name else ""

        lines = []
        for r in sorted(rounds.keys()):
            b = rounds[r]
            lines.append(f"\n--- Round {r} ---")
            if b["descriptions"]:
                lines.append("Descriptions:")
                for entry in b["descriptions"]:
                    lines.append(
                        f"- {entry['player_name']}{tag(entry['player_name'])}: \"{entry['description']}\""
                    )
            if b["votes"]:
                lines.append("Votes:")
                for entry in b["votes"]:
                    voter = entry["voter_name"]
                    voted = entry["voted_name"]
                    # Reasons are private: only show the viewer's own reasoning,
                    # never another player's.
                    if voter == self.name:
                        reason = entry.get("reason") or ""
                        suffix = f' ("{reason}")' if reason else ""
                    else:
                        suffix = ""
                    lines.append(f"- {voter}{tag(voter)} -> {voted}{tag(voted)}{suffix}")
            if b["had_elimination_phase"]:
                if not b["eliminations"]:
                    lines.append("Result: no eliminations this round")
                else:
                    for rec in b["eliminations"]:
                        if rec["name"] is None:
                            lines.append("Result: no plurality - no one eliminated by vote")
                        else:
                            role = rec["role"] or "unknown"
                            why = f' - {rec["reason"]}' if rec.get("reason") else ""
                            lines.append(
                                f"Result: {rec['name']}{tag(rec['name'])} was eliminated "
                                f"(was {role}){why}"
                            )
        return "\n".join(lines)

    def generate_description(self, round_num: int, game_history: List[Dict]) -> str:
        """
        Generate a description of the word without saying it directly.

        Args:
            round_num: Current round number
            game_history: List of previous descriptions and events

        Returns:
            Generated description string
        """
        client = self.get_client()

        history_block = self._format_history(game_history)
        history_context = f"\n\nGame history so far:\n{history_block}\n" if history_block else ""

        prompt = f"""You are playing a social deduction game called "Who is the Spy".

Your name: {self.name}
Your role: {self.role.upper()}
Your word: {self.word}

Game rules:
- Most players are CIVILIANS with the same word
- A few players are SPIES with a similar but different word
- Each round, players describe their word WITHOUT saying it directly
- After descriptions, players vote to eliminate someone suspicious
- CIVILIANS win if all spies are eliminated
- SPIES win if they equal or outnumber civilians

Current round: {round_num}
{history_context}

Your task: Generate a brief description (1-2 sentences) of your word that:
1. Accurately reflects your word
2. Helps your teammates identify you
3. CRITICAL: NEVER write the literal word "{self.word}" in your description.
   Saying it aborts the game and applies a {LEAK_PENALTY:+.1f} penalty to your
   team's reward.
4. Does NOT make your role too obvious if you're a spy
5. Stays consistent with anything you (marked "(you)" above) said in earlier rounds
6. IMPORTANT: If your description shares MORE THAN {REPEAT_LIMIT} nouns or adjectives
   with descriptions already given THIS ROUND by other players, you are eliminated
   immediately AND your team's reward is reduced by 1/(remaining teammates).
   Use original wording.

Output only the description, nothing else."""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic player in a social deduction game."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300,
            timeout=60
        )
        description = response.choices[0].message.content.strip()
        self.descriptions.append(description)
        return description

    def vote(self, alive_players: List['SpyGamePlayer'], game_history: List[Dict]) -> Tuple[str, str]:
        """
        Vote for the most suspicious player and explain why.

        Args:
            alive_players: List of players still in the game
            game_history: Full game history including descriptions

        Returns:
            (voted_name, reason) tuple
        """
        client = self.get_client()

        history_block = self._format_history(game_history)
        players_info = (
            f"\n\nPlayers and their descriptions:\n{history_block}\n" if history_block else ""
        )

        # Available players to vote for (alive, not self)
        available_players = [p.name for p in alive_players if p.name != self.name]
        players_list = ", ".join(available_players)

        prompt = f"""You are playing "Who is the Spy" game.

Your name: {self.name}
Your role: {self.role.upper()}
Your word: {self.word}
{players_info}
Available players to vote for: {players_list}

Your goal:
- If you're a CIVILIAN: Vote for the player who seems to have a DIFFERENT word (the spy)
- If you're a SPY: Vote strategically to survive and avoid suspicion

Analyze all descriptions carefully. Look for:
- Descriptions that don't quite match the majority
- Vague or contradictory descriptions
- Players who seem to be hiding something

Respond in EXACTLY this two-line format and nothing else:
Reason: <one short sentence explaining your choice>
Vote: <one name from {players_list}>"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strategic player making voting decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=60
        )
        vote_text = response.choices[0].message.content.strip()
        return self._parse_vote(vote_text, available_players)

    @staticmethod
    def _parse_vote(vote_text: str, available_players: List[str]) -> Tuple[str, str]:
        """Resolve the model's vote text to (name, reason). Raises on failure
        so a malformed model output is loud instead of being papered over by a
        random pick.
        """
        reason = ""
        vote_line = vote_text
        for line in vote_text.splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("reason:"):
                reason = stripped[len("reason:"):].strip()
            elif low.startswith("vote:"):
                vote_line = stripped[len("vote:"):].strip()

        cleaned = vote_line.strip().strip(".,!?\"'").lower()
        for player_name in available_players:
            if player_name.lower() == cleaned:
                return player_name, reason
        for player_name in sorted(available_players, key=len, reverse=True):
            if player_name.lower() in vote_line.lower():
                return player_name, reason
        for player_name in sorted(available_players, key=len, reverse=True):
            if player_name.lower() in vote_text.lower():
                return player_name, reason
        raise ValueError(
            f"Could not parse a vote from model output. "
            f"available={available_players!r}  output={vote_text!r}"
        )


class SpyGame:
    """Main game engine for "Who is the Spy" game."""

    def __init__(self, civilian_word: str, spy_word: str,
                 num_players: int, num_spies: int,
                 player_configs: List[Dict]):
        """
        Initialize a spy game.

        Args:
            civilian_word: Word for civilians
            spy_word: Word for spies
            num_players: Total number of players
            num_spies: Number of spies
            player_configs: List of player configurations with base_url, api_key, model, name
        """
        if num_spies <= 0:
            raise ValueError(f"num_spies must be >= 1, got {num_spies}")
        if num_players - num_spies <= num_spies:
            raise ValueError(
                f"Need strictly more civilians than spies at start "
                f"(got {num_spies} spies, {num_players - num_spies} civilians); "
                f"otherwise spies win before anyone plays."
            )
        if len(player_configs) != num_players:
            raise ValueError(
                f"player_configs has {len(player_configs)} entries but num_players={num_players}"
            )

        self.civilian_word = civilian_word
        self.spy_word = spy_word
        self.num_players = num_players
        self.num_spies = num_spies
        self.players: List[SpyGamePlayer] = []
        self.game_history: List[Dict] = []
        self.current_round = 0
        self.max_rounds = 10
        # Per-team additive reward adjustments accumulated mid-game.
        self.team_penalties: Dict[str, float] = {"civilian": 0.0, "spy": 0.0}
        # Set to "civilian" or "spy" if a player from that team spoke their
        # own secret word -- aborts the game and zeros that team's reward.
        self.aborted_by_role: Optional[str] = None

        # Assign roles randomly
        roles = ["spy"] * num_spies + ["civilian"] * (num_players - num_spies)
        random.shuffle(roles)

        # Create players
        for i, (role, config) in enumerate(zip(roles, player_configs)):
            word = spy_word if role == "spy" else civilian_word
            player = SpyGamePlayer(
                player_id=f"player_{i}",
                name=config["name"],
                role=role,
                word=word,
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config.get("model", "agentjet-model")
            )
            self.players.append(player)

    def get_alive_players(self) -> List[SpyGamePlayer]:
        """Get list of players still in the game."""
        return [p for p in self.players if p.is_alive]

    def _check_decisive_end(self) -> Tuple[bool, str, float]:
        """Check only team-count win conditions (ignores max-rounds clock).

        Used mid-round so that hitting max_rounds during the description phase
        does not skip the round's voting phase.
        """
        alive = self.get_alive_players()
        alive_spies = [p for p in alive if p.role == "spy"]
        alive_civilians = [p for p in alive if p.role == "civilian"]

        if len(alive_spies) == 0:
            return True, "civilians", 1.0
        if len(alive_spies) >= len(alive_civilians):
            return True, "spies", 0.0
        return False, "", 0.5

    def check_game_end(self) -> Tuple[bool, str, float]:
        """
        Check if game has ended and determine winner.

        Returns:
            (is_ended, winner, civilian_team_reward)
            winner: "civilians", "spies", or "draw"
            civilian_team_reward: 1.0 if civilians win, 0.0 if spies win, 0.5 for draw
        """
        decisive, winner, reward = self._check_decisive_end()
        if decisive:
            return decisive, winner, reward

        # Draw if max rounds reached.
        if self.current_round >= self.max_rounds:
            return True, "draw", 0.5

        return False, "", 0.5

    def play_round(self) -> bool:
        """
        Play one round of the game.

        Returns:
            True if game should continue, False if game ended
        """
        self.current_round += 1
        print(f"\n{'='*60}")
        print(f"ROUND {self.current_round}")
        print(f"{'='*60}")

        alive_players = self.get_alive_players()

        # Phase 1: Description phase
        print("\n--- Description Phase ---")
        round_descriptions = []
        prior_words: Set[str] = set()  # nouns + adjectives already used this round

        for player in alive_players:
            if not player.is_alive:
                continue
            description = player.generate_description(self.current_round, self.game_history)
            print(f"{player.name} ({player.role}): \"{description}\"")

            entry = {
                "type": "description",
                "round": self.current_round,
                "player_id": player.player_id,
                "player_name": player.name,
                "role": player.role,
                "description": description,
                "repeated_words": [],
            }
            self.game_history.append(entry)
            round_descriptions.append(entry)

            # Catastrophic rule: speaking your own secret word aborts the
            # game and applies LEAK_PENALTY to the leaker's team reward.
            if _description_leaks_word(description, player.word):
                player.is_alive = False
                self.aborted_by_role = player.role
                self.team_penalties[player.role] += LEAK_PENALTY
                reason = (
                    f"GAME ABORTED: {player.name} ({player.role}) said their own "
                    f"secret word '{player.word}' literally. "
                    f"Team penalty: {LEAK_PENALTY:+.2f}."
                )
                print(f"  -> {reason}")
                self.game_history.append({
                    "type": "elimination",
                    "round": self.current_round,
                    "eliminated_name": player.name,
                    "eliminated_role": player.role,
                    "votes_received": 0,
                    "reason": reason,
                    "aborted": True,
                    "team_penalty_applied": LEAK_PENALTY,
                })
                return False  # stop the round AND the game

            own_words = extract_content_words(description)
            repeated = sorted(own_words & prior_words)
            entry["repeated_words"] = repeated

            # Compare against earlier players ONLY, then add own words to the pool.
            if len(repeated) > REPEAT_LIMIT:
                player.is_alive = False
                # Penalty: -1 / number of remaining same-team players AFTER this
                # elimination. If this was the last teammate, skip (the team
                # has already lost; no one to spread the penalty across).
                remaining_same = sum(
                    1 for p in self.players if p.is_alive and p.role == player.role
                )
                applied = 0.0
                if remaining_same > 0:
                    applied = -1.0 / remaining_same
                    self.team_penalties[player.role] += applied
                reason = (
                    f"Auto-eliminated: shared {len(repeated)} nouns/adjectives "
                    f"with earlier descriptions this round: {repeated}. "
                    f"Team penalty: {applied:+.4f} "
                    f"(remaining {player.role}s: {remaining_same})"
                )
                print(f"  -> {player.name} eliminated immediately. {reason}")
                self.game_history.append({
                    "type": "elimination",
                    "round": self.current_round,
                    "eliminated_name": player.name,
                    "eliminated_role": player.role,
                    "votes_received": 0,
                    "reason": reason,
                    "team_penalty_applied": applied,
                })
                # End the round only if a team has actually won; do not exit
                # mid-round on the max-rounds clock (voting phase still owed).
                is_ended, _, _ = self._check_decisive_end()
                if is_ended:
                    return False

            prior_words |= own_words

        # Refresh alive list after possible auto-eliminations.
        alive_players = self.get_alive_players()
        if not alive_players:
            return False

        # Phase 2: Voting phase
        print("\n--- Voting Phase ---")
        votes: Dict[str, List[str]] = {p.name: [] for p in alive_players}

        for player in alive_players:
            voted_name, reason = player.vote(alive_players, self.game_history)
            votes[voted_name].append(player.name)
            print(f"{player.name} votes for: {voted_name} -- {reason}")

            self.game_history.append({
                "type": "vote",
                "round": self.current_round,
                "voter_name": player.name,
                "voted_name": voted_name,
                "reason": reason,
            })

        # Determine who gets eliminated.
        max_votes = max(len(v) for v in votes.values())
        candidates = [name for name, voters in votes.items() if len(voters) == max_votes]

        if len(candidates) == len(alive_players):
            # No plurality at all (everyone tied) -> skip elimination this round
            # rather than eliminating a random player on zero signal.
            print("\nNo plurality this round - no one is eliminated.")
            self.game_history.append({
                "type": "elimination",
                "round": self.current_round,
                "eliminated_name": None,
                "eliminated_role": None,
                "votes_received": 0,
            })
        else:
            if len(candidates) > 1:
                eliminated_name = random.choice(candidates)
            else:
                eliminated_name = candidates[0]

            eliminated_player = next(p for p in alive_players if p.name == eliminated_name)
            eliminated_player.is_alive = False

            print(f"\n{eliminated_name} ({eliminated_player.role}) has been eliminated!")
            print(f"Their word was: {eliminated_player.word}")

            self.game_history.append({
                "type": "elimination",
                "round": self.current_round,
                "eliminated_name": eliminated_name,
                "eliminated_role": eliminated_player.role,
                "votes_received": len(votes[eliminated_name])
            })

        # Check game end condition
        is_ended, winner, _ = self.check_game_end()
        return not is_ended

    def play_game(self) -> Dict:
        """
        Play the full game until completion.

        Returns:
            Game result dictionary with winner, rewards, and history
        """
        print(f"\n{'#'*60}")
        print(f"GAME START")
        print(f"Civilian word: {self.civilian_word}")
        print(f"Spy word: {self.spy_word}")
        print(f"Players: {self.num_players}, Spies: {self.num_spies}")
        print(f"{'#'*60}")

        # Print initial player assignments
        print("\nPlayers:")
        for player in self.players:
            print(f"  {player.name}: {player.role} (word: {player.word})")

        # Play rounds until game ends
        while self.play_round():
            pass

        # Determine winner. Aborted games have no winner (offender team did not
        # "win the game"; opposing team also did not "win the game" -- the game
        # was stopped early because the leak is too bad).
        if self.aborted_by_role is not None:
            winner = "aborted"
            civilian_base = 0.0
            spy_base = 0.0
        else:
            _is_ended, winner, _ = self.check_game_end()
            civilian_base = WIN_REWARD if winner == "civilians" else 0.0
            spy_base = WIN_REWARD if winner == "spies" else 0.0

        # Final reward = win-base + accumulated mistake penalties.
        # Losing the game alone does not change reward; only mistakes do.
        civilian_reward = civilian_base + self.team_penalties["civilian"]
        spy_reward = spy_base + self.team_penalties["spy"]

        print(f"\n{'#'*60}")
        print(f"GAME END - {winner.upper()}")
        print(
            f"civilian_reward={civilian_reward:+.4f} "
            f"(base={civilian_base:+.2f}, penalty={self.team_penalties['civilian']:+.4f}) "
            f"spy_reward={spy_reward:+.4f} "
            f"(base={spy_base:+.2f}, penalty={self.team_penalties['spy']:+.4f})"
        )
        if self.aborted_by_role:
            print(f"Aborted by team: {self.aborted_by_role}")
        print(f"{'#'*60}")

        # Calculate individual rewards
        player_rewards = {}
        for player in self.players:
            player_rewards[player.name] = (
                civilian_reward if player.role == "civilian" else spy_reward
            )

        return {
            "winner": winner,
            "civilian_reward": civilian_reward,
            "spy_reward": spy_reward,
            "team_penalties": dict(self.team_penalties),
            "aborted_by_role": self.aborted_by_role,
            "player_rewards": player_rewards,
            "total_rounds": self.current_round,
            "game_history": self.game_history,
            "final_alive": [p.name for p in self.get_alive_players()],
            "civilian_word": self.civilian_word,
            "spy_word": self.spy_word
        }
