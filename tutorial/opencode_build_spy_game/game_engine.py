import random
from typing import List, Dict, Tuple
from openai import OpenAI


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

        # Build context from game history
        history_context = ""
        if game_history:
            history_context = "\n\nPrevious descriptions from other players:\n"
            for entry in game_history:
                if entry.get("type") == "description":
                    history_context += f"- {entry['player_name']}: \"{entry['description']}\"\n"

        prompt = f"""You are playing a social deduction game called "Who is the Spy".

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
3. Does NOT reveal your word directly
4. Does NOT make your role too obvious if you're a spy

Output only the description, nothing else."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic player in a social deduction game."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=150,
                timeout=60
            )
            description = response.choices[0].message.content.strip()
            self.descriptions.append(description)
            return description
        except Exception as e:
            print(f"Error generating description for {self.name}: {e}")
            # Fallback description
            fallback = f"It's something related to {self.word[0]}... things."
            self.descriptions.append(fallback)
            return fallback

    def vote(self, alive_players: List['SpyGamePlayer'], game_history: List[Dict]) -> str:
        """
        Vote for the most suspicious player.

        Args:
            alive_players: List of players still in the game
            game_history: Full game history including descriptions

        Returns:
            Name of the player to vote for
        """
        client = self.get_client()

        # Build player list and their descriptions
        players_info = "\n\nPlayers and their descriptions:\n"
        for entry in game_history:
            if entry.get("type") == "description":
                players_info += f"- {entry['player_name']}: \"{entry['description']}\"\n"

        # Available players to vote for
        available_players = [p.name for p in alive_players if p.name != self.name]
        players_list = ", ".join(available_players)

        prompt = f"""You are playing "Who is the Spy" game.

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

Output ONLY the name of the player you want to vote for (choose from: {players_list})
Do not include any explanation, just the name."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic player making voting decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50,
                timeout=60
            )
            vote_text = response.choices[0].message.content.strip()

            # Extract valid player name from response
            for player_name in available_players:
                if player_name.lower() in vote_text.lower():
                    return player_name

            # Fallback: random vote
            return random.choice(available_players)

        except Exception as e:
            print(f"Error generating vote for {self.name}: {e}")
            return random.choice(available_players)


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
        self.civilian_word = civilian_word
        self.spy_word = spy_word
        self.num_players = num_players
        self.num_spies = num_spies
        self.players: List[SpyGamePlayer] = []
        self.game_history: List[Dict] = []
        self.current_round = 0
        self.max_rounds = 10

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

    def check_game_end(self) -> Tuple[bool, str, float]:
        """
        Check if game has ended and determine winner.

        Returns:
            (is_ended, winner, civilian_team_reward)
            winner: "civilians", "spies", or "draw"
            civilian_team_reward: 1.0 if civilians win, 0.0 if spies win, 0.5 for draw
        """
        alive = self.get_alive_players()
        alive_spies = [p for p in alive if p.role == "spy"]
        alive_civilians = [p for p in alive if p.role == "civilian"]

        # Spies win if they equal or outnumber civilians
        if len(alive_spies) >= len(alive_civilians):
            return True, "spies", 0.0

        # Civilians win if all spies are eliminated
        if len(alive_spies) == 0:
            return True, "civilians", 1.0

        # Draw if max rounds reached
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
        for player in alive_players:
            description = player.generate_description(self.current_round, self.game_history)
            print(f"{player.name} ({player.role}): \"{description}\"")

            entry = {
                "type": "description",
                "round": self.current_round,
                "player_id": player.player_id,
                "player_name": player.name,
                "role": player.role,
                "description": description
            }
            self.game_history.append(entry)
            round_descriptions.append(entry)

        # Check if game should end before voting
        is_ended, winner, _ = self.check_game_end()
        if is_ended:
            return False

        # Phase 2: Voting phase
        print("\n--- Voting Phase ---")
        votes: Dict[str, List[str]] = {p.name: [] for p in alive_players}

        for player in alive_players:
            voted_name = player.vote(alive_players, self.game_history)
            votes[voted_name].append(player.name)
            print(f"{player.name} votes for: {voted_name}")

            self.game_history.append({
                "type": "vote",
                "round": self.current_round,
                "voter_name": player.name,
                "voted_name": voted_name
            })

        # Determine who gets eliminated
        max_votes = max(len(v) for v in votes.values())
        candidates = [name for name, voters in votes.items() if len(voters) == max_votes]

        if len(candidates) > 1:
            # Tie - randomly eliminate one
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

        # Determine final result
        is_ended, winner, civilian_reward = self.check_game_end()

        print(f"\n{'#'*60}")
        print(f"GAME END - {winner.upper()} WIN!")
        print(f"{'#'*60}")

        # Calculate individual rewards
        player_rewards = {}
        for player in self.players:
            if player.role == "civilian":
                player_rewards[player.name] = civilian_reward
            else:  # spy
                player_rewards[player.name] = 1.0 - civilian_reward

        return {
            "winner": winner,
            "civilian_reward": civilian_reward,
            "spy_reward": 1.0 - civilian_reward,
            "player_rewards": player_rewards,
            "total_rounds": self.current_round,
            "game_history": self.game_history,
            "final_alive": [p.name for p in self.get_alive_players()],
            "civilian_word": self.civilian_word,
            "spy_word": self.spy_word
        }
