# -*- coding: utf-8 -*-
"""
This file defines a multi-step workflow for the FrozenLake environment.
Modified from https://github.com/rllm-org/rllm/blob/main/rllm/environments/frozenlake/frozenlake.py
"""

from __future__ import annotations

import copy
import re
import traceback
from typing import Dict, Optional, Tuple

import numpy as np
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from loguru import logger

from ajet import ModelTuner, Workflow, WorkflowOutput, WorkflowTask

SYSTEM_PROMPT = """You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your short thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""


class FrozenLakeWorkflow(Workflow):
    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        config = model_tuner.config

        self.env_max_steps = config.ajet.rollout.multi_turn.max_steps
        self.agent_max_steps = config.ajet.rollout.multi_turn.max_steps

        # Extract task-specific arguments
        self.raw_task = workflow_task.task.metadata

        self.size = config.frozen_lake.frozen_lake_size
        self.seed = workflow_task.task.metadata["random_number"]
        self.p = 0.8  # Probability that a tile is frozen

        # Agent-related state
        self.step_count: int = 0

        # init agent and environment
        self.agent = FrozenLakeAgent(
            model=model_tuner,
            max_steps=self.agent_max_steps,
        )
        self.env = FrozenLakeEnv(
            max_steps=self.env_max_steps,
            is_slippery=config.frozen_lake.is_slippery,
            size=self.size,
            p=self.p,  # Probability that a tile is frozen
            seed=self.seed,
        )

        return await self.run_frozenlake()

    async def run_frozenlake(self):
        self.env.reset(self.raw_task)
        terminate_reason = None
        observation_str = str(self.env.render())
        rewards = []
        step_count = 0
        done = False
        for _ in range(self.agent_max_steps):
            step_count += 1
            try:
                action = await self.agent.step(current_observation=observation_str)
            except Exception:
                logger.error(
                    f"Agent failed to produce action due to error:\n{traceback.format_exc()}"
                )
                terminate_reason = "agent_error"
                break
            observation, reward, done, _ = self.env.step(action)
            observation_str = str(observation)
            rewards.append(reward)
            if done:
                terminate_reason = "success"
                break

        if terminate_reason is None:
            terminate_reason = "max_steps_reached"

        final_reward = sum(rewards)
        return WorkflowOutput(
            reward=final_reward,
            metadata={
                "terminate_reason": terminate_reason,
                "step_count": step_count,
            },
        )


class FrozenLakeAgent:
    INVALID_ACTION = "still"

    def __init__(self, model: ModelTuner, max_steps: int = 20):
        self.agent = ReActAgent(
            name="frozenlake_agent",
            sys_prompt=SYSTEM_PROMPT,
            model=model,
            formatter=DashScopeChatFormatter(),
            max_iters=2,
        )
        self.agent.set_console_output_enabled(False)
        self.current_step = 0
        self.last_action = None
        self.last_observation = None
        self.max_steps = max_steps

    def get_prompt(self, observation: str) -> str:
        prompt = (
            f"Current Observation ({self.current_step}): \n"
            + observation
            + "\n"
            + "You have not achieved the goal, P has not reached G yet. Please give the next action."
        )
        if self.current_step > 0 and self.last_action is not None:
            if self.last_observation == observation:
                prompt += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."

        if self.max_steps is not None and self.max_steps - self.current_step > 0:
            prompt += (
                f"\nThe maximum number of steps remaining is {self.max_steps - self.current_step}."
            )

        return prompt

    def get_action(self, msg: Msg) -> str:
        logger.info(f"Agent response: {msg.content}")
        response: str = msg.content if isinstance(msg.content, str) else msg.content[0].get("text")
        action = self.INVALID_ACTION

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            action = last_match_content.lower()
            if action not in ["up", "down", "left", "right"]:
                action = self.INVALID_ACTION

        return action

    async def step(self, current_observation: str) -> str:
        prompt = self.get_prompt(current_observation)
        msg = await self.agent.reply(Msg("user", prompt, role="user"))
        action = self.get_action(msg)
        self.last_observation = current_observation
        self.last_action = action
        self.current_step += 1
        return action


class FrozenLakeEnv(GymFrozenLakeEnv):
    # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,
        b"F": 1,
        b"H": 2,
        b"G": 3,
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        "still": 0,
        "left": 1,
        "down": 2,
        "right": 3,
        "up": 4,
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(
        self,
        max_steps: int = 8,
        is_slippery: bool = False,
        size: int = 8,
        p: float = 0.8,
        seed: int = 42,
    ):
        self.max_steps = max_steps or 8
        self.is_slippery = is_slippery
        self.size = size
        self.p = p
        self.seed = seed
        try:
            import gymnasium as gym
            from gymnasium.envs.toy_text.frozen_lake import (
                FrozenLakeEnv as GymFrozenLakeEnv,
            )
        except ImportError as e:
            error_message = (
                f"Gymnasium is not installed. Please install gymnasium first before "
                f"running the frozen_lake workflow. Error: {str(e)}"
            )
            logger.error(error_message)
            raise ImportError(error_message)

        random_map, goal_position = generate_random_map(
            size=self.size, p=self.p, seed=self.seed, max_steps=self.max_steps
        )

        self.goal_position = goal_position

        GymFrozenLakeEnv.__init__(self, desc=random_map[:], is_slippery=self.is_slippery)
        self.action_space = gym.spaces.Discrete(4, start=1)

        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": None,
            "seed": seed,
        }

        self.action_map = {
            1: 0,  # left
            2: 1,  # down
            3: 2,  # right
            4: 3,  # up
        }

    def _get_player_position(self) -> Tuple[int, int]:
        return (self.s // self.ncol, self.s % self.ncol)  # (row, col)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute a step in the environment.

        Maps custom action to gymnasium FrozenLakeEnv action and takes the step.
        Checks if the action is effective (whether player moves in the env).

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.success():
            return self.render(), 1, True, {"action_is_effective": False}

        action_id: int = self.ACTION_LOOKUP.get(action.lower(), 0)

        if not action_id:
            action_id = self.INVALID_ACTION

        if action_id == self.INVALID_ACTION or action_id not in self.action_map:
            return self.render(), 0, False, {"action_is_effective": False}

        prev_player_position = int(self.s)

        player_pos, reward, done, _, _ = GymFrozenLakeEnv.step(self, self.action_map[action_id])

        obs = self.render()
        return obs, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}

    def render(self, mode="tiny_rgb_array"):
        """Render the environment.

        Args:
            mode: Rendering mode. Options: "tiny_rgb_array", "list", "state", "rgb_array", "ansi".

        Returns:
            Rendered observation based on the mode.
        """
        assert mode in ["tiny_rgb_array", "list", "state", "rgb_array", "ansi"]
        if mode in ["rgb_array", "ansi"]:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b"S")
        room_state[position_S] = b"F"

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b"P"

        if mode == "state":
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b"H":
                room_state[position_P] = 4
            elif self.desc[position_P] == b"G":
                room_state[position_P] = 5
            return room_state

        room_state = self.render(mode="state").tolist()

        if mode == "list":

            def lookup(cell):
                return self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()

            return [" ".join(lookup(cell) for cell in row) for row in room_state]

        if mode == "tiny_rgb_array":

            def lookup(cell):
                return self.GRID_LOOKUP.get(cell, "?")

            result = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
            return result

    def reset(self, task: Optional[Dict] = None):
        task = task or {}
        self.__init__(  # type: ignore [misc]
            size=task.get("size", self.map_kwargs["size"]),
            p=task.get("p", self.map_kwargs["p"]),
            seed=task.get("seed", self.env_kwargs["seed"]),
            is_slippery=task.get("is_slippery", self.env_kwargs["is_slippery"]),
        )
        GymFrozenLakeEnv.reset(self, seed=self.seed)
        return self.render(mode="tiny_rgb_array"), {}

    def finished(self) -> bool:
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"GH"  # type: ignore [index,operator]

    def success(self):
        """
        Check if the agent has reached the goal (G).
        """
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"G"


def is_valid(board: list[list[str]], max_size: int, max_steps: int) -> bool:
    """DFS to check that it's a valid path.

    Args:
        board: The board representation as a list of lists.
        max_size: Maximum size of the board.
        max_steps: Maximum number of steps allowed.

    Returns:
        True if there's a valid path from start to goal within max_steps, False otherwise.
    """
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0], 0))  # row, col steps
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c, steps = frontier.pop()
        if steps > max_steps:
            continue

        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new, steps + 1))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: int = 0, max_steps: int = 5
) -> Tuple[list[str], Tuple[int, int]]:
    """Generates a random valid map (one that has a path from start to goal).

    Args:
        size: Size of each side of the grid.
        p: Probability that a tile is frozen.
        seed: Seed to ensure the generation of reproducible maps.
        max_steps: Maximum number of steps allowed.

    Returns:
        A tuple containing a random valid map and the goal position (row, col).
    """
    valid = False
    board: list[list[str]] = []  # initialize to make pyright happy

    try:
        from gymnasium.utils import seeding

        np_random, _ = seeding.np_random(seed)
    except ImportError:
        raise ImportError(
            "Gymnasium is not installed. Please install gymnasium first before running the frozen_lake workflow."
        )

    # generate random start and end points
    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()

        while True:
            start_r = int(np_random.integers(0, size))
            start_c = int(np_random.integers(0, size))
            goal_r = int(np_random.integers(0, size))
            goal_c = int(np_random.integers(0, size))

            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break

        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"

        valid = is_valid(board, size, max_steps)
    return ["".join(x) for x in board], (goal_r, goal_c)


def get_goal_position(random_map: np.ndarray) -> Optional[Tuple[int, int]]:
    """Get the goal position from a random map.

    Args:
        random_map: The map as a numpy array.

    Returns:
        Tuple of (row, col) if goal found, None otherwise.
    """
    positions = np.argwhere(random_map == b"G")
    if positions.size == 0:
        return None  # G not found
    return tuple(positions[0])  # returns (row, col)
