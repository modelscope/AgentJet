# -*- coding: utf-8 -*-
# flake8: noqa: E501

"""The main entry point for the werewolf game."""

from typing import List
import numpy as np
import dotenv
dotenv.load_dotenv()

from textwrap import dedent

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter, OpenAIMultiAgentFormatter
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from loguru import logger
from pydantic import Field

from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask
from tutorial.example_werewolves.game import BadGuyException, werewolves_game


def get_official_agent_prompt(name) -> str:
    system_prompt = dedent(
        f"""
        You're a werewolf game player named {name}.

        # YOUR TARGET
        Your target is to win the game with your teammates as much as possible.

        # GAME RULES
        - In werewolf game, players are divided into three werewolves, three villagers, one seer, one hunter and one witch.
            - Werewolves: kill one player each night, and must hide identity during the day.
            - Villagers: ordinary players without special abilities, try to identify and eliminate werewolves.
                - Seer: A special villager who can check one player's identity each night.
                - Witch: A special villager with two one-time-use potions: a healing potion to save a player from being killed at night, and a poison to eliminate one player at night.
                - Hunter: A special villager who can take one player down with them when they are eliminated.
        - The game alternates between night and day phases until one side wins:
            - Night Phase
                - Werewolves choose one victim
                - Seer checks one player's identity
                - Witch decides whether to use potions
                - Moderator announces who died during the night
            - Day Phase
                - All players discuss and vote to eliminate one suspected player

        # GAME GUIDANCE
        - Try your best to win the game with your teammates, tricks, lies, and deception are all allowed, e.g. pretending to be a different role.
        - During discussion, don't be political, be direct and to the point.
        - The day phase voting provides important clues. For example, the werewolves may vote together, attack the seer, etc.
        ## GAME GUIDANCE FOR WEREWOLF
        - Seer is your greatest threat, who can check one player's identity each night. Analyze players' speeches, find out the seer and eliminate him/her will greatly increase your chances of winning.
        - In the first night, making random choices is common for werewolves since no information is available.
        - Pretending to be other roles (seer, witch or villager) is a common strategy to hide your identity and mislead other villagers in the day phase.
        - The outcome of the night phase provides important clues. For example, if witch uses the healing or poison potion, if the dead player is hunter, etc. Use this information to adjust your strategy.
        ## GAME GUIDANCE FOR SEER
        - Seer is very important to villagers, exposing yourself too early may lead to being targeted by werewolves.
        - Your ability to check one player's identity is crucial.
        - The outcome of the night phase provides important clues. For example, if witch uses the healing or poison potion, if the dead player is hunter, etc. Use this information to adjust your strategy.
        ## GAME GUIDANCE FOR WITCH
        - Witch has two powerful potions, use them wisely to protect key villagers or eliminate suspected werewolves.
        - The outcome of the night phase provides important clues. For example, if the dead player is hunter, etc. Use this information to adjust your strategy.
        ## GAME GUIDANCE FOR HUNTER
        - Using your ability in day phase will expose your role (since only hunter can take one player down)
        - The outcome of the night phase provides important clues. For example, if witch uses the healing or poison potion, etc. Use this information to adjust your strategy.
        ## GAME GUIDANCE FOR VILLAGER
        - Protecting special villagers, especially the seer, is crucial for your team's success.
        - Werewolves may pretend to be the seer. Be cautious and don't trust anyone easily.
        - The outcome of the night phase provides important clues. For example, if witch uses the healing or poison potion, if the dead player is hunter, etc. Use this information to adjust your strategy.

        # NOTE
        - [IMPORTANT] DO NOT make up any information that is not provided by the moderator or other players.
        - This is a TEXT-based game, so DO NOT use or make up any non-textual information.
        - Always critically reflect on whether your evidence exist, and avoid making assumptions.
        - Your response should be specific and concise, provide clear reason and avoid unnecessary elaboration.
        - Generate your one-line response by using the `generate_response` function.
        - Don't repeat the others' speeches."""
    )
    return system_prompt


class ExampleWerewolves(Workflow):
    trainable_targets: List[str] | None = Field(default=["werewolf"], description="List of agents to be fine-tuned.")

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:

        # ensure trainable targets is legal
        assert self.trainable_targets is not None, "trainable_targets cannot be None in ExampleWerewolves (because we want to demonstrate a explicit multi-agent case)."

        # bad guys and good guys cannot be trained simultaneously
        # (because mix-cooperation-competition MARL needs too many advanced techniques to be displayed here)
        if "werewolf" in self.trainable_targets:
            assert len(self.trainable_targets) == 1, "Cannot train hostile roles simultaneously."
        else:
            assert len(self.trainable_targets) != 0, "No trainable targets specified."

        # make and shuffle roles (fix random seed for reproducibility)
        roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
        task_id = workflow_task.task.metadata["random_number"]
        np.random.seed(int(task_id))
        np.random.shuffle(roles)

        # initialize agents
        players = []
        for i, role in enumerate(roles):
            default_model = OpenAIChatModel(
                model_name="/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3-235B-A22B-Instruct-2507/",
                stream=False,
                client_args={"base_url": "http://22.17.52.4:2888/v1"},
                api_key="no_api_key",
                generate_kwargs={"temperature": 0.01},
            )
            model_for_this_agent = tuner.as_agentscope_model(
                agent_name=f"Player{i + 1}",    # the name of this agent
                target_tag=role,                # `target_tag in self.trainable_targets` means we train this agent, otherwise we do not train this agent.
                debug_model=default_model,      # the model used when this agent is not in `self.trainable_targets`
            )
            agent = ReActAgent(
                name=f"Player{i + 1}",
                sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
                model=model_for_this_agent,
                formatter=DashScopeMultiAgentFormatter()
                     if role in self.trainable_targets
                     else OpenAIMultiAgentFormatter(),
                max_iters=3 if role in self.trainable_targets else 5,
            )
            # agent.set_console_output_enabled(False)
            players += [agent]

        # reward condition
        try:
            good_guy_win = await werewolves_game(players, roles)
            raw_reward = 0
            is_success = False
            if (good_guy_win and self.trainable_targets[0] != "werewolf") or (
                not good_guy_win and self.trainable_targets[0] == "werewolf"
            ):
                raw_reward = 1
                is_success = True
            logger.warning(f"Raw reward: {raw_reward}")
            logger.warning(f"Is success: {is_success}")
        except BadGuyException as e:
            logger.bind(exception=True).exception(
                f"Error during game execution. Game cannot continue, whatever the cause, let's punish trainable agents  (Although they maybe innocent)."
            )
            raw_reward = -0.1
            is_success = False
        except Exception as e:
            logger.bind(exception=True).exception(
                f"Error during game execution. Game cannot continue, whatever the cause, let's punish trainable agents  (Although they maybe innocent)."
            )
            raw_reward = -0.1
            is_success = False

        return WorkflowOutput(reward=raw_reward, is_success=is_success)
