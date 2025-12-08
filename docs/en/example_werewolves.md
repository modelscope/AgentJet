# Werewolves

Werewolves role playing game is a typical POMDP (Partially Observable Markov Decision Process) problem. We can train agents in this cooperative multi-agent problem using shared-parameter methods.

Terms explained:
- **Partially Observable**: Agents are only able to receive **local information**. One agent cannot obtain others' perception even if they are teammate.
- **Markov Decision Process**: Making decisions according to current situations.
- **Shared-parameter**: Using one model as policy for multiple agents. But notice agents **share** policy (model parameters) but **do not shared** perception (model input).
- **Cooperative multi-agent problem**: Agents have aligned interests (reward).
- **Environment**: We use static **`Qwen3-235B-A22B`** as the brain of opponents. We use **`Qwen2-7B`** as the brain of trainable agents (`trainable_targets`).

![image](https://img.alicdn.com/imgextra/i2/O1CN012JgVZC2ABczBhAzJs_!!6000000008165-0-tps-2048-2048.jpg)

This page shows how to use the Werewolves social deduction game as a multi-agent environment to prepare data and environment, write an AgentScope Workflow, configure the reward module (Judge), and complete the full process from local debugging to formal training.

1. Scenario Overview
- Scenario: Classic Werewolves game, including roles such as werewolf, villager, seer, witch, and hunter.
- Goal: Train a specific role (in this example, the witch) to achieve a higher win rate in games.
Below we will go through how to prepare the AgentScope Workflow, configure the YAML file, debug, and start training.

2. Prepare the AgentScope Workflow
The example code is located at tutorial/example_werewolves/start.py.
You can also create your own AgentScope Workflow anywhere in the project, as long as the entry point is correctly configured in the YAML.
First, define the AgentScope workflow:
```
class ExampleWerewolves(AgentScopeLearnProtocol):

    async def agentscope_execute(self, init_messages, astune_proxy: ModelTuner, config) -> WorkflowOutput:
        train_which_role = "werewolf"
        roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]

        # Set random seed for reproducibility
        workflow_task = astune_proxy.get_agentscope_input_dictionary()["workflow_task"]
        task_id = workflow_task.task.task_id

        np.random.seed(int(task_id))
        np.random.shuffle(roles)

        players = [
            get_official_agents(
                f"Player{x + 1}", roles[x], train_which_role, astune_proxy
            )
            for x in range(9)
        ]

        good_guy_win = await werewolves_game(players, roles)
        raw_reward = 1 if (good_guy_win and train_which_role != "werewolf") or (
            not good_guy_win and train_which_role == "werewolf"
        ) else 0

        astune_proxy.update_judge_input_dictionary(raw_reward=raw_reward)
        astune_proxy.update_judge_input_dictionary(is_success=(raw_reward == 1))
        return astune_proxy
```

3. Configuration
3.1 Configure YAML
This section corresponds to tutorial/example_werewolves/werewolves.yaml.
You can copy and modify the key parameters in that file. The parts most relevant to this document are marked with ✨✨✨✨ in the YAML.
The key configuration items are as follows:
```yaml
astuner:
  task_reader:
    # random seed to shuffle players
    type: random_dummy
  task_judge:
    # ✨✨✨✨ Implement and select the evaluation function
    # (in this example you can first set it to null and rely purely on the rollout's internal reward)
    judge_protocol: null
  model:
    # ✨✨✨✨ Set the model to be trained
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ Implement and select the AgentScope Workflow entry
    agentscope_learn_protocol: tutorial.example_werewolves.start->ExampleWerewolves
```
You can add or replace your own Workflow / Judge / Model following the structure above.
Just make sure that the paths and class names are consistent with the actual code.

3.2 Debugging
Before running formal training, it is recommended to use --backbone='debug' for fast single-machine debugging without Ray:
```
# It is recommended to kill all processes related to ray, env_service, and vllm before starting
# ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf tutorial/example_werewolves/werewolves.yaml --backbone='debug' --with-logview
```
When --backbone=debug, the program runs without Ray, making it convenient to do breakpoint debugging on your local machine. You can configure launch.json in VSCode as follows:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Launch rollout",
      "type": "debugpy",
      "request": "launch",
      "program": "launcher.py",
      "console": "integratedTerminal",
      "args": [
        "--backbone", "debug",
        "--conf", "tutorial/example_werewolves/werewolves.yaml"
      ],
      "env": {}
    }
  ]
}
```

4. Start Training
After you finish validating in Debug mode, simply switch backbone to trinity and enable Ray to launch formal training:
```
# It is recommended to kill all processes related to ray, vllm, and env_service before starting
# ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf tutorial/example_werewolves/werewolves.yaml --backbone='trinity' --with-ray
```



## result

Qwen2-7B is able to reach 60% percent win rate in about 20 steps.
![image](https://img.alicdn.com/imgextra/i3/O1CN01ldZYDT1ZqGLHuwsrS_!!6000000003245-2-tps-2000-839.png)

### Behavior Shifts


Significant role-playing improvement is observed during the experiment.
1. For example, when voted out, the original model trends to reveal its identity as `werewolf`, but fine-tuning, agent will try to cheat its opponents and protect teammates. For example:


![](https://img.alicdn.com/imgextra/i1/O1CN01v8VqLB1aYEMfzyTHr_!!6000000003341-2-tps-2104-1016.png)

2. Agent develop multiple strategy for winning.

For example:
- **Misleding opponents**: "Let's keep an eye on the seer and the witch. They could be werewolves trying to hide".
- **Appealing to reson**: "We need to be wary of fake seers and watch for inconsistencies in stories, Player-Y as hunter should act carefully".

3. Sometime agents can take advantage of suspect between non-werewolf players to eliminate opponents.

![](https://img.alicdn.com/imgextra/i2/O1CN01Sx7wkU23pHyPXyqPH_!!6000000007304-2-tps-968-575.png)

### Expanding Qwen2-7B to Qwen2-14B

![](https://img.alicdn.com/imgextra/i1/O1CN01TLZcQF1FJ1HPbpLfj_!!6000000000465-2-tps-1842-1008.png)
