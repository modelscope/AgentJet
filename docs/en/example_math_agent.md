# Math Agent
This page demonstrates how to prepare data, build an Agent and Workflow, set up Rewards, and finally train a Math Agent from scratch.

## 1. Prepare Dataset
Download the `openai/gsm8k` dataset

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

## 2. Prepare AgentScope Workflow
See details in `tutorial/example_math_agent/math_agent.py`. You can create new AgentScope Workflow code anywhere in the project.

+ Define the AgentScope workflow

```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)
self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=astune_proxy,  # type: ignore
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)
msg = Msg("user", init_messages[0]['content'], role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)
```

In the AgentScope Workflow, register any key data needed by the evaluation function.

```python
astune_proxy.update_judge_input_dictionary(final_answer=final_answer)
```



## 3. Prepare Reward
Two simple Judges are provided in `astune/task_judge/math_answer_as_judge.py`. You can create new Judge code anywhere in the project.

Judge input parameters include:

```python
judge_input_dictionary['env']: env_service external environment (if env_service is used)
judge_input_dictionary['workflow_task']: Task information (if reference answer is included, it can be retrieved from here)
judge_input_dictionary['grouped_steps']: History of every LLM conversation turn (if intermediate processes are important, they can be retrieved from here)
judge_input_dictionary['final_answer']: By default, there is no final_answer. You need to manually call astune_proxy.update_judge_input_dictionary(final_answer=final_answer) in the agentscope workflow to register it.
```

Judge return values:

+ raw_reward
+ is_success

## 4. Start Training
### 4.1 Configure
Copy and modify key parameters in [tutorial/example_math_agent/math_agent.yaml](../tutorial/example_math_agent/math_agent.yaml). The most relevant parts in the yaml file are marked with ✨✨✨✨ symbols.

1. Read task (corresponds to configuration field `astune.task_reader`)
2. Define Workflow (corresponds to configuration field `astune.rollout.agentscope_learn_protocol`)
    - Example: If agentscope workflow is defined in `ExampleMathLear` class of `tutorial/math_agent.py`
    - Then set `astune.rollout.agentscope_learn_protocol`=`tutorial.math_agent->ExampleMathLearn`
3. Define scoring function (corresponds to configuration field `astune.task_judge.judge_protocol`)
    - Example: If agentscope workflow is defined in `MathAnswerAndLlmAsJudge` class of `astune/task_judge/math_answer_as_judge.py`
    - Then set `astune.task_judge.judge_protocol`=`astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge`
4. Specify model (corresponds to configuration field `astune.model.path`)

```yaml
astune:
    task_reader:
        type: huggingface_dat_repo # ✨✨✨✨ `env_service` or `dataset_file` or `huggingface_dat_repo`
    rollout:
        use_agentscope_protocol: True
        agentscope_learn_protocol: tutorial.math_agent->ExampleMathLearn # ✨✨✨✨ Write and select Agent
    task_judge:
        # ✨✨✨✨ Write and select evaluation function
        judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge
    model:
        # ✨✨✨✨ Set the model to be trained
        path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
```



### 4.2 Debug
```bash
# It is recommended to kill all ray and env_service processes before starting ( python launcher.py --kill="python|ray" )
clear && \
python launcher.py --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
```

When --backbone=debug, the program no longer uses Ray. You can configure vscode's launch.json for convenient breakpoint debugging. launch.json configuration:

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
                "--backbone",  "debug",
                "--conf", "xxxx/xxxx/xxxx.yaml"
            ],
            "env": {
            }
        },
    ]
}
```



### 4.3 Start Training
```bash
# It is recommended to kill all ray, vllm, and env_service processes before starting ( python launcher.py --kill="python|ray|vllm" )
python launcher.py --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

## 5 Reference Result
<div align="center">
<img width="700" alt="image" src="https://github.com/user-attachments/assets/6777b4fb-b71b-4573-81eb-8dccb7bdd3d2"/>
    <p style="margin-top: 10px; color: #666; font-size: 14px;">
    <em>Training curve</em>
  </p>
</div>

## Tuning Observation

Agents are already able to deal with the majority of problems at the begining of the training process, however, small models inevitably produce many answers that does not satisfy standard `<tool_call>` format. In some problems, the agent writes python code to compute the answer but it is so eager to produce answer right away that it immediately guessed an answer and skipped tool_call execution phrase.


```
# bad case 1: forget to print result in python code.
<|im_start|>assistant
To find the height difference between Bella and Anne's sister, let's first determine their heights based on the information provided.
- Anne's height = 80 cm
- Anne is 2 times as tall as her sister, so Anne's sister's height = \( \frac{Anne's\ height}{2} \)
- Bella is 3 times as tall as Anne, so Bella's height = \( 3 \times Anne's\ height \)
We can calculate these values and then determine the height difference between Bella and Anne's sister.
<tool_call>
{"name": "execute_python_code", "arguments": {"code": "anne_height = 80\nsister_height = anne_height / 2\nbella_height = 3 * anne_height\nheight_difference = bella_height - sister_height\nheight_difference"}}
</tool_call><|im_end|>

# bad case 2: too impatient to wait for tool execution.
<|im_start|>assistant 
To find the number of board members who attended the meeting, let's break down the information given: 1. There are 40 chairs, each with a capacity of 2 people. 2. The total capacity of the chairs is \( 40 \times 2 = 80 \) people. 3. Since the number of board members is the same as the chairs' capacity, there are 80 board members in total. 4. \( \frac{2}{5} \) of the chairs were not occupied, so \( \frac{2}{5} \times 40 = 16 \) chairs were empty. 5. Therefore, \( 40 - 16 = 24 \) chairs were occupied. 6. Each of these 24 chairs was fully occupied by 2 people, meaning \( 24 \times 2 = 48 \) people attended the meeting.  Let's confirm this with a Python calculation. 
<tool_call> {"name": "execute_python_code", "arguments": {"code": "total_chairs = 40\nchair_capacity = 2\nunoccupied_fraction = 2/5\nunoccupied_chairs = total_chairs * unoccupied_fraction\noccupied_chairs = total_chairs - unoccupied_chairs\nattended_board_members = occupied_chairs * chair_capacity\nattended_board_members"}} </tool_call>
 <tool_call> {"name": "generate_response", "arguments": {"response": "The number of board members who did attend the meeting is \(\boxed{48}\)."}} 
</tool_call>
<|im_end|>
```

However, tuning resolves these issues, as shown in the example below:


<img width="1367" height="684" alt="image" src="https://github.com/user-attachments/assets/abf83861-d87c-4376-a4ec-04b2fde2be0a" />
<img width="1363" height="422" alt="image" src="https://github.com/user-attachments/assets/e7a8a199-4577-468c-8ad4-5664bc7a2db1" />



