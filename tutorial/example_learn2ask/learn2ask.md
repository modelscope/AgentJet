# Learning to Ask

Traditional LLMs primarily function by generating a direct answer or completing text based on a given prompt or question, whereas the core of the **Learning to Ask** task is to train an agent to learn how to ask questions that elicit the most valuable information and best advance the task.

![](https://img.alicdn.com/imgextra/i4/O1CN01m9WJCM1WJL1aJCSaS_!!6000000002767-2-tps-1024-559.png)

This document demonstrates how to prepare data, build an agent and workflow, set up rewards, and ultimately train a 7B agent for this task.

## 1. Prepare Dataset
Download [RealMedConv](https://huggingface.co/datasets/datajuicer/RealMedConv) dataset from HuggingFace, and put the files in `data/realmedconv`.

Run the following command to preprocess the dataset:

```bash
export DASHSCOPE_API_KEY=your_api_key

python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/train_original.jsonl --output_file data/realmedconv/train_processed.jsonl
python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/train_processed.jsonl --output_file data/realmedconv/train.jsonl

python tutorial/example_learn2ask/data_preprocess/step1.py --input_file data/realmedconv/test_original.jsonl --output_file data/realmedconv/test_processed.jsonl
python tutorial/example_learn2ask/data_preprocess/step2.py --input_file data/realmedconv/test_processed.jsonl --output_file data/realmedconv/test.jsonl
```

You now will get two datasets:
- `train.jsonl`: the train split
- `test.jsonl`: the test split

Next, we will prepare a workflow to train an agent with these data.

## 2. Prepare Workflow
Refer to `tutorial/example_learn2ask/learn2ask.py` for details of workflow.

In this workflow, we implement:
- `ExampleLearn2Ask`: the workflow and agent
- `reward_fn`: the reward of learn2ask task
- `llm_reward`: a llm-as-a-judge used in `reward_fn`

The agent reads the messages between a doctor and a patient, and tries predicting the next question to collect information and diagnose the patient. The question is judged by a LLM to check if it is valuable and relevant.

## 3. Prepare Training Configuration
Next, we need a YAML configuration file to tell ASTuner what data, workflow, reward to use.

In `tutorial/example_learn2ask/learn2ask.yaml`, we set them respectively:
- Set experiment name.
- Use `dataset_file` reader to read the training and validation data.
- Set the pretrained model.
- Use `ExampleLearn2Ask` as the workflow.
- Set other training parameters.

You can use this exisiting configuration file straigtly and fill the dataset path in `tutorial/example_learn2ask/learn2ask.yaml`.

## 3. Start Training

Start the training with

```bash
astuner --conf tutorial/example_learn2ask/learn2ask.yaml --backbone='trinity' --with-ray
```

## 4. Results

### Main Result

![curve](https://img.alicdn.com/imgextra/i2/O1CN01STeE8G1ricp6jySto_!!6000000005665-2-tps-866-614.png)

## Analysis

### Sample 1


**Before training.**
```
User: Bitten by a midge

Agent: Has irritation subsided past initial acute swelling phase afterwards unnoticed scratched overly causing secondary bacterial dermatitis chance affecting skin rendering needed topical antibiotic application options among azelaic acid cream!?
```

The agent provided a slightly related but completely unreasonable answer and failed to form any effective question to the user.

**During Training, ~50% step.**
```
User: Bitten by a midge

Agent: Have these itchy bumps appeared soon after getting bitten?(A.) Occasionally immediately afterward ; B fkknore ×later throughout day )/)
```

The agent provided a relatively effective question, attempting to gather the user's general symptoms and offering two options.

**After training.**
```
User: Bitten by a midge

Agent: Has itching or reddening appeared around this bite site recently without healing despite scratching frequently?(A,Yes indeed worse every evening B,No pain but small swollen spots persist day after incident/)
```

The agent's question is more precise and informative, providing two specific and effective options. This is effective in guiding the user to recall symptom details.

## Next

To learn more about the task and results on larger models, refer to [Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs](https://arxiv.org/abs/2510.25441).