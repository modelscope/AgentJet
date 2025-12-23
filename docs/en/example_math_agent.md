# Math Agent

Train a **tool-using Math Agent** (ReAct + Python executor) to solve GSM8K-style math problems.  
Rewards come from a **judge** that checks final-answer correctness (and can optionally penalize bad tool-call behaviors).

---

### 1. Overview

In **Math Agent**, each training sample is a math word problem (e.g., GSM8K). The agent learns to:

- **reason step by step** (ReAct-style),
- **call a Python tool** when computation is needed,
- produce a final answer that matches the reference.

This tutorial is organized in two steps:

1) **Run it**: download the dataset and start training with the default YAML config.  
2) **Understand & customize**: read the workflow (`ExampleMathLearn`) and the judge/reward (`MathAnswerAndLlmAsJudge`).

---

### 2. Quick Start

#### 2.1 Prepare Dataset

Download the `openai/gsm8k` dataset:

```bash
python scripts/download_dataset.py --target=openai/gsm8k --path=/the/path/to/store/dataset
```

#### 2.2 Start Training

```bash
# (optional) recommended cleanup before training
# astuner --kill="python|ray|vllm"

astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='trinity' --with-ray
```

#### 2.3 Debug Locally (No Ray)

If you want to breakpoint-debug the workflow/judge locally:

```bash
# (optional) recommended cleanup before debug
# astuner --kill="python|ray"

clear && \
astuner --conf tutorial/example_math_agent/math_agent.yaml --backbone='debug' --with-logview
```

When `--backbone=debug`, Ray is disabled. You can use a VSCode `launch.json` like below:

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
        "--conf", "xxxx/xxxx/xxxx.yaml"
      ],
      "env": {}
    }
  ]
}
```

---

### 3. Understand

#### 3.1 What happens each step

Each training step does:

1. **Load one problem** from the dataset (`task_reader`).
2. Run the **AgentScope workflow**:

   * Build the prompt from the problem text,
   * Let the ReAct agent optionally call a Python tool for computation,
   * Extract the **final answer**.
3. Register key info for evaluation (important!):

   * The workflow must call:

     * `astune_proxy.update_judge_input_dictionary(final_answer=final_answer)`
4. Run the **judge** to compute reward:

   * compare `final_answer` with the reference answer from the task,
   * output `raw_reward` and `is_success`,
   * the trainer uses them to update the policy.

#### 3.2 YAML Configuration

Most wiring happens in `tutorial/example_math_agent/math_agent.yaml`. The key fields are:

* `astune.task_reader`: where tasks come from
* `astune.rollout.agentscope_workflow`: which workflow runs per sample
* `astune.task_judge.judge_protocol`: which judge computes rewards
* `astune.model.path`: pretrained model you fine-tune

Minimal example:

```yaml
astune:
  task_reader:
    type: huggingface_dat_repo   # also supports: dataset_file / env_service (if enabled)

  rollout:
    agentscope_workflow: tutorial.math_agent->ExampleMathLearn

  task_judge:
    judge_protocol: astune.task_judge.math_answer_as_judge->MathAnswerAndLlmAsJudge

  model:
    path: /mnt/data/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
```

#### 3.3 Code Walkthrough

**Workflow (AgentScope):** `tutorial/example_math_agent/math_agent.py`

The workflow typically:

* registers tools (e.g., `execute_python_code`)
* constructs a ReAct agent
* runs one turn from the user problem
* parses the final answer
* exposes it to the judge via `update_judge_input_dictionary`

Workflow sketch:

```python
self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)

self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=astune_proxy,  # trainer-managed model wrapper
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)

msg = Msg("user", init_messages[0]["content"], role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)

# IMPORTANT: provide final answer to the judge
astune_proxy.update_judge_input_dictionary(final_answer=final_answer)
```

**Judge / Reward:** `astune/task_judge/math_answer_as_judge.py`

Two simple judges are provided there; you can add your own judge anywhere in the project.

#### 3.4 Reward

The judge reads from `judge_input_dictionary`, commonly including:

* `env`: env_service external environment (if enabled)
* `workflow_task`: task info; reference answer can be retrieved from here
* `grouped_steps`: all LLM conversation turns (useful if you want process-based scoring)
* `final_answer`: not present by default — you must set it in the workflow via:

  * `astune_proxy.update_judge_input_dictionary(final_answer=final_answer)`

The judge returns:

* `raw_reward`
* `is_success`

**Practical tip:**
If you observe the model “almost solved it but messed up tool-call formatting / impatiently skipped tool execution”, you can extend the judge to:

* add a format penalty (invalid `<tool_call>`)
* add a behavior penalty (tool called but no `print` / execution result not used)
* keep answer correctness as the primary signal

---

### 4. Results

#### 4.1 Training Curve

![Tracing curve](https://img.alicdn.com/imgextra/i4/O1CN01gzwgLq1fkCnauydEu_!!6000000004044-2-tps-1422-550.png)

Interpretation: as training progresses, reward increases. This usually means the agent becomes more stable on **two things**:

* **Using tools when it should**: it can correctly emit a `<tool_call>` and call `execute_python_code` for computation.
* **Producing more reliable answers**: it can use the tool return (e.g., `<tool_response>`) to output a final answer aligned with the reference.

> In practice, the gain here is often less about “stronger math ability” and more about “better tool discipline + more consistent use of execution results”.

---

#### 4.2 Case Study: from “can solve” to “can solve with tools”

Before training, the agent may already solve many problems. However, smaller models often fail at **tool-call discipline**, e.g.:

* forgetting to `print` the computed value in Python (the tool ran, but produced no usable output),
* outputting the final answer before the tool execution finishes (premature answering),
* malformed `<tool_call>` blocks (tool not triggered / parsing fails).

##### Bad cases: what typical failures look like

```text
# bad case 1: forgot to print the result in python code
<tool_call>
{"name": "execute_python_code", "arguments": {"code": "... height_difference"}}
</tool_call>

# bad case 2: too impatient — outputs final answer without waiting for the tool result
<tool_call> {"name": "execute_python_code", ...} </tool_call>
<tool_call> {"name": "generate_response", "arguments": {"response": "... \\boxed{48} ..."}} </tool_call>
```

These failures are usually not because the model “can’t do math”, but because it **does not close the loop** by incorporating the tool execution result:

* bad case 1: the tool may succeed, but without `print`, `stdout` is empty and the model can’t reliably read the value.
* bad case 2: the model generates a tool call and a final answer back-to-back in the same turn, effectively **skipping the “wait for `<tool_response>`” step**.

---

##### Good case: after tuning, the tool-use loop becomes closed

After tuning, the agent often follows a clean 3-stage pattern (corresponding to Message 3/4/5 in the screenshots):

1. **Message 3 (assistant)**: decomposes the problem + emits a `<tool_call>`, and uses `print(...)` to output key values
2. **Message 4 (tool_response)**: the tool returns execution results (e.g., `returncode=0`, `stdout=...`)
3. **Message 5 (assistant)**: reads `stdout` and then produces the final answer (e.g., `\\boxed{18}`)

![image](https://img.alicdn.com/imgextra/i4/O1CN01v1gGQZ1ftMiil5Cxg_!!6000000004064-2-tps-1367-684.png)

![image](https://img.alicdn.com/imgextra/i4/O1CN01WarPpf1yNk4awZOIO_!!6000000006567-2-tps-1363-422.png)

On the right side of the figure, the colored blocks are a **token-level sequence visualization**:

* **each block is one token** (the number inside is the token id),
* the order of blocks is the order the model **consumed/generated** tokens,
* what matters is not the token id itself, but whether you can see clear boundary markers such as:

  * `<im_start> assistant ... <tool_call> ... <im_end>`
  * `<im_start> user <tool_response> ... <stdout>18.0</stdout> ... <im_end>`
  * `<im_start> assistant ... \\boxed{18} ... <im_end>`

A “good” tool-call behavior typically shows up in logs as:

* `<tool_call>` and `<tool_response>` appear in **separate turns** (call → response → answer),
* `<tool_response>` contains **non-empty stdout**,
* the final answer appears **after** the tool returns, rather than being produced prematurely.

