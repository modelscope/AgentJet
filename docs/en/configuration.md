# Configuration Reference

AgentJet uses YAML files to configure every aspect of a training run.
This page is a **lookup reference** for every configuration key.
For conceptual introductions, see [Workflow](../workflow/), [Data Pipeline](../data_pipeline/), and [Task Judger](../task_judger/).

<br/>

## How Configuration Works

AgentJet uses [Hydra](https://hydra.cc/) to compose a final config. Your experiment YAML only needs to contain the keys you want to override — everything else is filled in by defaults.


### Classic Mode Config Chain

There are three layers, merged top-to-bottom (later layers win):

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                          (lowest prior)                               │
 │            verl_default.yaml (auto generated, do not edit)            │
 │                                                                       │
 │  Backend defaults (actor_rollout_ref.*, algorithm.*, trainer.*, ...)  │
 │  You almost never touch these directly.                               │
 │                                                                       │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 │  overridden by
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                        ajet_default.yaml                              │
 │                                                                       │
 │  AgentJet defaults for all ajet.* keys.                               │
 │  (ajet.* keys are auto-converted to verl keys — see "Backend-        │
 │   Specific: verl" section below for the full mapping)                  │
 │                                                                       │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 │  overridden by
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                       (highest priority)                              │
 │                      your_experiment.yaml                             │
 │                                                                       │
 │  YOUR overrides — only write the keys you want to change.             │
 │                                                                       │
 └───────────────────────────────────────────────────────────────────────┘

 Priority:  your_experiment.yaml  >  ajet_default.yaml  >  verl_default.yaml
            (highest)                                       (lowest)
```



### Swarm Mode Config Chain


```
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                        (lowest prior)                                 │
 │            verl_default.yaml (auto generated, do not edit)            │
 │                                                                       │
 │  Backend defaults (actor_rollout_ref.*, algorithm.*, trainer.*, ...)  │
 │                                                                       │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 │  overridden by
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                        ajet_default.yaml                              │
 │  AgentJet defaults for all ajet.* keys.                               │
 │                                                                       │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 │  overridden by
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                     ajet_swarm_default.yaml                           │
 │  Swarm-specific overrides on top of ajet_default:                     │
 │   - enable_swarm_mode: true                                           │
 │   - enable_interchange_server: true                                   │
 │   - interchange_server.interchange_server_port: 10086                 │
 │   - task_reader.type: random_dummy  (tasks come from remote workers)  │
 │   - task_judge.judge_protocol: null  (rewards come from remote too)   │
 │   - trainer_common.logger: tensorboard                                │
 │                                                                       │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 │  overridden by
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                                                                       │
 │                   AgentJetJob Argument (highest priority)             │
 │                                                                       │
 └───────────────────────────────────────────────────────────────────────┘


 Priority:  AgentJetJob Argument  >  ajet_swarm_default  >  ajet_default  >  verl_default
            (highest)                                                         (lowest)
```



<br/>
<br/>
<br/>


## Full Key Reference

All keys live under the `ajet:` root.


### `ajet.project_name`

- **Type:** str.
- **Default:** `"ajet_default_project"`.
- **Description:** Project name for logging and experiment tracking. Passed to the logging backend (SwanLab, WandB, TensorBoard, etc.) to group related experiment runs under one project. All runs sharing the same `project_name` appear together in the logger's dashboard.

### `ajet.experiment_name`

- **Type:** str.
- **Default:** `"read_yaml_name"`.
- **Description:** Experiment name used for logging, directory naming, and checkpoint organization. When set to the special value `"read_yaml_name"`, AgentJet automatically derives the name from the YAML config filename and appends a timestamp: `{yaml_basename}_{YYYYMMDD_HHMM}` (e.g., `train_qwen_20260401_1430` for a file named `train_qwen.yaml`). Any pipe characters (`|`) in the name are converted to hyphens (`-`). You can also set an explicit string to use a fixed experiment name.

### `ajet.experiment_dir`

- **Type:** str.
- **Default:** `"auto"`.
- **Description:** Root output directory for this experiment. When set to `"auto"`, resolves to `{exp_base_dir}/{experiment_name}`, where `exp_base_dir` comes from the `--exp-dir` CLI flag (default `saved_experiments`). For example, with defaults and a config file named `train_qwen.yaml`, the resolved path would be `saved_experiments/train_qwen_20260401_1430/`. This directory stores experiment YAML backups, training artifacts, and log outputs. You can also set an absolute path to use a fixed directory.

### `ajet.backbone`

- **Type:** str.
- **Default:** `"verl"`.
- **Description:** Training backend that controls the entire training pipeline. Valid values:
    - `"verl"` — Full distributed training using [VeRL](https://github.com/volcengine/verl). Supports FSDP (Fully Sharded Data Parallel), multi-node training, and all production features. Recommended for real training runs.
    - `"trinity"` — Alternative distributed backend. Includes special batch size validation (auto-adjusts `train_batch_size` to be divisible by `fsdp_world_size`) and uses different config mappings. Supports multiple vLLM engine instances via `n_vllm_engine`.
    - `"debug"` — Lightweight single-machine mode that connects to an external OpenAI-compatible vLLM server instead of launching a full training loop. Only runs inference and rollout—no weight updates. Sets `AJET_DEBUG=1` environment variable. Ideal for fast iteration on workflows and reward functions.


## `ajet.model`

### `ajet.model.path`

- **Type:** str.
- **Default:** _(required)_.
- **Description:** Model to train. Accepts two formats:
    - **Local filesystem path** (detected when the path contains 3+ slashes): e.g. `/data/models/Qwen2.5-7B-Instruct`. AgentJet validates that the path exists at launch and raises an error if not found.
    - **HuggingFace Hub ID** (fewer than 3 slashes): e.g. `Qwen/Qwen2.5-7B-Instruct`. Automatically downloaded by the transformers library and cached locally before training. VeRL may copy the model to shared memory (`use_shm=True`) or local cache via `verl.utils.fs.copy_to_local()` for faster loading.


## `ajet.data`

Controls tokenization limits and batch sizing.

Effective sample count per update = `train_batch_size` x `rollout.num_repeat` x `rollout.multi_turn.expected_steps`.

### `ajet.data.max_prompt_length`

- **Type:** int.
- **Default:** `3000`.
- **Description:** Maximum number of tokens allowed for the prompt (system message + user query + conversation history before the model's response). This is a **hard constraint**—if any sample's `prompt_ids` exceeds this limit, AgentJet raises a `RuntimeError` and training stops immediately. It is not a soft truncation. Increase this value if your tasks have long system prompts or multi-turn conversation histories.

### `ajet.data.max_response_length`

- **Type:** int.
- **Default:** `15000`.
- **Description:** Maximum number of tokens allowed for the full response across all turns in an episode. This is a **hard constraint**—if any sample's `response_ids` exceeds this limit, AgentJet raises a `RuntimeError` and training stops immediately. This is the cumulative response budget for the entire episode (all assistant turns combined), not a per-turn limit. For per-turn limits, see `rollout.max_response_length_in_one_turn`.

### `ajet.data.train_batch_size`

- **Type:** int.
- **Default:** `32`.
- **Description:** Number of unique tasks per training batch. The effective number of samples per gradient update is `train_batch_size × rollout.num_repeat × rollout.multi_turn.expected_steps`. For the **verl** backend, `train_batch_size × num_repeat` must be divisible by the total number of GPUs (`nnodes × n_gpus_per_node`). For the **trinity** backend, `train_batch_size` is automatically rounded up to be divisible by `fsdp_world_size` if needed.


## `ajet.rollout`

Controls how the agent interacts during rollout (inference).

### `ajet.rollout.user_workflow`

- **Type:** str.
- **Default:** _(required)_.
- **Description:** Import path to your Workflow class, loaded at runtime via `dynamic_import()` with a thread-safe lock. Two syntaxes are supported:
    - **Module syntax:** `module.path->ClassName` (e.g. `tutorial.example_math_agent.math_agent->MathAgentWorkflow`). Uses `importlib.import_module()`.
    - **File syntax:** `path/to/file.py->ClassName`. Uses `importlib.util.spec_from_file_location()`.

    The class is instantiated per task execution in `GeneralRunner.execute()`.

### `ajet.rollout.force_disable_toolcalls`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Globally disable all tool calling during rollout. When `true`, the LLM bridge skips tool call extraction and parsing (even if `<tool_call>` tags appear in the output), and the context tracker removes tool-related messages when spawning timelines. Useful for pure text-generation tasks where the model should never invoke tools.

### `ajet.rollout.max_env_worker`

- **Type:** int.
- **Default:** `64`.
- **Description:** Maximum number of parallel environment/rollout worker threads. Controls the `ThreadPoolExecutor(max_workers=...)` size used during rollout. Each worker runs one agent episode concurrently. The actual number of parallel tasks is `min(available_tasks, ceil(max_env_worker / num_repeat))`, since each task spawns `num_repeat` threads. In swarm mode, also recommended to match `interchange_server.max_inference_tracker_threads`.

### `ajet.rollout.num_repeat`

- **Type:** int.
- **Default:** `4`.
- **Description:** Number of independent rollout episodes per task in each training batch (GRPO group size). Each task is executed `num_repeat` times to collect multiple trajectories, and advantage is estimated by comparing rewards across the group. Higher values give more stable advantage estimates but increase compute proportionally. Directly affects the effective batch size: `real_train_batch_size = train_batch_size × num_repeat`.

### `ajet.rollout.temperature`

- **Type:** float.
- **Default:** `0.9`.
- **Description:** Sampling temperature during **training** rollouts. Passed directly to the inference engine (vLLM/SGLang) as `temperature` in the sampling parameters. Higher values produce more diverse outputs for better exploration. During validation, this is overridden by `rollout.val_kwargs.temperature` (default `0.0` for greedy decoding).

### `ajet.rollout.top_p`

- **Type:** float.
- **Default:** `1.0`.
- **Description:** Nucleus (top-p) sampling threshold during training rollouts. Only tokens with cumulative probability ≤ `top_p` are considered. `1.0` disables nucleus sampling. During validation, overridden by `rollout.val_kwargs.top_p`.

### `ajet.rollout.top_k`

- **Type:** int.
- **Default:** `-1`.
- **Description:** Top-k sampling during training rollouts. Limits sampling to the top-k most probable tokens. `-1` disables top-k filtering. During validation, overridden by `rollout.val_kwargs.top_k`.

### `ajet.rollout.gamma`

- **Type:** float.
- **Default:** `1.0`.
- **Description:** Step reward discount factor for multi-step episodes. Applied as `step_reward = global_reward × gamma^(total_steps - step_index - 1)`, so earlier steps receive more discounted rewards when `gamma < 1.0`. **Currently only `gamma = 1.0` is supported** (no discounting); multi-step reward discounting is planned for a future release. Setting `gamma != 1.0` raises an assertion error.

### `ajet.rollout.max_response_length_in_one_turn`

- **Type:** int.
- **Default:** `4096`.
- **Description:** Maximum number of tokens the model can produce in a single LLM call (one assistant turn). Enforced at multiple levels: (1) passed as `max_tokens` to the vLLM engine to hard-stop generation, (2) validated via assertion after generation, and (3) used to compute the effective context budget: if `prompt_length ≥ max_model_len - max_response_length_in_one_turn`, the turn is terminated early with `finish_reason="length"`. This is distinct from `data.max_response_length`, which limits the cumulative response across all turns.

### `ajet.rollout.max_model_len`

- **Type:** int.
- **Default:** `18000`.
- **Description:** Maximum total context window length in tokens (prompt + all accumulated responses across turns). Auto-mapped to four verl keys simultaneously: `actor_rollout_ref.rollout.max_model_len`, `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`, `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`, and `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`. This ensures consistent memory allocation across the rollout engine, actor training, and reference model. Also used to compute the per-turn context budget: `max_seq_length = max_model_len - max_response_length_in_one_turn`.

### `ajet.rollout.tensor_model_parallel_size`

- **Type:** int.
- **Default:** `1`.
- **Description:** Tensor-parallel size for the vLLM rollout engine. Splits model layers across this many GPUs for inference. When combined with `n_vllm_engine`, total inference GPU allocation = `n_vllm_engine × tensor_model_parallel_size`. Increase for models too large to fit on a single GPU. In debug mode, use `debug.debug_tensor_parallel_size` instead.

### `ajet.rollout.n_vllm_engine`

- **Type:** int.
- **Default:** `1`.
- **Description:** Number of independent vLLM engine instances for inference. **Only effective with the `trinity` backend**—ignored by the `verl` and `debug` backends. Multiple engines allow horizontal scaling of inference throughput. Total inference GPU allocation = `n_vllm_engine × tensor_model_parallel_size`. Also affects worker distribution: each engine handles up to `max_env_worker / n_vllm_engine` concurrent requests.

### `ajet.rollout.max_num_seqs`

- **Type:** int.
- **Default:** `10`.
- **Description:** Maximum number of sequences batched in parallel per vLLM engine instance. Passed to vLLM as `--max-num-seqs`. Higher values improve throughput but increase GPU memory usage. Lower values reduce memory but may cause underutilization. Tune based on available GPU memory and `max_model_len`.

### `ajet.rollout.name`

- **Type:** str.
- **Default:** `"vllm"`.
- **Description:** Inference engine backend. Options:
    - `"vllm"` — Uses vLLM. Sampling params use `max_tokens`, `logprobs=1`, and `min_tokens=1`.
    - `"sglang"` — Uses SGLang. Sampling params use `max_new_tokens` instead, and omit logprobs/min_tokens.

    The choice affects how sampling parameters are constructed and passed to the engine.

### `ajet.rollout.agent_madness_termination`

- **Type:** bool.
- **Default:** `true`.
- **Description:** Terminate agent episodes that are detected as producing degenerate output ("madness"). When `true`, the context tracker stops recording the timeline immediately upon detection, ending the episode early. The detection is performed via `compute_string_madness()` using the checks specified in `compute_madness_checklist`. This prevents wasting compute on clearly degenerate rollouts.

### `ajet.rollout.agent_madness_reward`

- **Type:** float.
- **Default:** `-1.0`.
- **Description:** Reward assigned to episodes flagged as "mad" (degenerate). When madness is detected, the step reward is overridden to this value and `reward_structure.madness` is set to `-1.0`. A negative reward teaches the model to avoid degenerate behaviors (repetition, leaked special tokens, malformed tool calls).

### `ajet.rollout.compute_madness_checklist`

- **Type:** list.
- **Default:** `["nonsense"]`.
- **Description:** List of madness detection checks to enable. Options:
    - `"nonsense"` — Detects degenerate text patterns: leaked special tokens (`<|im_start|>`), repeated word sequences (5-word window with patience=10), character-level repetition (4-char window with patience=200).
    - `"wrong_toolcall"` — Validates tool call JSON structure: checks for valid `function` and `arguments` fields, verifies arguments parse as a JSON dict, detects `<tool_call>` tags in content without successfully parsed tool calls.
    - `"non_ascii"` — Flags non-ASCII characters in output. When combined with `"nonsense"`, non-ASCII detection is included in the nonsense check. When only `"nonsense"` is specified (default), non-ASCII detection is skipped.


## `ajet.rollout.multi_turn`

### `ajet.rollout.multi_turn.max_sample_per_task`

- **Type:** int.
- **Default:** `30`.
- **Description:** Maximum number of training sample groups to keep from a single episode trajectory. In multi-turn episodes, each turn produces a sample group. If the trajectory produces more groups than this limit, a random subset of `max_sample_per_task` groups is selected for training. This controls memory usage and prevents a single long episode from dominating the training batch.

### `ajet.rollout.multi_turn.max_steps`

- **Type:** int.
- **Default:** `30`.
- **Description:** Hard limit on the number of interaction turns (agent steps) per episode. When the agent reaches this many turns, the episode is forcefully terminated regardless of completion status. Each "step" is one assistant response + optional tool execution cycle. Set higher for complex tasks that require many reasoning steps.

### `ajet.rollout.multi_turn.expected_steps`

- **Type:** int.
- **Default:** `1`.
- **Description:** Expected average number of turns per episode, used for batch size planning. Affects the effective sample count formula: `N_sample = train_batch_size × num_repeat × expected_steps`. For single-turn tasks (e.g. math QA), keep at `1`. For multi-turn agentic tasks (e.g. tool-using agents), set to the expected average number of interaction rounds. This value does not limit or enforce step counts—it only affects batch size calculation and resource allocation.


## `ajet.rollout.val_kwargs`

Sampling parameters used during **validation** only.

### `ajet.rollout.val_kwargs.temperature`

- **Type:** float.
- **Default:** `0.0`.
- **Description:** Sampling temperature used **only during validation**. Overrides `rollout.temperature` when running validation episodes. Default `0.0` enables greedy (deterministic) decoding for reproducible evaluation results.

### `ajet.rollout.val_kwargs.top_k`

- **Type:** int.
- **Default:** `-1`.
- **Description:** Top-k sampling used during validation. Overrides `rollout.top_k`. `-1` disables top-k filtering.

### `ajet.rollout.val_kwargs.top_p`

- **Type:** float.
- **Default:** `1.0`.
- **Description:** Nucleus sampling threshold used during validation. Overrides `rollout.top_p`. `1.0` disables nucleus sampling.

### `ajet.rollout.val_kwargs.do_sample`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Whether to use sampling during validation. When `false` (default), combined with `temperature: 0.0`, the model uses fully deterministic greedy decoding for consistent evaluation metrics.


## `ajet.task_reader`

Selects and configures the data source. See [Data Pipeline](../data_pipeline/) for detailed usage.

### `ajet.task_reader.type`

- **Type:** str.
- **Default:** `"huggingface_dat_repo"`.
- **Description:** Selects the data source implementation via `RouterTaskReader`. Valid values:
    - `"huggingface_dat_repo"` — Loads datasets from HuggingFace Hub or local files (including `.parquet`). See `task_reader.huggingface_dat_repo.*`.
    - `"jsonl_dataset_file"` — Reads tasks from local JSONL files. See `task_reader.jsonl_dataset_file.*`.
    - `"env_service"` — Fetches task IDs from an external environment service via HTTP. See `task_reader.env_service.*`.
    - `"data_generation"` — Generates training tasks from source documents using an LLM. See `task_reader.data_generation.*`.
    - `"tracing"` — Reads tasks from tracing data sources (with optional filtering and deduplication).
    - `"random_dummy"` — Generates random dummy tasks for testing and debugging.


## `ajet.task_reader.huggingface_dat_repo`

Active when `task_reader.type: huggingface_dat_repo`.

### `ajet.task_reader.huggingface_dat_repo.dataset_path`

- **Type:** str.
- **Default:** `"gsm8k"`.
- **Description:** HuggingFace dataset name, local directory path, or local `.parquet` file path. When a `.parquet` file is provided, it is loaded via `datasets.load_dataset("parquet", data_files=...)`. For HuggingFace datasets, uses `datasets.load_dataset()` which auto-downloads and caches. Data is automatically shuffled after loading. Each example is converted to a Task with `main_query` (falls back to `question` field), `task_id`, and the raw example stored as `metadata`.

### `ajet.task_reader.huggingface_dat_repo.dataset_name`

- **Type:** str.
- **Default:** `null`.
- **Description:** Dataset configuration/subset name passed to `datasets.load_dataset()` as the `name` parameter. For example, `"main"` for GSM8K, `"default"` for some datasets. Set to `null` if the dataset has no named configurations.

### `ajet.task_reader.huggingface_dat_repo.training_split`

- **Type:** str.
- **Default:** `"train"`.
- **Description:** Split name for training data, passed to `datasets.load_dataset()` as `split`. Common values: `"train"`, `"train[:1000]"` (slicing syntax supported by HuggingFace datasets).

### `ajet.task_reader.huggingface_dat_repo.validation_split`

- **Type:** str.
- **Default:** `"validation"`.
- **Description:** Split name for validation data. Common values: `"validation"`, `"test"`, `"dev"`.

### `ajet.task_reader.huggingface_dat_repo.http_proxy_address`

- **Type:** str.
- **Default:** `""`.
- **Description:** HTTP proxy URL for downloading datasets from HuggingFace Hub. When set, configures `huggingface_hub` to use a custom `httpx.Client` with the specified proxy. Useful in air-gapped or firewalled environments. Example: `"http://proxy.corp:8080"`.


## `ajet.task_reader.jsonl_dataset_file`

Active when `task_reader.type: jsonl_dataset_file`.

Each JSONL line must be valid JSON with these fields:

- `main_query` (str, **required**) — The task query or question.
- `task_id` (str, optional, default `""`) — Unique task identifier.
- `init_messages` (list, optional, default `[]`) — Initial conversation messages to prepend.
- `env_type` (str, optional, default `"no_env"`) — Environment type for the task.
- `metadata` (dict, optional) — Arbitrary metadata. If omitted, the entire JSON object is used as metadata (allowing you to include custom fields like `answer`, `difficulty`, etc. that your judge can access).

### `ajet.task_reader.jsonl_dataset_file.training.file_path`

- **Type:** str.
- **Default:** _(required)_.
- **Description:** Path to the training JSONL file. Read line-by-line; empty lines are skipped. Raises `JSONDecodeError` on invalid JSON.

### `ajet.task_reader.jsonl_dataset_file.validation.file_path`

- **Type:** str.
- **Default:** _(required)_.
- **Description:** Path to the validation JSONL file. Same format as training.


## `ajet.task_reader.env_service`

Active when `task_reader.type: env_service`.

### `ajet.task_reader.env_service.env_type`

- **Type:** str.
- **Default:** `"appworld"`.
- **Description:** Environment type identifier passed to the environment service when fetching task profiles. Determines which set of tasks and environment logic the service returns. Supported values depend on your environment service deployment (e.g. `"appworld"`, `"webshop"`, `"bfcl"`).

### `ajet.task_reader.env_service.env_url`

- **Type:** str.
- **Default:** `"http://127.0.0.1:8080"`.
- **Description:** Base URL of the external environment service. AgentJet creates an `EnvClient` pointed at this URL and calls `get_env_profile(env_type, split=...)` to fetch available task IDs. The service must implement the AgentJet environment protocol.

### `ajet.task_reader.env_service.env_action_preference`

- **Type:** str.
- **Default:** `"code"`.
- **Description:** Preferred action format for agent-environment interaction. Options:
    - `"code"` — Agent sends executable code as actions.
    - `"text"` — Agent sends natural language text commands.
    - `"box"` — Agent interacts via structured box/form inputs.

    Used by the context tracker and rollout system to format agent actions appropriately for the environment.

### `ajet.task_reader.env_service.training_split`

- **Type:** str.
- **Default:** `"train"`.
- **Description:** Split identifier for training tasks, passed to `get_env_profile(env_type, split=...)`. The environment service returns task IDs for this split.

### `ajet.task_reader.env_service.validation_split`

- **Type:** str.
- **Default:** `"dev"`.
- **Description:** Split identifier for validation tasks. Passed to the environment service similarly to `training_split`.


## `ajet.task_reader.data_generation`

Active when `task_reader.type: data_generation`.

### `ajet.task_reader.data_generation.document_reader.document_path`

- **Type:** list.
- **Default:** _(required)_.
- **Description:** List of source document file paths to read for task generation. Supports PDF and other document formats handled by the `DocReader`. Documents are chunked and used as knowledge context for generating training tasks.

### `ajet.task_reader.data_generation.document_reader.languages`

- **Type:** list.
- **Default:** `["eng"]`.
- **Description:** Languages of the source documents, used by the document reader for parsing. Standard language codes (e.g. `"eng"`, `"zho"`).

### `ajet.task_reader.data_generation.document_reader.chunk_size`

- **Type:** int.
- **Default:** `5120`.
- **Description:** Maximum character count per text chunk when splitting documents. Documents are split into chunks of this size for processing. Also configurable: `split_by` (default `"sentence"`, options: `"sentence"`, `"paragraph"`, `"character"`) and `cache_enabled` (default `true`).

### `ajet.task_reader.data_generation.task_num`

- **Type:** int.
- **Default:** `10`.
- **Description:** Number of augmented training tasks to generate. The generation process has two phases: (1) document-based task generation (`ceil(task_num / 10)` rounds of `KnowledgeAugmentor`) producing validation tasks, and (2) augmented task generation (`task_num` calls to `TaskAugmentor`) producing training tasks. Generated tasks are cached to JSONL files keyed by config MD5 hash to avoid regeneration.

### `ajet.task_reader.data_generation.llm_model`

- **Type:** str.
- **Default:** `"qwen-long"`.
- **Description:** LLM model name used for generating tasks. Called via the DashScope API (requires `DASHSCOPE_API_KEY`). Used by both `KnowledgeAugmentor` and `TaskAugmentor` to create questions and tasks from source documents.

### `ajet.task_reader.data_generation.num_workers`

- **Type:** int.
- **Default:** `32`.
- **Description:** Number of parallel `ThreadPoolExecutor` workers for task generation. Controls concurrency of LLM API calls during the generation process. Higher values speed up generation but may hit API rate limits.


## `ajet.task_judge`

Configures reward computation. See [Task Judger](../task_judger/) for how to write custom judges.

### `ajet.task_judge.judge_type`

- **Type:** str.
- **Default:** `"customized_protocol"`.
- **Description:** Selects the reward computation strategy:
    - `"customized_protocol"` — Loads a user-defined judge class via `dynamic_import()`. You implement the reward logic in your own class. See `task_judge.judge_protocol`.
    - `"rubrics_auto_grader"` — Uses the RM Gallery auto-grader, which generates evaluation rubrics from reference data using an LLM, then scores agent outputs against those rubrics. Requires additional configuration under `task_judge.rubrics_auto_grader.*`.

### `ajet.task_judge.judge_protocol`

- **Type:** str.
- **Default:** _(required when `judge_type` is `customized_protocol`)_.
- **Description:** Import path to your custom judge class using `module.path->ClassName` syntax (e.g. `tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge`). The class must inherit from `BaseJudge` and implement `compute_reward(workflow_task, workflow_output) -> (raw_reward: float, is_success: bool)`. Loaded at runtime via `dynamic_import()` in `BaseAgentRunner`.

### `ajet.task_judge.alien_llm_model`

- **Type:** str.
- **Default:** `"qwen3-235b-a22b-instruct-2507"`.
- **Description:** External LLM model name used for LLM-as-Judge evaluation or auto-grading. Called via the DashScope OpenAI-compatible API at `https://dashscope.aliyuncs.com/compatible-mode/v1`. Requires `DASHSCOPE_API_KEY` environment variable. Used by both custom judges (via `create_external_llm_fn()`) and the rubrics auto-grader. Supports any model available through the DashScope API (e.g. `"qwen-max"`, `"qwen3-235b-a22b-instruct-2507"`).

### `ajet.task_judge.alien_llm_response_length`

- **Type:** int.
- **Default:** `512`.
- **Description:** Maximum number of completion tokens for the external helper LLM. Applied as `max_completion_tokens` in the DashScope API call. Limits the evaluation response length to control cost and latency. Increase if your judge needs longer reasoning chains (e.g. for complex rubric evaluation).


## `ajet.task_judge.rubrics_auto_grader`

Active when `task_judge.judge_type: rubrics_auto_grader`.

### `ajet.task_judge.rubrics_auto_grader.model_name`

- **Type:** str.
- **Default:** `"qwen-max"`.
- **Description:** LLM model for rubric generation and evaluation. Used by the `IterativeRubricsGenerator` to create evaluation rubrics via a Propose-Evaluate-Revise loop, and by the `LLMGrader` to score agent outputs. Called via DashScope API.

### `ajet.task_judge.rubrics_auto_grader.grader_mode`

- **Type:** str.
- **Default:** `"pointwise"`.
- **Description:** Evaluation mode for the auto-grader:
    - `"pointwise"` — Evaluates each output independently, producing a single score between `min_score` and `max_score`. Reference data should contain `answer` and `score` fields.
    - `"listwise"` — Evaluates multiple candidate outputs together, producing a ranking. Reference data should contain a `candidates` list with `answer` and `rank` fields.

### `ajet.task_judge.rubrics_auto_grader.language`

- **Type:** str.
- **Default:** `"en"`.
- **Description:** Language for rubric generation and evaluation prompts. Options: `"en"` (English), `"zh"` (Chinese). Affects the language of the generated rubrics and grading prompts.

### `ajet.task_judge.rubrics_auto_grader.min_score`

- **Type:** int.
- **Default:** `0`.
- **Description:** Minimum score for pointwise grading. Defines the lower bound of the scoring range. Only used when `grader_mode: "pointwise"`.

### `ajet.task_judge.rubrics_auto_grader.max_score`

- **Type:** int.
- **Default:** `1`.
- **Description:** Maximum score for pointwise grading. Defines the upper bound of the scoring range. Only used when `grader_mode: "pointwise"`. The score is normalized for reward computation.

### `ajet.task_judge.rubrics_auto_grader.query_field`

- **Type:** str.
- **Default:** `"main_query"`.
- **Description:** Field name to extract the query/question from the task object. Used when converting task data for rubric generation and evaluation.

### `ajet.task_judge.rubrics_auto_grader.answer_field`

- **Type:** str.
- **Default:** `"final_answer"`.
- **Description:** Field name to extract the agent's answer from workflow output metadata. The auto-grader reads `workflow_output.metadata[answer_field]` to get the text to evaluate.

### `ajet.task_judge.rubrics_auto_grader.reference_field`

- **Type:** str.
- **Default:** `"answer"`.
- **Description:** Field name to extract the reference (ground truth) answer from task metadata. Used during rubric generation to learn what correct answers look like. The auto-grader reads `task.metadata[reference_field]`.


## `ajet.context_tracker`

Controls the Context Tracker, which intercepts LLM calls, builds aligned timelines, and merges shared conversation prefixes (1.5x-10x training speedup).

### `ajet.context_tracker.timeline_merging_policy.timeline_compare_level`

- **Type:** str.
- **Default:** `"text"`.
- **Description:** Controls how timelines are compared when deciding whether to merge shared conversation prefixes:
    - `"text"` (relaxed) — Compares `content_for_compare` strings between timeline messages. More aggressive merging at very little cost, resulting in higher training speedup.
    - `"token"` (strict) — Compares exact `token_arr` sequences between timeline messages. Less aggressive merging since tokenization differences (e.g. whitespace handling) prevent matches. Use when tokenization fidelity is critical.

### `ajet.context_tracker.timeline_merging_policy.ignore_tools`

- **Type:** bool.
- **Default:** `true`.
- **Description:** When `true`, the timeline merging algorithm skips comparison of tool availability lists (`tools` field) between timelines. This allows more aggressive merging even when different episodes have different tool sets available. When `false`, timelines with different available tool lists are never merged, reducing the merge rate but ensuring strict tool-set consistency.

### `ajet.context_tracker.fix_retokenization_drift`

- **Type:** bool.
- **Default:** `true`.
- **Description:** Reconciles token array discrepancies between the inference engine's tokenization and the context tracker's tokenization. When enabled, `patch_prompt_tokens()` calls `ensure_retokenization_perfect_match()` to detect cases where prompt text matches but token IDs differ, and attempts recovery using `prompt_text` and `prompt_token_ids` from the engine output. These mismatches typically have minimal influence on training quality. Primarily a compatibility patch for the Trinity backend.

### `ajet.context_tracker.log_tool_format_check`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Enable logging of tool call format validation results. When `true` and `"wrong_toolcall"` is in `compute_madness_checklist`, logs whether each tool call passes or fails format validation—checking that `tool_calls[i]["function"]["arguments"]` contains valid JSON that parses to a dict.

### `ajet.context_tracker.log_tool_format_error_detail`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Enable detailed error logging for malformed tool calls. When `true` alongside `log_tool_format_check`, logs the specific error type (`"cannot parse arguments"`, `"arguments not json"`, `"no function or no arguments"`) and the full malformed tool call structure, wrapped with `---*({err_type})*---` delimiters for structured debugging.

### `ajet.context_tracker.detect_timeline_snap`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Detect non-linear timeline growth by checking whether consecutive saved timelines can be merged. A "snap" occurs when two adjacent timelines in the same episode fail the mergeability check, indicating a discontinuity in the conversation flow. When detected, logs a warning with `"merge failure discovered"`. **Adds meaningful computation overhead**: calls `is_timeline_mergeable()` on every pair of consecutive timelines, each involving message-by-message comparison and potentially tokenization. Only enable for debugging timeline issues.


## `ajet.debug`

Active only when `backbone: debug`. Connects to an external OpenAI-compatible endpoint instead of launching a full training loop.

### `ajet.debug.debug_max_parallel`

- **Type:** int.
- **Default:** `4`.
- **Description:** Maximum number of concurrent task rollout workers in debug mode. Passed to `VerlRolloutManager(max_parallel=...)` which spawns this many asyncio workers for simultaneous task execution. Keep low for step-through debugging; increase for faster iteration when testing workflows.

### `ajet.debug.debug_first_n_tasks`

- **Type:** int.
- **Default:** `2`.
- **Description:** Number of tasks to process from the dataset in debug mode. Uses simple slicing (`tasks[:n]`)—no randomization or weighted sampling. Deterministic task selection for reproducible debugging. Set to `1` for focused single-task debugging, increase for broader coverage.

### `ajet.debug.debug_vllm_port`

- **Type:** int.
- **Default:** `18000`.
- **Description:** HTTP port of the external OpenAI-compatible vLLM server. AgentJet creates a `ChatCompletionScheduler` pointing to `http://localhost:{port}/v1` and uses the OpenAI client to call `chat.completions.create()`. The vLLM server must be running before launching AgentJet in debug mode, or AgentJet can auto-launch it via `LaunchCommandWhenAbsent`.

### `ajet.debug.debug_vllm_seed`

- **Type:** int.
- **Default:** `12345`.
- **Description:** Random seed passed to the vLLM server as `--seed` when auto-launching. Controls randomness in token sampling during inference. Use a fixed seed for reproducible rollout results during debugging.

### `ajet.debug.debug_tensor_parallel_size`

- **Type:** int.
- **Default:** `4`.
- **Description:** Tensor-parallel size for the vLLM engine in debug mode. Passed as `--tensor-parallel-size` to the vLLM server. Automatically capped to the number of available GPUs if the configured value exceeds availability. This is separate from `rollout.tensor_model_parallel_size`, which is used by the verl/trinity backends during training.


## `ajet.trainer_common`

Training loop parameters shared across backends.

### `ajet.trainer_common.total_epochs`

- **Type:** int.
- **Default:** `50`.
- **Description:** Number of complete passes through the training dataset. The outer training loop iterates `total_epochs` times, processing all batches from the dataloader in each epoch.

### `ajet.trainer_common.save_freq`

- **Type:** int.
- **Default:** `20`.
- **Description:** Save a checkpoint every N training steps. Checkpoints are also saved on the last step and when an ESI (expiration) signal is detected. Uses VeRL's `FSDPCheckpointManager` to save model weights, optimizer state, and metadata. Checkpoints are organized as `{checkpoint_base_dir}/step_{N}/`. Set to `0` to disable periodic saving.

### `ajet.trainer_common.test_freq`

- **Type:** int.
- **Default:** `20`.
- **Description:** Run validation every N training steps. Validation is also triggered on the last training step. Skipped in swarm mode (swarm has its own validation mechanism). Set to `0` to disable periodic validation.

### `ajet.trainer_common.val_before_train`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Run a full validation pass before the first training step to establish baseline metrics. Results are logged to the configured logger at step 0 and optionally written to `val_print_to_markdown_file_path`. Useful for comparing pre-training vs. post-training performance.

### `ajet.trainer_common.val_pass_n`

- **Type:** int.
- **Default:** `4`.
- **Description:** Number of independent validation attempts per task for computing pass@n metrics. Each task is repeated `val_pass_n` times, and the following metrics are computed:
    - `task_pass_rate@n` — Fraction of tasks where at least 1 of n attempts succeeds.
    - `task_pass_rate@n-all-pass` — Fraction of tasks where all n attempts succeed.
    - Intermediate metrics at `pass_rate@2`, `pass_rate@4`, `pass_rate@8`, `pass_rate@16` (when applicable).

### `ajet.trainer_common.val_only`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Skip training and exit after validation. Must be used with `val_before_train: true` to trigger the validation pass. The trainer calls `_validate()` once and then returns immediately from `fit()`. Useful for evaluating a checkpoint without training.

### `ajet.trainer_common.val_print_to_markdown_file_path`

- **Type:** str.
- **Default:** `null`.
- **Description:** Path to a file where validation metrics are appended after each validation run. Written in append mode (`a+`), with parent directories created automatically. Each line contains a Python dict representation of the metrics (e.g. `{'pass_n': 4, 'total_tasks': 10, ...}`). Set to `null` to disable.

### `ajet.trainer_common.train_print_to_markdown_file_path`

- **Type:** str.
- **Default:** `null`.
- **Description:** Path to a file where training metrics are appended after every training step. Same format as `val_print_to_markdown_file_path`—each line is a dict with keys like `training/global_step`, `training/epoch`, `actor/pg_loss`, etc. Useful for offline analysis or monitoring.

### `ajet.trainer_common.nnodes`

- **Type:** int.
- **Default:** `1`.
- **Description:** Number of training nodes for distributed training. Total GPU count = `nnodes × n_gpus_per_node`. The effective batch size (`train_batch_size × num_repeat`) must be divisible by the total GPU count for the verl backend. Multi-node setups require `interchange_server.interchange_method: "tcp"` (IPC does not work across nodes).

### `ajet.trainer_common.n_gpus_per_node`

- **Type:** int.
- **Default:** `8`.
- **Description:** Number of GPUs per training node. Combined with `nnodes` to determine total GPU count for FSDP sharding, batch size validation, and resource allocation. Must match the actual GPU availability on each node.

### `ajet.trainer_common.logger`

- **Type:** str or list.
- **Default:** `"swanlab"`.
- **Description:** Logging backend for experiment tracking. Initialized via VeRL's `Tracking` class with the configured `project_name` and `experiment_name`. Options:
    - `"swanlab"` — SwanLab experiment tracker (default).
    - `"tensorboard"` — TensorBoard logging.
    - `"console"` — Print metrics to stdout.
    - `"wandb"` — Weights & Biases experiment tracker.

    Can also be a list for multiple loggers: `["console", "swanlab"]`.

### `ajet.trainer_common.save_trajectory_as_json_file`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Save detailed rollout trajectories to JSON files for analysis. When enabled, each episode's trajectory is saved with task_id, task_tag (success/failure/half_success), reward_structure, and the full conversation (roles, content, tool_calls, tool_results). Files are organized under `{BEST_LOGGER_PATH}/trajectory/{train|eval}/step_{N}/task_{id}.json`.

### `ajet.trainer_common.mini_batch_num`

- **Type:** int.
- **Default:** `1`.
- **Description:** Number of mini-batches to split each training batch into. When > 0, the batch is divided into `mini_batch_num` chunks, and `optimizer.step()` is called for each chunk. Loss is scaled by `response_mask.shape[0] / mini_batch_split_size` to maintain gradient magnitude. When ≤ 0, falls back to VeRL's default `ppo_mini_batch_size`. Increase to reduce peak GPU memory at the cost of more optimizer steps per batch.

### `ajet.trainer_common.loss_extra_scale_ratio`

- **Type:** float.
- **Default:** `1.0`.
- **Description:** Additional multiplicative scaling factor applied to the computed loss in `update_policy()`. The total loss scaling is: `loss_scale_factor = (base_scale) × loss_extra_scale_ratio`. Default `1.0` has no effect. Values > 1.0 amplify gradients; values < 1.0 dampen them. Useful for fine-tuning gradient magnitude when experimenting with different batch sizes or loss formulations.

### `ajet.trainer_common.checkpoint_base_dir`

- **Type:** str.
- **Default:** `"./saved_checkpoints"`.
- **Description:** Base directory for saving FSDP checkpoints (model weights, optimizer state, and metadata). Checkpoints are organized in step-numbered subdirectories: `{checkpoint_base_dir}/step_{N}/`. VeRL's `FSDPCheckpointManager` handles the actual save/load logic.


## `ajet.trainer_common.algorithm`

### `ajet.trainer_common.algorithm.adv_estimator`

- **Type:** str.
- **Default:** `"grpo"`.
- **Description:** Advantage estimation method used in `compute_advantage()`. Options include:
    - `"grpo"` — Group Relative Policy Optimization. Computes advantage by comparing rewards across the `num_repeat` rollouts for each task (group-level normalization).
    - `"gae"` — Generalized Advantage Estimation (standard PPO-style).
    - Other VeRL-supported estimators: `"reinforce++"`, `"a2c"`, etc.

### `ajet.trainer_common.algorithm.use_kl_in_reward`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Add KL divergence penalty to token-level rewards before advantage computation. When `true`, calls `apply_kl_penalty()` which adjusts `token_level_scores` by subtracting a KL penalty term (controlled by `kl_penalty` and `kl_ctrl`). When `false`, raw reward scores are used directly as `token_level_rewards`. This is separate from `use_kl_loss`, which adds KL divergence to the training loss instead.


## `ajet.trainer_common.optim`

### `ajet.trainer_common.optim.optimizer`

- **Type:** str.
- **Default:** `"AdamW"`.
- **Description:** Optimizer class name. Resolved from the `optimizer_impl` module. Common options: `"AdamW"`, `"Adam"`, `"SGD"`.

### `ajet.trainer_common.optim.optimizer_impl`

- **Type:** str.
- **Default:** `"torch.optim"`.
- **Description:** Python module containing the optimizer class. The optimizer is instantiated as `getattr(importlib.import_module(optimizer_impl), optimizer)(...)`. Use `"torch.optim"` for standard PyTorch optimizers.

### `ajet.trainer_common.optim.lr`

- **Type:** float.
- **Default:** `1e-6`.
- **Description:** Peak learning rate. Used as the base rate for the LR scheduler. Low default (`1e-6`) is typical for RLHF/GRPO fine-tuning to avoid catastrophic forgetting.

### `ajet.trainer_common.optim.weight_decay`

- **Type:** float.
- **Default:** `0.01`.
- **Description:** L2 regularization coefficient for AdamW. Applied to all parameters except biases and layer norms (standard practice).

### `ajet.trainer_common.optim.lr_scheduler_type`

- **Type:** str.
- **Default:** `"constant"`.
- **Description:** Learning rate schedule type. Options: `"constant"` (no decay), `"cosine"` (cosine annealing to `min_lr_ratio × lr`), and other schedules supported by VeRL. When using `"cosine"`, configure `min_lr_ratio` and optionally `num_cycles` (default `0.5`).

### `ajet.trainer_common.optim.lr_warmup_steps`

- **Type:** int.
- **Default:** `-1`.
- **Description:** Number of linear warmup steps from 0 to `lr`. Takes precedence over `lr_warmup_steps_ratio` when set to a positive value. `-1` means auto-compute from `lr_warmup_steps_ratio × total_training_steps`.

### `ajet.trainer_common.optim.lr_warmup_steps_ratio`

- **Type:** float.
- **Default:** `0.0`.
- **Description:** Warmup steps as a fraction of total training steps. Used when `lr_warmup_steps` is `-1`. For example, `0.1` warms up for the first 10% of training. `0.0` disables warmup.

### `ajet.trainer_common.optim.betas`

- **Type:** list.
- **Default:** `[0.9, 0.999]`.
- **Description:** Adam optimizer beta parameters: `[beta1, beta2]`. `beta1` controls the exponential decay rate for the first moment (momentum), `beta2` for the second moment (variance).

### `ajet.trainer_common.optim.min_lr_ratio`

- **Type:** float.
- **Default:** `0.0`.
- **Description:** Minimum learning rate as a fraction of the peak LR. Used with cosine and other decay schedules. The LR never drops below `min_lr_ratio × lr`. `0.0` allows decay to zero.

### `ajet.trainer_common.optim.grad_clip`

- **Type:** float.
- **Default:** `20.0`.
- **Description:** Maximum gradient norm for gradient clipping. Gradients are clipped to this norm during `optimizer.step()` to prevent training instability from gradient explosions. Lower values (e.g. `1.0`) provide more aggressive clipping.


## `ajet.trainer_common.fsdp_config`

### `ajet.trainer_common.fsdp_config.param_offload`

- **Type:** bool.
- **Default:** `true`.
- **Description:** Offload FSDP-sharded model parameters to CPU memory when not in use, loading them to GPU on demand during forward/backward passes. Significantly reduces GPU memory usage at the cost of CPU-GPU transfer overhead. Called via `offload_fsdp_model_to_cpu()` after each training step. Recommended for large models or limited GPU memory.

### `ajet.trainer_common.fsdp_config.optimizer_offload`

- **Type:** bool.
- **Default:** `true`.
- **Description:** Offload optimizer states (momentum, variance buffers) to CPU memory. Called via `offload_fsdp_optimizer()`. Combined with `param_offload`, can reduce GPU memory by 3-4x at the cost of slower training steps due to CPU-GPU data transfers. Both defaults are `true` for memory efficiency.


## `ajet.trainer_common` KL Loss

### `ajet.trainer_common.use_kl_loss`

- **Type:** bool.
- **Default:** `true`.
- **Description:** Add KL divergence regularization loss to the policy gradient objective. When `true`, computes `kl_penalty(logprob, ref_logprob, kl_penalty=kl_loss_type)` and adds it to the policy loss as: `total_loss = policy_loss + kl_loss × kl_loss_coef`. Requires a reference model to be available. This prevents the policy from diverging too far from the reference model during training.

### `ajet.trainer_common.kl_loss_coef`

- **Type:** float.
- **Default:** `0.002`.
- **Description:** Coefficient (weight) for the KL divergence loss term. Controls the strength of the regularization: higher values keep the policy closer to the reference model but may slow learning, lower values allow more policy divergence. Typical range: `0.001`–`0.01`.

### `ajet.trainer_common.kl_loss_type`

- **Type:** str.
- **Default:** `"low_var_kl"`.
- **Description:** KL divergence computation method. Options:
    - `"low_var_kl"` (default) — Low-variance KL estimator. Recommended for stable training as it reduces gradient variance compared to standard KL estimation.
    - Other VeRL-supported KL penalty types (e.g. standard KL, reverse KL).


## `ajet.trainer_common` Sequence Parallelism

### `ajet.trainer_common.ulysses_sequence_parallel_size`

- **Type:** int.
- **Default:** `1`.
- **Description:** Ulysses-style sequence parallelism degree. When > 1, sequences are split across this many GPUs using a 2D device mesh `(dp, sp)`, reducing per-GPU memory for long sequences. The FSDP data-parallel degree is automatically adjusted: `dp = world_size / ulysses_sequence_parallel_size`. Mini-batch sizes are also scaled down: `ppo_mini_batch_size //= ulysses_sequence_parallel_size`. Default `1` disables sequence parallelism (pure data parallelism). Increase for very long context lengths that exceed single-GPU memory.


## `ajet.lora`

LoRA fine-tuning. Disabled by default. Set `lora_rank > 0` to enable.

### `ajet.lora.lora_rank`

- **Type:** int.
- **Default:** `0`.
- **Description:** LoRA rank dimension. Controls the toggle between full fine-tuning and LoRA:
    - `0` — LoRA disabled, all model parameters are trainable (full fine-tuning).
    - `> 0` — LoRA enabled with the specified rank. Only low-rank adapter matrices are trainable, dramatically reducing memory and compute. Higher rank = more expressive adapters but more parameters. Common values: `8`, `16`, `32`, `64`.

    LoRA is also enabled when `lora_adapter_path` is configured (for loading pre-trained adapters), regardless of `lora_rank`.

### `ajet.lora.lora_alpha`

- **Type:** int.
- **Default:** `16`.
- **Description:** LoRA scaling factor. The adapter output is scaled by `alpha / rank` before being added to the base model output. A common practice is to set `lora_alpha = lora_rank` (scaling factor of 1.0) or `lora_alpha = 2 × lora_rank`. Higher values amplify the LoRA adapter's contribution.

### `ajet.lora.target_modules`

- **Type:** str.
- **Default:** `"all-linear"`.
- **Description:** Which model layers to apply LoRA adapters to. Options:
    - `"all-linear"` — Apply LoRA to all linear layers in the model (most comprehensive, recommended default).
    - Specific module names as a list, e.g. `["q_proj", "v_proj"]` — Apply only to attention query and value projections (fewer parameters, faster training).

    Passed to VeRL's LoRA initialization via `actor_rollout_ref.model.target_modules`.

### `ajet.lora.load_format`

- **Type:** str.
- **Default:** `"auto"`.
- **Description:** Weight loading format for the vLLM rollout engine when using LoRA. Mapped to `actor_rollout_ref.rollout.load_format`. Options:
    - `"auto"` — Auto-detect the weight format from the model directory.
    - `"safetensors"` — Explicitly use SafeTensors format.

    This primarily affects how the inference engine loads model weights, not how checkpoints are saved.

Example:

```yaml
ajet:
  lora:
    lora_rank: 32
    lora_alpha: 32
    target_modules: all-linear
    load_format: safetensors
```


## `ajet.task_runner`

### `ajet.task_runner.wrapper_type`

- **Type:** str.
- **Default:** `"asyncio-with-gc"`.
- **Description:** Controls how workflow execution is wrapped. Options:
    - `"asyncio-with-gc"` (recommended) — Wraps workflow execution in an async function with explicit `del user_workflow` cleanup after each task. Ensures the workflow object is properly dereferenced, allowing garbage collection to reclaim event loop references. Safe for long training runs with thousands of tasks.
    - `"asyncio"` — Direct `asyncio.run()` execution. Slightly faster but event loop references may persist in memory across task executions, causing gradual memory growth over long training runs.
    - `"multi-processing"` — Spawns a separate process per task with timeout enforcement. Completely isolates event loops (no leak risk) but has process creation overhead. Use `wrapper_multiprocessing_timeout` to configure the timeout.

### `ajet.task_runner.wrapper_multiprocessing_timeout`

- **Type:** int.
- **Default:** `3600`.
- **Description:** Timeout in seconds for the multi-processing wrapper (only effective when `wrapper_type: "multi-processing"`). If a workflow execution in the spawned process exceeds this duration, the process is forcefully terminated (`p.terminate()`) and a `TimeoutError` is raised. Default 3600 seconds (1 hour). Increase for complex long-running tasks.


## Swarm Mode

Swarm mode decouples rollout workers from the training loop. Workers can run on GPU-less machines.

### `ajet.enable_swarm_mode`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Enable swarm mode, which decouples rollout workers from the training loop. When enabled, the vLLM engine switches from `"sample"` mode to `"sample-ts"` (time-series, distributed), and rollout workers communicate via the interchange server using ZMQ sockets. Workers can run on GPU-less machines and receive `base_url`/`api_key` credentials for OpenAI-compatible inference. Requires `enable_interchange_server: true`.

### `ajet.enable_interchange_server`

- **Type:** bool.
- **Default:** `false`.
- **Description:** Enable the interchange server, which acts as a reverse proxy and API gateway for distributed training. The server provides: (1) a FastAPI-based HTTP API with OpenAI-compatible endpoints, (2) ZMQ middleware for point-to-point messaging between episodes and trainers, and (3) an episode registry tracking claimed/running/completed episodes. Required for swarm mode and for the `as_oai_baseurl_apikey` feature, which provides remote workers with an OpenAI-compatible `base_url` and auth token (`sk-ajet-{base64_encoded_auth_data}`) to access the training model.


## `ajet.interchange_server`

### `ajet.interchange_server.interchange_method`

- **Type:** str.
- **Default:** `"ipc"`.
- **Description:** Communication protocol for ZMQ messaging between episodes and trainers:
    - `"ipc"` — Unix domain sockets at `/tmp/ajet/{episode_uuid}-{tag}.sock`. Fastest option, single-node only. Raises a `ValueError` if used with `nnodes > 1`.
    - `"tcp"` — TCP sockets on dynamically allocated ports. Required for multi-node setups. Uses `MASTER_NODE_IP` environment variable (defaults to `"localhost"` for single-node).

### `ajet.interchange_server.interchange_server_port`

- **Type:** int or str.
- **Default:** `"auto"`.
- **Description:** HTTP port for the interchange server's FastAPI endpoint. When `"auto"`, a free port is found by binding to port 0 and reading the OS-assigned port. The resolved port is stored in the `AJET_DAT_INTERCHANGE_PORT` environment variable. Set to a fixed integer (e.g. `10086`, the swarm default) for predictable networking.

### `ajet.interchange_server.num_fastapi_process`

- **Type:** int.
- **Default:** `2`.
- **Description:** Number of uvicorn worker processes for the interchange server's HTTP API. Each process handles endpoints like `/register_episode`, `/update_engine_status`, `/update_current_batch_rollout_pool_information`, and proxied chat completion requests. Typical values: `1`, `2`, or `4`.

### `ajet.interchange_server.max_fastapi_threads`

- **Type:** int.
- **Default:** `512`.
- **Description:** Maximum threads in the `ThreadPoolExecutor` within each FastAPI worker process. Controls concurrency for handling blocking operations inside HTTP request handlers. Typical values: `64`, `128`, or `512`.

### `ajet.interchange_server.max_inference_tracker_threads`

- **Type:** int.
- **Default:** `64`.
- **Description:** Thread pool size for the inference result tracker, which asynchronously collects and processes inference results from remote episodes. Controls the `SharedInterchangeThreadExecutor` used by the interchange client. Recommended to set equal to `rollout.max_env_worker` for optimal parallelism.


## Sample Collection

### `ajet.swarm_mode_sample_collection_method`

- **Type:** str.
- **Default:** `"rollout_until_finish_enough_tasks"`.
- **Description:** Determines when to stop collecting rollout episodes and begin a training step. Options:
    - `"rollout_until_finish_enough_episodes"` — Stops when `total_completed_episodes >= train_batch_size × num_repeat`. Simplest method, but may have uneven task distribution (some tasks may have many more episodes than others).
    - `"rollout_until_finish_enough_tasks"` (default) — Stops when `total_completed_tasks >= train_batch_size`, where a task counts as "completed" only when it has `>= num_repeat` episodes. Ensures balanced coverage across tasks. Includes memory management: warns at 80% of `max_cached_episodes` and clears the cache if exceeded.
    - `"rollout_until_finish_enough_non_dummy_tasks"` — Same as above, but only counts tasks where the `num_repeat` episodes have **varying** rewards. Tasks where all episodes get the same reward (all success or all failure) are filtered out as "dummy" tasks that provide no training signal for GRPO advantage estimation.

### `ajet.swarm_mode_sample_collection_max_cached_episodes`

- **Type:** int.
- **Default:** `9999`.
- **Description:** Maximum number of episodes to cache in memory before triggering cleanup. Prevents unbounded memory growth when workers submit episodes faster than the trainer processes them. At 80% capacity, a warning is logged. At 100% capacity, the entire cache is cleared (`completed_task_id_map_ct.clear()`) and collection restarts. Increase for very large batch sizes or decrease if memory is constrained.


## Internal / Testing

These keys exist but are not intended for end users.

### `ajet.execute_test`

- **Type:** bool or str.
- **Default:** `false`.
- **Description:** Enable automated benchmark robot testing. When `true`, test probes monitor specific training metrics at predefined steps and validate they fall within expected ranges. Special value `"do_not_test"` explicitly disables testing. Test results can be reported to an external benchmark server via HTTP (requires `BENCHMARK_ACCESS_TOKEN`). **For CI/automated testing only—not intended for manual use.**

### `ajet.execute_testing_lambda`

- **Type:** str.
- **Default:** `""`.
- **Description:** Import path to a test probe class using `module.path->ClassName` syntax (e.g. `ajet.utils.testing_utils.BenchmarkProbe`). The class must inherit from `BaseProbe` and define a `probe_list` of metric keys to monitor. At each training step, the probe checks monitored metrics against expected ranges and raises `TestFailException` or `TestSuccessException` accordingly. **For CI/automated testing only.**


## Backend-Specific: verl

When `backbone: verl`, AgentJet translates `ajet.*` keys into the full verl configuration tree (defined in `ajet/default_config/verl/verl_default.yaml`). This means you write simple `ajet.*` keys in your YAML, and AgentJet automatically sets the correct verl-native keys for you.

### How Auto-Conversion Works

The mapping file `ajet/default_config/verl/config_auto_convertion_verl.jsonc` defines how each `ajet.*` key translates to one or more verl-native keys. There are two patterns:

**1:1 mapping** — one `ajet.*` key sets one verl key:

| `ajet.*` key (you write this) | verl key (set automatically) |
|---|---|
| `ajet.model.path` | `actor_rollout_ref.model.path` |
| `ajet.trainer_common.optim.lr` | `actor_rollout_ref.actor.optim.lr` |
| `ajet.trainer_common.total_epochs` | `trainer.total_epochs` |
| `ajet.data.train_batch_size` | `data.train_batch_size` |
| `ajet.lora.lora_rank` | `actor_rollout_ref.model.lora_rank` |
| `ajet.project_name` | `trainer.project_name` |

**1:N fan-out** — one `ajet.*` key sets multiple verl keys at once:

| `ajet.*` key | verl keys (all set to the same value) |
|---|---|
| `ajet.rollout.num_repeat` | `actor_rollout_ref.actor.rollout_n` + `actor_rollout_ref.rollout.n` |
| `ajet.rollout.max_model_len` | `actor_rollout_ref.rollout.max_model_len` + `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` + `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` + `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` |

This fan-out is why `ajet.*` keys exist — without them, you would have to set 4 separate verl keys just to change the context window length.

### Directly Overriding verl Keys

You almost never need to set verl keys directly. If you do (e.g. to tune a verl-specific parameter that has no `ajet.*` equivalent), add them as **top-level YAML keys alongside `ajet:`**:

```yaml
ajet:
  # ... your config ...

# Advanced: override verl keys directly (rarely needed)
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 32
    clip_ratio: 0.2
  rollout:
    gpu_memory_utilization: 0.85
    enforce_eager: true
```

### Notable verl Defaults

These are verl-native defaults (from `verl_default.yaml`) that you may want to be aware of. They can be overridden using the direct override syntax above.

| verl key | Default | Description |
|---|---|---|
| `actor_rollout_ref.actor.fsdp_config.dtype` | `bfloat16` | Mixed precision dtype |
| `actor_rollout_ref.actor.fsdp_config.use_torch_compile` | `true` | Enables torch compile for the actor |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | PPO clip ratio |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | Max tokens per GPU for actor training |
| `actor_rollout_ref.rollout.mode` | `async` | Async rollout mode |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.85` | vLLM GPU memory fraction |
| `actor_rollout_ref.rollout.enforce_eager` | `true` | Disables CUDA graphs |
| `actor_rollout_ref.rollout.enable_sleep_mode` | `true` | Sleeps vLLM between steps to free memory |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | Activation checkpointing |
| `algorithm.adv_estimator` | `grpo` | GRPO advantage estimator |
| `algorithm.gamma` | `1.0` | Reward discount |
| `algorithm.lam` | `1.0` | GAE lambda |


## CLI Flags

### `ajet`

```
ajet --conf <yaml> [--backbone verl|trinity|debug] [flags]
```

- `--conf PATH` — Path to experiment YAML. Required.

- `--backbone` — Override backbone. Options: `verl`, `trinity`, `debug`.

- `--exp-dir DIR` — Experiment output directory. Default: `saved_experiments`.

- `--autokill` — Kill stale ray/vllm/python processes before launch.

- `--kill KEYWORDS` — Kill processes matching `|`-separated keywords.

- `--with-ray` — Start a local Ray cluster.

- `--with-ray-cluster` — Connect to an existing Ray cluster.

- `--with-appworld` — Start the AppWorld environment service.

- `--with-deepfinance` — Start the DeepFinance environment.

- `--with-webshop` — Start the WebShop environment.

- `--with-bfcl` — Start the BFCL environment.

- `--with-crafters` — Start the Crafters environment.

- `--with-logview` — Start the log viewer.

- `--swarm-server` — Run as a swarm server.

- `--swarm-overwatch URL` — Monitor a swarm server.

- `--skip-check-avail-gpu` — Skip GPU free-memory check.

- `--debug`, `--db` — Debug helper flag.

### `ajet-swarm`

- `ajet-swarm start` — Start the swarm server. Key flags: `--swarm-port 10086`, `--conf`, `--exp-dir`.

- `ajet-swarm overwatch` — Monitor the swarm server (TUI). Key flags: `--swarm-url http://localhost:10086`, `--refresh-interval 2.0`.

- `ajet-swarm top` — Alias for `overwatch`.


## Environment Variables

- `DASHSCOPE_API_KEY` — API key for DashScope-based LLM-as-Judge. Pipe-separated for multiple keys.

- `DASHSCOPE_API_KEY_BACKUP` — Backup API key.

- `VERL_PYTHON` — Python executable for verl subprocess. Used in benchmarks.

- `APPWORLD_PATH` — Path to AppWorld data pack.

- `APPWORLD_SCRIPT` — Command to launch AppWorld.

- `AJET_SWARM_URL` — Swarm server URL for client scripts.

- `REMOTE_MODEL_PATH` — Model path on the swarm server.


## Common Recipes

### Reduce GPU Memory

When training large models on limited GPU hardware, combine several memory-saving techniques. FSDP offloading moves model parameters and optimizer states to CPU, freeing GPU VRAM at the cost of slower training steps. Reducing `max_model_len` shrinks the per-sequence memory allocation in the vLLM engine. Lowering `max_num_seqs` limits how many sequences vLLM batches in parallel, further reducing peak memory. A smaller `train_batch_size` reduces the number of samples processed per gradient update. Finally, enabling LoRA (e.g. `lora_rank: 16`) trains only low-rank adapter matrices instead of all model weights, dramatically cutting both memory and compute.

```yaml
ajet:
  trainer_common:
    fsdp_config:
      param_offload: true
      optimizer_offload: true
  rollout:
    max_model_len: 8000
    max_num_seqs: 4
  data:
    train_batch_size: 16
  lora:
    lora_rank: 16
```


### Validation Only (No Training)

Evaluate a model checkpoint without running any training steps. Setting `val_only: true` causes the trainer to exit immediately after validation, and `val_before_train: true` triggers the validation pass at step 0. This is useful for benchmarking a pre-trained or fine-tuned model, comparing checkpoints, or verifying that your workflow and judge produce sensible metrics before committing to a full training run. The pass@n metrics (controlled by `val_pass_n`) are computed as usual.

```yaml
ajet:
  trainer_common:
    val_only: true
    val_before_train: true
```


### Multi-Node Training

Scale training across multiple machines. Set `nnodes` to the number of nodes and `n_gpus_per_node` to match the hardware. The total GPU count (`nnodes × n_gpus_per_node`) determines the FSDP sharding degree, and `train_batch_size × num_repeat` must be divisible by it. For multi-node setups, you also need to configure the interchange server with `interchange_method: "tcp"` (IPC only works on a single node) and set the `MASTER_NODE_IP` environment variable.

```yaml
ajet:
  trainer_common:
    nnodes: 2
    n_gpus_per_node: 8
```


### Swarm Mode (Minimal)

Decouple rollout workers from the training loop so that environment interactions can run on separate (potentially GPU-less) machines. The interchange server acts as a gateway — remote workers receive OpenAI-compatible API credentials and send completed episodes back. Use `interchange_method: tcp` for multi-node communication and set a fixed port (e.g. `10086`) for predictable networking. Remote workers connect to this port to submit rollout results. See the [Swarm Mode Config Chain](#swarm-mode-config-chain) section for how defaults are layered.

```yaml
ajet:
  enable_swarm_mode: true
  enable_interchange_server: true
  interchange_server:
    interchange_method: tcp
    interchange_server_port: 10086
```


### Fast Debug Iteration (Deprecated, recommend to use swarm mode instead)

Use debug mode to quickly test your workflow and reward function without launching a full distributed training loop. Debug mode connects to an external vLLM server and only runs inference + rollout — no weight updates occur. Setting `debug_max_parallel: 1` and `debug_first_n_tasks: 1` runs a single task sequentially, making it easy to step through the logic and inspect outputs. You can also override the backbone from the command line with `--backbone=debug` without modifying your YAML.

```yaml
ajet:
  backbone: debug
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```

```bash
ajet --conf my_config.yaml --backbone=debug
```




## Next Steps

<div class="card-grid">
<a href="../workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:workflow.svg" class="card-icon card-icon-general" alt=""><h3>Workflow</h3></div><p class="card-desc">How to define your trainable agent workflow.</p></a>
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:database.svg" class="card-icon card-icon-general" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Task readers and data formats in detail.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:scale.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Built-in and custom reward functions.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent Example</h3></div><p class="card-desc">See all configuration applied in a real training run.</p></a>
</div>
