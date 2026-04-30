# DeepFinance: Training a Financial Deep-Research Agent with Reinforcement Learning

> This article is a translated version of the [Chinese original](./example_deep_finance.zh.md).

## Overview

DeepFinance is a training recipe for financial deep-research Agents built on top of the AgentJet framework. The goal: use GRPO reinforcement learning to teach an LLM to autonomously call financial tools, gather data from multiple sources, cross-validate it, and finally produce structured, well-cited investment research reports.

Unlike traditional SFT, DeepFinance does **not** rely on human-written "ground truth" answers as training supervision. Instead, it designs a **multi-dimensional reward system** that serves as the RL training signal — letting the model explore optimal report-writing strategies on its own, guided by feedback from 5 orthogonal scoring dimensions.

**Training loop**:

```plain
Financial question → Agent calls tools to collect data → Generates research report → Multi-dimensional Judge scoring → GRPO policy update → Next rollout
```

------

## Pipeline

The training pipeline is composed of 4 core modules:

| Module       | File                               | Responsibility                                                       |
| ------------ | ---------------------------------- | -------------------------------------------------------------------- |
| **Reader**   | `deep_finance_reader.py`           | Loads JSON training data, assembles System Prompt + User Query       |
| **Workflow** | `deep_finance.py`                  | Defines the multi-turn ReAct Agent logic and maintains chat history  |
| **Judge**    | `deep_finance_judge.py` + `judge/` | Multi-dimensional reward scoring (the core innovation)               |
| **Config**   | `deep_finance.yaml` / `*.sh`       | Training hyperparameters, reward weights, environment configuration  |

```plain
┌─────────────────────────────────────────────────────────────┐
│                    AgentJet Training Framework               │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────┐               │
│  │ DeepFinance   │    │  ExampleDeepResearch │               │
│  │ Reader        │───>│  Protocol (Workflow) │               │
│  │ Data load +   │    │  Multi-turn ReAct    │               │
│  │ Prompt assemb.│    └──────────┬───────────┘               │
│  └──────────────┘               │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  EnvService (FinWorld) │               │
│                    │  19 financial tools+MCP│               │
│                    │  MongoDB caching       │               │
│                    └────────────┬───────────┘               │
│                                 │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  DeepFinanceJudge      │               │
│                    │  Multi-dim reward      │               │
│                    │  (built on OpenJudge)  │               │
│                    └────────────┬───────────┘               │
│                                 │                           │
│                                 v                           │
│                    ┌────────────────────────┐               │
│                    │  GRPO Trainer (verl)   │               │
│                    │  Multi-node Ray cluster│               │
│                    └────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

------

## Workflow Design

### Two-Stage Deep-Research Procedure

The Agent's System Prompt (`prompt/finance_analyst_prompt.md`) requires the model to follow a two-stage research method:

**Stage 1: Outline first, then investigate**

1. Identify the type of user question (single-stock analysis / sector study / event interpretation / macro analysis / stock screening).
2. **Output a research outline first** (H1/H2 headings + Key Questions per section). No tool calls in this stage.
3. Investigate section-by-section along the outline, summarizing after every round of tool calls.

**Stage 2: Deep analysis and report generation**

1. Once data is sufficient, generate a Markdown research report grounded in real data.
2. If evidence is found lacking during writing, the Agent is allowed 1–2 additional tool-call rounds to gather more support.
3. End the report with a `[TASK_COMPLETED]` marker.

### Citation Convention

The Agent is required to follow an academic-paper-style citation format:

- Every key factual sentence must end with a citation marker `[n]`.
- The report must include a `## References` section at the end.
- Citations must be traceable to actual tool-returned data — fabrication is forbidden.

------

## Tool Suite

DeepFinance integrates **19 financial tools**, exposed to the Agent over the MCP (Model Context Protocol) channel via EnvService. They cover the full data needs of financial research.

| Category                  | Tool                    | Function                                                  |
| ------------------------- | ----------------------- | --------------------------------------------------------- |
| **Entity & Computation**  | `extract_entities_code` | Extract financial entities from natural language and look up codes |
|                           | `history_calculate`     | A-share historical price analysis (natural-language Q&A)  |
| **General Capability**    | `dashscope_search`      | Internet search                                           |
|                           | `execute_code`          | Python code execution                                     |
|                           | `execute_shell`         | Shell command execution                                   |
| **Tonghuashun Data**      | `crawl_ths_company`     | Listed-company basic profile                              |
|                           | `crawl_ths_holder`      | Shareholder research                                      |
|                           | `crawl_ths_operate`     | Operations analysis                                       |
|                           | `crawl_ths_finance`     | Financial analysis                                        |
|                           | `crawl_ths_worth`       | Earnings forecasts                                        |
|                           | `crawl_ths_news`        | News & announcements                                      |
|                           | `crawl_ths_concept`     | Concept / thematic info                                   |
|                           | `crawl_ths_equity`      | Equity structure                                          |
|                           | `crawl_ths_capital`     | Capital operations                                        |
|                           | `crawl_ths_position`    | Major-holder positions                                    |
|                           | `crawl_ths_bonus`       | Dividends & financing                                     |
|                           | `crawl_ths_event`       | Major corporate events                                    |
|                           | `crawl_ths_field`       | Industry comparison                                       |

Tool-call rules:

- At most **3 tools per turn**, encouraging multi-round progressive investigation.
- The Agent must search to confirm information (e.g. ticker codes) before drilling down.
- After every round of tool calls, summarize first, then decide on the next research direction.

------

## Reward Design

This is DeepFinance's core innovation. We design **5 orthogonal scoring dimensions** (Graders) and combine them via configurable weights into the final reward, plus an additional tool-call penalty.

### Overall Formula

```plain
final_reward = Σ(w_i × grader_i_score) + tool_penalty
```

Where the grader weights are normalized (`Σw_i = 1`), and `tool_penalty` is an additional adjustment.

### The 5 Scoring Dimensions

| Dimension                | Name                | Evaluates                          | Core Question                                                               |
| ------------------------ | ------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| **Analytical Adequacy**  | RM Gallery          | Overall report quality             | Is the analysis thorough? Is the reasoning sound?                           |
| **Presentation Quality** | PresentationQuality | Layout and structure of the report | Is it pleasant to read? Is information easy to extract?                     |
| **Citation Compliance**  | Grounding           | Citation coverage and authenticity | Do all key facts have citations? Are the citations real?                    |
| **Evidence Traceability**| EBTU                | Evidence anchoring of atomic claims| Can each number/fact be traced back to original tool-returned data?         |
| **Citation-Logic Audit** | Audit               | Logical entailment of citations    | Do the citations actually support the corresponding statements? Any inflation/fabrication? |

Default weight configuration (tunable in the shell scripts):

```bash
RM_WEIGHT=0.5                       # Analytical adequacy
PRESENTATION_QUALITY_WEIGHT=0.2     # Presentation quality
GROUNDING_WEIGHT=0.1                # Citation compliance
EBTU_WEIGHT=0.2                     # Evidence traceability (optional)
AUDIT_WEIGHT=0.0                    # Citation-logic audit (optional)
```

------

### 1) Analytical Adequacy (RM Gallery)

**Goal**: Evaluate the analytical depth, coverage, and reasoning of the report — answering "is the analysis any good?"

**Mechanism**: Uses the `finance_composition` evaluator. An independent Judge LLM (`qwen-max`) compares the generated report against a reference answer.

**Evaluation aspects (sharded by financial domain)**:

- Analytical depth: how deeply the core question has been investigated.
- Coverage: whether multiple analytical angles are covered (fundamentals, financials, valuation, industry, news, …).
- Reasoning: completeness of the reasoning chain, soundness of conclusions.

**I/O**:

- Input: User Query + Agent-generated report + reference answer.
- Output: a normalized score in `[0, 1]`.

------

### 2) Presentation Quality

**Goal**: Evaluate user experience and information architecture — answering "is it well-laid-out and easy to read?"

**Strictly does NOT evaluate**: factual correctness, citation accuracy, content depth (those are handled by other graders).

**8 sub-metrics (1/3/5 scale)**:

| Category                       | Metric                          | 5-point Standard                                                                  |
| ------------------------------ | ------------------------------- | --------------------------------------------------------------------------------- |
| **Scan-ability**               | A1 Conclusion-first             | Standalone abstract / TL;DR at the top — main conclusion is visible without scrolling |
|                                | A2 Structural Navigation        | Clear hierarchy (H1/H2/H3); long pieces have explicit signposts                   |
|                                | A3 Visual Emphasis              | Bold/italics used precisely to highlight core insights; high signal-to-noise      |
| **Information Structuring**    | B1 Decomposing Dense Info       | Complex data presented via tables / nested lists at a glance                      |
|                                | B2 Comparison & Alignment       | A vs B / past vs present uses tables, with horizontally comparable dimensions     |
|                                | B3 Consistency & Rendering      | Uniform formatting, clean Markdown rendering                                      |
| **Editorial Clarity**          | C1 Visible Argument Chain       | Logical chain is visible (claim → evidence → conclusion); citation anchors clear  |
|                                | C2 Risks & Actions              | Standalone section listing risks/limitations and next-step recommendations        |

**Scoring**:

```plain
score = Σ(8 sub-scores) / 40    # normalized to [0, 1]
```

**Anti-gaming**: empty tables, meaningless repeated lists, format-for-the-sake-of-format → flat 1 point.

------

### 3) Citation Compliance (Grounding)

**Goal**: Evaluate citation coverage and authenticity — answering "do all the key facts have sources, and are the citations real?"

**Procedure**:

1. Extract User Query, Evidence (tool calls + returns), and the final report from the conversation trace.
2. An LLM auditor identifies all "key factual sentences" (containing numbers / dates / financial metrics / definitive statements).
3. Check whether each key sentence ends with a citation marker `[n]`.
4. Check that each citation has a valid entry in the References section (valid URL or a complete no-url record).
5. Check whether the citation content is consistent with Evidence (detect fake citations).

**Output fields**:

- `total_key_facts`: total number of key factual sentences.
- `cited_key_facts`: number of those that end with a citation.
- `fake_count`: citations clearly contradicting the evidence.
- `missing_count`: key facts lacking a citation.
- `invalid_reference_nums`: malformed reference numbers.

**Scoring**:

```plain
citation_coverage = cited_key_facts / total_key_facts     # citation coverage
grounding_score = 1 - fake_count / cited_key_facts        # citation authenticity
final_score = 0.5 × coverage + 0.5 × grounding            # combined score
```

------

### 4) Evidence Traceability (EBTU – Evidence-Backed Trace Units)

**Goal**: Audit each "atomic claim" in the report for evidence anchoring — answering "can every number, every fact, be traced back to data returned by a tool?"

**Core principle: Evidence-first.** The auditor must produce evidence anchors (step + quote) **before** issuing a verdict; reasoning backwards from a conclusion to "find" evidence is forbidden.

**Audit procedure**:

1. Extract all atomic claims (Trace Units) from the report and tag each with a type (numeric / temporal / event / comparison / causal / …).
2. Tag hardness: `hard` (definitive fact) vs `soft` (explicitly marked as speculation/hypothesis).
3. For each claim, find anchors in Evidence:

- - Precise to a step number and an in-line quote (≤ 120 chars).
  - Numbers / dates must literally appear in the Evidence text.

4. Issue a verdict:

| Verdict          | Meaning                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `supported`      | The anchor directly supports the claim                               |
| `contradicted`   | The anchor explicitly conflicts with the claim                       |
| `no_evidence`    | No support in Evidence, and the claim is presented as definitive     |
| `speculative_ok` | The claim is explicitly speculative/hypothetical, not disguised fact |
| `unclear`        | Evidence is related but insufficient to support or refute            |

5. Tag the issue type: `entity_mismatch` / `time_mismatch` / `value_mismatch` / `scope_mismatch` / `logic_leap` / `over_precision` / `missing_anchor`.

**Scoring** (deterministic, computed in Python — not produced by the LLM):

```plain
base = (supported - 1.4×contradicted - 0.9×no_evidence - 0.4×unclear) / hard_units
misattrib_factor = max(0, 1 - 0.7 × misattrib_rate)     # misattribution penalty
selection_factor = min(1, extracted_units / expected)   # coverage factor
cov_factor = 0.65 + 0.35 × digit_coverage               # number/date coverage
score = base × misattrib_factor × selection_factor × cov_factor
```

Key design point: the LLM only emits structured outputs (claim extraction + anchor labels + verdicts); the score itself is computed deterministically by code, avoiding the instability of LLM self-scoring.

------

### Tool-Call Penalty

On top of the weighted score, an extra tool-call penalty encourages the Agent to actively gather data with tools:

| Tool-call count | Penalty           |
| --------------- | ----------------- |
| 0               | -1.0              |
| 1–2             | -0.5              |
| ≥ 3             | 0.0 (no penalty)  |

------

## Quick Start

### Environment Setup

1. Install AgentJet and its dependencies:

```bash
cd /path/to/AgentJet
bash install.sh
```

2. Configure the `.env` file (API keys, model paths, data paths, etc.):

```bash
# Example .env
MODEL_PATH=/path/to/Qwen3-8B
TRAIN_DATA_PATH=/path/to/train.json
VAL_DATA_PATH=/path/to/val.json
TRAIN_REF_ANS_PATH=/path/to/train_ref_answers.json
VAL_REF_ANS_PATH=/path/to/val_ref_answers.json
CKPT_SAVE_PATH=/path/to/checkpoints
OPENJUDGE_API_KEY=your_api_key
RM_API_KEY=your_api_key
```

3. Start EnvService (the financial-tools service).

### Single-Node Debugging Mode

```bash
bash tutorial/example_deep_finance/deep_finance_single.sh
```

This script runs with `--backbone="debug"`, which is ideal for validating the workflow and debugging.

### Multi-Node Training Mode

```bash
# Submit on PAI-DLC or any multi-node environment
bash tutorial/example_deep_finance/deep_finance.sh
```

This script will:

1. Generate the config file dynamically from the YAML template.
2. Start Ray Head + the training task on the master node.
3. Worker nodes automatically join the Ray cluster.

### Key Parameters

| Parameter                     | Default | Description                                                  |
| ----------------------------- | ------- | ------------------------------------------------------------ |
| `NUM_REPEAT`                  | 4       | Group size — number of rollouts per query                    |
| `NUM_STEPS`                   | 6       | Max number of interaction rounds per sample                  |
| `TRAIN_BATCH_SIZE`            | 32      | Training batch size                                          |
| `RM_WEIGHT`                   | 0.5     | Analytical-adequacy weight                                   |
| `PRESENTATION_QUALITY_WEIGHT` | 0.25    | Presentation-quality weight                                  |
| `GROUNDING_WEIGHT`            | 0.25    | Citation-compliance weight                                   |
| `EBTU_WEIGHT`                 | 0.0     | Evidence-traceability weight (optional)                      |
| `AUDIT_WEIGHT`                | 0.0     | Citation-logic-audit weight (optional)                       |

------

## Experimental Results


![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843906200-9dd35ac4-f71e-40dc-b130-f03e3e6bae6a.png)

![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843940824-4e3637d7-a16e-4994-8878-242effc2c0d7.png)![img](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/107756372/1771843950142-09def779-5521-41f0-a457-a7715a819cc7.png)
