# `terminate_rollout` (sleep_replica) vLLM Worker Crash — Bug Trace

> Box: `tui-dlc6vshe20lhf3b3-master-0` (`/mnt/data_cpfs/qingxu.fu/agentjet`), verl 0.8.0.dev0, vLLM 0.26, swarm/async rollout.
> Status: **cause determined (assumed from code) + fix applied (ajet-internal), awaiting real-run validation.**

## 1. Symptom

Right after `ENGINE.WEIGHT_SYNCING`, a vLLM TP worker dies during a model step:

```
fit:392 gen_batch_output.info batch.keys=...
Changed engine status to ENGINE.WEIGHT_SYNCING
(vLLMHttpServer pid=1079380, ip=10.29.255.116) (Worker_TP7 pid=...) ERROR
  ... gpu_model_runner.execute_model -> _prepare_inputs -> commit_block_table
  -> block_table.copy_to_gpu -> self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
  torch.AcceleratorError: CUDA error: invalid argument
```

Worker_TP7 dies → cascade → engine `OFFLINE`. Looks like OOM/driver crash but isn't.

## 2. Root cause (assumed, from code — elimination)

The crashing `generate` is **case (3): the request entered vLLM *before* `sleep_replicas`, was queued in the engine, `sleep_replicas` freed the KV-cache without aborting/draining it, and the next `execute_model` step committed its block table to the now-freed GPU buffer.**

Elimination (each nailed by code):

- **(2) "stuck in verl, only reached vLLM after sleep" — NO.** Crash stack is *inside* the engine (`execute_model`→`_prepare_inputs`→`commit_block_table`), i.e. the request was already scheduled by the vLLM scheduler. "Stuck in verl" cannot produce an `execute_model` frame. And `async_rollout_manager.generate` submits straight to the engine core (no verl buffer queue).
- **(1) "generate issued after sleep_replica" — NO.** After `sleep_replicas()` the fit loop only does `response_mask` / `balance_batch` / reward / `update_actor` — no generate, no re-entry into rollout. The bridge staleness guard (`_stop_writing_new_timeline`, set during the rollout hard-stop, before sleep) only discards results post-generate; it issues no new requests.
- **(4) "mid-forward when sleep hit" — NO.** The frame is `_prepare_inputs`→`commit_block_table`→`copy_to_gpu` — **input preparation at the very start of a freshly-dispatched step**, not an attention/matmul kernel. A clean synchronous traceback there = a new step being set up *after* sleep freed memory, i.e. a queued request dispatched post-sleep.

Positive evidence for (3): the five `run_infer:451 request outdated: stop_writing_new_timeline` logs (18:54–18:55, *after* `end batch rollout` at 18:53). In `async_llm_bridge.run_infer` the `await self.llm_inference_fn(...)` (= `async_rollout_manager.generate`) runs **before** the staleness check, so those requests *did* reach vLLM and completed (result then discarded). They are siblings of the crashing request — same rollout-tail wave; one of them simply did not finish before sleep.

**The verl safety hole:** `verl/checkpoint_engine/base.py::CheckpointEngine.sleep_replicas()` is just `await asyncio.gather(*[r.sleep() for r in self.replicas])` — it frees KV-cache memory **without aborting or draining in-flight requests** (contrast `update_weights()`, which calls `r.abort_all_requests()` first). And in ajet's *naive* colocated backend, `update_weights()` returns early, so nothing ever clears in-flight requests before sleep.

## 3. Why in-flight requests exist at sleep time (ajet swarm architecture)

`swarm_runner.execute` registers episodes to the **interchange server** via HTTP (`register_episode_and_wait_output`) and the local thread blocks on the result. The rollout's hard-stop flips `observation_window["stop"]`, so `rollout_env_worker_loop` threads return (`end batch rollout`) **without** waiting for the remote episode workers to finish. Those remote episodes keep calling `generate` → vLLM; their late results are discarded via `stop_writing_new_timeline`. So at `sleep_replicas()` time the vLLM engine still has in-flight requests from the just-ended rollout.

## 4. Fix (ajet-internal — verl untouched)

`ajet/backbone/trainer_verl.py` — new helper, replaces all 3 bare `self.checkpoint_manager.sleep_replicas()` call sites:

```python
def _drain_and_sleep_replicas(self):
    """Drain in-flight rollout requests, then sleep replicas.

    verl's ``CheckpointEngine.sleep_replicas()`` frees KV-cache memory
    without draining in-flight requests, so a request still queued/running
    in the rollout engine hits a freed block-table buffer on the next
    ``execute_model`` step -> ``CUDA error: invalid argument`` in
    ``block_table.copy_to_gpu`` (kills the VLLMHttpServer worker). In ajet
    swarm mode the rollout returns on hard-stop while remote episodes still
    have in-flight requests; drain them on the rollout server handles (each
    drains the whole vLLM engine across all TP workers) before sleeping.
    """
    import ray
    handles = getattr(self.async_rollout_manager, "server_handles", None) or []
    refs = [h.wait_for_requests_to_drain.remote() for h in handles]
    if refs:
        ray.get(refs)
    self.checkpoint_manager.sleep_replicas()
```

Call sites replaced: `init_workers` tail (init), `fit` post-rollout (the crash site), `fit` post-validate.

Why this shape:
- **Drain, not abort.** `abort_all_requests` internally `pause_generation`s and *leaves the engine paused* (vllm_async_server docstring). In naive mode `update_weights()` does **not** `resume_generation`, so an abort here would leave the engine paused → next rollout hangs. `wait_for_requests_to_drain` is gentle (no pause/resume), lets late episodes finish normally (their `await generate` gets a real result), and leaves the engine idle-and-running for sleep.
- **In ajet, not verl** (per directive). Reuses `self.async_rollout_manager.server_handles` (ajet already couples to it at the `init_workers` server zip) — each is a rollout-replica primary `VLLMHttpServer` actor whose `wait_for_requests_to_drain()` (`await self.engine.wait_for_requests_to_drain()`) drains the **whole** vLLM engine across all TP workers. No verl file edited; uses verl's already-existing actor method.

## 5. Validation

- `python -m py_compile ajet/backbone/trainer_verl.py` → OK.
- Structure verified: 1 helper def; helper's internal `self.checkpoint_manager.sleep_replicas()` preserved; exactly 3 call sites now `_drain_and_sleep_replicas()`.
- **Pending: real training run.** Expect a brief drain pause between `ENGINE.WEIGHT_SYNCING` and sleep when late episodes finish; no more `CUDA error: invalid argument` at `commit_block_table`.

## 6. Alternatives considered

- **Patch verl `sleep_replicas` (abort, or add adapter `wait_for_requests_to_drain`).** Correct layer, but (a) abort breaks naive next-cycle (pause not resumed); (b) rejected by directive — fix must live in ajet.
- **Drain at the rollout (fix the hard-stop leak).** Deeper; the late-episode behavior is intentional (results discarded). Draining at the sleep boundary covers all callers uniformly.
- **`update_weights`-style abort+resume in the helper.** More state churn than a pure drain; drain is strictly safer for the still-running episode workers.

## 7. Files / backups

- Changed: `ajet/backbone/trainer_verl.py`.
- Backup: `/tmp/trainer_verl.py.bak.terminate` (revert: `cp` back).
- verl backups (untouched, made pre-emptively then not used): `/tmp/base.py.bak.terminate`, `/tmp/replica.py.bak.terminate`.


============= next ===========



---

## Revision 2 — wake_up `KeyError: req_id_to_index` (drain was the wrong tool)

**Symptom after Revision 1 (drain):** step 1 now completes (the original sleep CUDA crash is gone — real progress), but the next cycle's `wake up begin` → `update_weights` crashes inside the engine:

```
vllm/v1/core/sched/scheduler.py:1670 update_from_output
  req_index = model_runner_output.req_id_to_index[req_id]
KeyError: 'b3bcacfd3e5d4ea5917ce071cce964f3-93150614'
```

Same root cause (a request left in the engine across sleep), new symptom. The drain approach was wrong:

- `wait_for_requests_to_drain` (`vllm/v1/engine/async_llm.py:978`) is **not** a sleep helper — it's used by `scale_elastic_ep`. Its body just polls `engine_core.dp_engines_running()` every 1s until "idle", **it removes nothing**. A request that is *queued but not currently executing* makes the engine report idle → drain returns → the request stays in the scheduler → frozen by sleep → at wake the scheduler still considers it active but the runner's per-step `req_id_to_index` doesn't → `KeyError`. (Confirmed by operator: no new requests at wake_up — the stale id is a pre-sleep leftover, not a fresh submission.)

So `wait_for_requests_to_drain` only checks "is the engine running", not "are there requests". Insufficient.

**Final fix (Revision 2):** switch the helper from drain to **`abort_all_requests` + `resume_generation`** (renamed `_drain_and_sleep_replicas` → `_safe_sleep_replicas`):

```python
def _safe_sleep_replicas(self):
    import ray
    handles = getattr(self.async_rollout_manager, "server_handles", None) or []
    if handles:
        ray.get([h.abort_all_requests.remote() for h in handles])   # finish_requests(ABORTED) -> removes from scheduler
        ray.get([h.resume_generation.remote() for h in handles])   # abort pause_generation's; engine empty AND running
    self.checkpoint_manager.sleep_replicas()
```

- `abort_all_requests` (`vllm_async_server.py:677`) → `pause_generation` + `finish_requests(..., FINISHED_ABORTED)` on every known request → **definitively removes them from the scheduler**. No leftover → no CUDA-at-sleep AND no KeyError-at-wake.
- `resume_generation` (`vllm_async_server.py:740`) undoes the pause so the next rollout isn't starved (naive-mode `update_weights` doesn't resume on its own).
- This is exactly verl `update_weights`'s abort (step 1) + resume (step 6) pair, just without the weight transfer.

**Validation:** `py_compile` OK; structure verified (helper + 3 call sites: init / post-rollout / post-validate). Pending a real run: expect neither the `commit_block_table` CUDA error nor the `req_id_to_index` KeyError across the WEIGHT_SYNCING→sleep→wake transition.

**Backups:** original `/tmp/trainer_verl.py.bak.terminate`; drain version `/tmp/trainer_verl.py.bak.drain`.

---

## Validation result (2026-08-14 ~00:50, real run)

Fix **validated** on the real swarm (restarted via `activate.sh` clean-restart; ajet editable, so the new engine loaded `_safe_sleep_replicas`). Both original crash points were exercised cleanly:

- **Stage ①** init `_safe_sleep_replicas` (abort+resume+sleep): engine booted and entered step 1 rollout — no crash. `abort_all_requests`/`resume_generation` API calls work.
- **Stage ② (the core validation)** step 1 end → sleep → step 2 `wake up`: episodes are now claimed for **`global step: 2`**, i.e. the step-2 `wake up begin → update_weights` completed **without** the `KeyError: req_id_to_index` and the step-1-end sleep completed **without** the `CUDA error: invalid argument` / `commit_block_table` crash.

Neither symptom recurred. The `wait_for_requests_to_drain` (Revision 1) → `abort_all_requests + resume_generation` (Revision 2) switch is what fixed the wake desync. Monitor (`/tmp/wake_monitor.sh`, PID 1460727) will write `SUCCESS_STEP2` to `/tmp/wake_verdict.txt` when step 2 finishes its full cycle (~50 min) as final confirmation.
