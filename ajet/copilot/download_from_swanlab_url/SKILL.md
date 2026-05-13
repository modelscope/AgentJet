# Skill: Download metric data from a SwanLab run URL

## Goal

Given a SwanLab cloud URL of the form

```
https://swanlab.cn/@<username>/<project>/runs/<exp_cuid>/chart
```

fetch the per-step time-series data for one or more metrics (e.g. reward, entropy, response length) as a `pandas.DataFrame` for downstream plotting or analysis.

## Trial-and-error log (read first; saves hours)

The shortcuts you might try **do not work**:

1. **WebFetch / `curl` of the chart page** — the page is a Vue SPA. The HTML body is just `<div id="app"></div>`; no chart data is embedded. Don't bother scraping HTML.

2. **Direct REST probe (`curl https://api.swanlab.cn/api/v1/runs/<cuid>`)** — returns `404 Not Found` even with a valid api key. The public REST surface is not the v1 path; use the Python SDK.

3. **`swanlab verify` says you're logged in** — but the env may point to a private cloud. On this host:
   ```
   $ swanlab verify
   swanlab: You are logged into https://cloud-20.agent-matrix.com as fuqingxu
   ```
   This is **not** swanlab.cn. The shell exports `SWANLAB_API_HOST` and `SWANLAB_WEB_HOST` in `~/.bashrc`, redirecting the SDK to a different deployment. Calling `OpenApi(api_key=...)` then fails with `Login failed: 404 Not Found` because `/login/api_key` only exists on swanlab.cn's API host.

4. **Metric keys are NOT what the UI shows.** The chart card titled "reward" is logged as `critic/rewards/mean`. Asking `get_metrics(keys=['reward'])` returns code 404. You **must** probe candidate keys (see the cheat sheet below).

5. **`get_summary` to enumerate keys** — does not work for non-cloned runs. When `rootExpId` / `rootProId` are `None` (i.e. the run was not cloned from another project), `experiment.get_summary` returns HTTP 400 "Bad Request". Skip it; probe candidate keys directly with `get_metrics`.

## The working recipe

### 1. Prerequisites

- `swanlab` Python package (>= 0.7.4 confirmed).
- An api key for `swanlab.cn`. On this machine it is stored in `~/.swanlab/.netrc`:
  ```
  machine https://api.swanlab.cn
      login https://swanlab.cn
      password <API_KEY>
  ```
  Read the password field from there if you don't have it on hand.

### 2. Parse the URL

```python
import re

URL = "https://swanlab.cn/@binaryhusky/spy-game-rl/runs/zku3ujg2k3unvt61jbu0s/chart"
m = re.match(r"https?://swanlab\.cn/@([^/]+)/([^/]+)/runs/([^/]+)", URL)
username, project, exp_id = m.group(1), m.group(2), m.group(3)
```

### 3. Override env vars BEFORE importing `swanlab`

Critical: if your shell has `SWANLAB_API_HOST` / `SWANLAB_WEB_HOST` / `SWANLAB_API_KEY` pointing to a private cloud, the SDK silently uses them. Either run via `env -i` or unset them:

```bash
unset SWANLAB_API_KEY SWANLAB_API_HOST SWANLAB_WEB_HOST
SWANLAB_API_HOST=https://api.swanlab.cn/api \
SWANLAB_WEB_HOST=https://swanlab.cn \
python your_script.py
```

Note: `SWANLAB_API_HOST` for swanlab.cn ends with `/api` (the SDK default in `swanlab/env.py`); without that suffix login also returns 404.

### 4. Open the API and fetch metadata

```python
import swanlab
api = swanlab.OpenApi(api_key="<API_KEY>")  # raises ValidationError if hosts wrong

exp = api.get_experiment(project=project, exp_id=exp_id, username=username)
assert exp.code == 200, exp.errmsg
print(exp.data.name, exp.data.state)
# exp.data.profile['config'] contains the full training config (verl-style nested dict)
```

### 5. Discover the right metric keys

The keys you ask for must match what was logged. For a verl-based AgentJet run, the working names are:

| You might think | Actual logged key       |
| --------------- | ----------------------- |
| `reward`        | `critic/rewards/mean`   |
| `entropy`       | `actor/entropy`         |
| `response_length` | `response_length/mean` |
| `pg_loss`       | `actor/pg_loss`         |

Other common ones: `response_length/max`, `response_length/min`, `critic/score/mean`. Probe defensively — `get_metrics` returns `code=200` and N rows on hit, `code=404, rows=0` on miss:

```python
candidates = [
    "critic/rewards/mean", "critic/score/mean",
    "actor/entropy", "actor/entropy_loss", "actor/pg_loss",
    "response_length/mean", "response_length/max", "response_length/min",
]
for k in candidates:
    r = api.get_metrics(exp_id=exp_id, keys=k)
    print(f"{k:40s} code={r.code} rows={0 if r.data is None else len(r.data)}")
```

### 6. Fetch and save

```python
keys = ["critic/rewards/mean", "actor/entropy", "response_length/mean"]
r = api.get_metrics(exp_id=exp_id, keys=keys)
df = r.data  # indexed by `step`; one column per key + one `<key>_timestamp` column
df.to_csv("metrics.csv")
```

The DataFrame layout is:

```
              actor/entropy  actor/entropy_timestamp  critic/rewards/mean  ...
step
1                  0.5569            1774003810000               0.7271
2                  0.5732            1774004325000               0.7589
...
```

The `_timestamp` columns are unix millis; usually you can drop them and plot against the `step` index.

## End-to-end runnable snippet

```python
"""Fetch reward/entropy/response_length curves from a swanlab.cn run URL."""
import os, re, sys
# Strip any private-cloud overrides BEFORE importing swanlab.
for v in ("SWANLAB_API_KEY", "SWANLAB_API_HOST", "SWANLAB_WEB_HOST"):
    os.environ.pop(v, None)
os.environ["SWANLAB_API_HOST"] = "https://api.swanlab.cn/api"
os.environ["SWANLAB_WEB_HOST"] = "https://swanlab.cn"

import swanlab

URL = sys.argv[1]
API_KEY = sys.argv[2]  # or read from ~/.swanlab/.netrc
m = re.match(r"https?://swanlab\.cn/@([^/]+)/([^/]+)/runs/([^/]+)", URL)
username, project, exp_id = m.groups()

api = swanlab.OpenApi(api_key=API_KEY)
keys = ["critic/rewards/mean", "actor/entropy", "response_length/mean"]
r = api.get_metrics(exp_id=exp_id, keys=keys)
assert r.code == 200, r.errmsg
df = r.data.rename(columns={
    "critic/rewards/mean": "reward",
    "actor/entropy": "entropy",
    "response_length/mean": "response_length",
})
df.to_csv("metrics.csv")
print(df.head())
```

## Plot recipe (seaborn, optional)

```python
import seaborn as sns, matplotlib.pyplot as plt, pandas as pd
df = pd.read_csv("metrics.csv").sort_values("step")
sns.set_theme(context="paper", style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
palette = sns.color_palette("deep")
for ax, (col, title, c) in zip(axes, [
    ("reward", "Reward", palette[0]),
    ("entropy", "Policy Entropy", palette[3]),
    ("response_length", "Response Length", palette[2]),
]):
    sns.lineplot(data=df, x="step", y=col, ax=ax, color=c, linewidth=1.6)
    ax.set_title(title); ax.set_xlabel("Training step"); ax.set_ylabel("")
fig.tight_layout()
fig.savefig("curves.pdf", bbox_inches="tight")
```

## Troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ValidationError: Login failed: 404 Not Found` on `OpenApi(api_key=...)` | `SWANLAB_API_HOST` points to a private cloud, or missing `/api` suffix | Unset the env vars and explicitly set `SWANLAB_API_HOST=https://api.swanlab.cn/api` |
| `get_metrics` returns `code=404, "No data found"` | Wrong key name (UI label != log key) | Probe with the cheat sheet in §5; remember verl prefixes (`actor/`, `critic/`, `response_length/...`) |
| `get_summary` returns `code=400, "Bad Request"` | Run is not a clone; `rootExpId`/`rootProId` are None | Don't use `get_summary` for non-cloned runs; just probe `get_metrics` |
| `OpenApi.login_info` is not callable | It's a property, not a method | Access as `api.login_info` (no `()`) |
| WebFetch returns "no data" / SPA shell | Chart page is Vue-rendered client-side | Use the SDK; do not scrape HTML |
| `swanlab verify` shows wrong host | `~/.bashrc` exports redirect SDK to a private cloud | Override env at script start, before `import swanlab` |
