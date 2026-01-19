# TinkerScript Design Blueprint

TinkerScript represents a client-server architecture designed to decouple the **Training Loop** (Server-side) from the **Rollout Execution** (Client-side). This allows for distributed, flexible, and potentially remote execution of agent rollouts (inference + reward calculation) while centralizing the model training and weight updates.

## 1. System Architecture

The system consists of three main components:

### A. TinkerScript Server (The Trainer)
*   **Role**: Manages the training lifecycle, generates tasks (episodes), serves the model (LLM) API, and updates model weights.
*   **Technology**: Python, FastAPI, ZeroMQ (IPC/TCP), Shared Memory (Multiprocessing).
*   **Location**: Runs on the GPU cluster/Training node.
*   **Key Functionality**:
    *   Maintains a queue of "Episodes" (training tasks).
    *   Exposes an HTTP API for external clients to claim tasks and submit results.
    *   Acts as a bridge between the HTTP world and the internal ZeroMQ-based training pipeline.

### B. TinkerScript Client (The User Script)
*   **Role**: Fetches tasks, runs the agent logic, computes rewards, and reports back.
*   **Technology**: Python (Requests/HTTPX).
*   **Location**: Can run locally, on a separate CPU cluster, or even a different cloud environment.
*   **Key Functionality**:
    *   Connects to the Server URL.
    *   Claims episodes via `begin_episode()`.
    *   Executes the agent logic (e.g., calling the LLM, running Python code).
    *   Calculates rewards (e.g., verifying math answers).
    *   Submits results via `end_episode()`.

### C. The LLM Serving Layer (Implicit)
*   The system provides an OpenAI-compatible API endpoint (`base_url`, `api_key`) to the client for LLM inference. This is likely hosted by the training system itself or a proxy, enabling the client to query the model being trained.

---

## 2. Detailed Workflow

### Step 1: Episode Generation & Registration (Server Side)
The training loop (e.g., RL algorithm like GRPO) generates a new task.
1.  An internal component registers a new episode via `register_episode`.
2.  The server stores this in `shared_mem_dict` with status `registered`.
3.  The episode is added to the `unclaimed_episodes` queue.
4.  The server sets up a ZeroMQ socket to listen for the result of this specific episode.

### Step 2: Task Claiming (Client Side)
The user's script calls `tinkerjet_remote.begin_episode()`.
1.  **Request**: `POST /claim_episode`
2.  **Server Logic**:
    *   Checks `unclaimed_episodes`.
    *   If available, pops one episode.
    *   Updates status to `claimed`.
    *   Records `client_uuid` and `latest_activity_timestamp`.
3.  **Response**: Returns `episode_uuid` and **OpenAI Credentials** (Base URL + API Key) specific to this session/model.

### Step 3: Rollout & Execution (Client Side)
The user's script (`execute_agent`) runs:
1.  Uses the provided OpenAI API to chat with the model (performing the actual inference step of the RL loop).
2.  Parses the model's output.
3.  Computes a reward (e.g., checking if `\boxed{answer}` matches ground truth).

### Step 4: Result Submission (Client Side)
The user's script calls `tinkerjet_remote.end_episode()`.
1.  **Request**: `POST /end_episode` with `workflow_output` (Reward + Metadata).
2.  **Server Logic**:
    *   Validates the episode exists and is claimed by this client.
    *   Connects to the internal ZeroMQ socket associated with this episode.
    *   Forwards the `workflow_output` payload into the ZeroMQ socket, effectively pushing it back into the training loop.
    *   Waits for an acknowledgment.
    *   Deletes the episode record from memory upon success.

### Step 5: Failure Recovery & Timeouts
*   **Crash Recovery**: If a client crashes after claiming a task, the server tracks `latest_activity_timestamp`.
*   **Requisition**: A background check (`find_claimed_episodes_that_need_to_be_unclaimed`) reverts "stale" claimed episodes back to `registered` status so other clients can pick them up.
*   **Weight Updates**: If the server moves to a weight update phase, it might verify if an episode is still valid via `can_continue_episode`.

---

## 3. Data Structures & API Design

### Episode Status Object
Stored in Server Shared Memory:
```python
class EpisodeStatus:
    episode_uuid: str              # Unique ID for the task
    client_uuid: str               # ID of the worker claiming it
    episode_status: str            # "registered", "claimed"
    openai_base_url: str           # Endpoint for the model
    openai_api_key: str            # Auth for the model
    zmq_listen_result_addr: str    # Internal address to forward results to
    latest_activity_timestamp: float
```

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/claim_episode` | Worker requests a job. Returns UUID + LLM credentials. |
| `POST` | `/end_episode` | Worker submits results (Reward). Completes the cycle. |
| `POST` | `/can_continue_episode` | Checks if the episode is still valid (e.g., weights haven't changed). |
| `POST` | `/register_episode` | (Internal/Debug) Adds a task to the queue. |
| `GET` | `/get_engine_status` | Returns system health/state (e.g., "booting", "ready"). |
| `POST` | `/sync_train_config` | Syncs configuration yaml (logging/debug). |

---

## 4. Key Configurations

From `ajet_tinkerscript_default.yaml`, we see how this mode is activated:

```yaml
experiment_dir: "auto"
enable_tinkerscript_mode: True  # Activates the HTTP API Server
interchange_server:
  interchange_method: 'ipc'     # Internal communication (ZeroMQ)
  interchange_server_port: 10086 # HTTP API Port
```

## 5. Benefits of this Design

1.  **Flexibility**: Users can write custom python logic for "Rollout" without modifying the core C++/Python training engine.
2.  **Distributed Generation**: You can have 1 training node and 1000 cheap CPU nodes just running the python script to generate data.
3.  **Complex Logic Support**: Since the rollout is just a client script, it can call external tools, Sandboxed code interpreters, or APIs (Google Search) easily before calculating the reward.
