import threading
import uuid
import time
import httpx
import json
import re
import yaml
import tempfile
import os
from urllib.parse import urlparse
from beast_logger import print_dict
from beast_logger import register_console
from typing import List, Tuple
from loguru import logger
from ajet.schema.task import WorkflowOutput, Task
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.sync_train_code import create_tracked_ajet_zip_from_dir
from ajet.tuner_lib.experimental.swarm_overwatch_utils import CurrentBatchRolloutPoolInformation
from ajet.tuner_lib.experimental.interchange_utils import (
    SyncTrainConfigRequest,
    ClaimEpisodeRequest,
    ClaimEpisodeResponse,
    CanContinueEpisodeRequest,
    CanContinueEpisodeResponse,
    EndEpisodeRequest,
    EndEpisodeResponse,
    EpisodeStatus,
    EpisodeBufferResponse,
    SwarmThrottlePolicy,
    AgreeSyncWeightRequest,
    BoolResponse,
)

# general http timeout
GENERAL_TIMEOUT = 30
# To prevent stale records from accumulating, do not need to be changed
CLEAN_RECORD_TIMEOUT = 10
START_EPISODE_RETRY_DELAY = 15
TROTTLE_EPISODE_RETRY_DELAY = 2
WAIT_MORE_AVAIL_EPISODE_RETRY_DELAY = 2
# agree_sync_weight retry policy. The call must succeed -- a dropped
# agreement can stall the trainer's stop condition. Retries cover both
# transport errors and server-side rejection (e.g. when a just-completed
# end_episode hasn't yet propagated to the server's active list).
AGREE_SYNC_WEIGHT_RETRY_DELAY = 2.0
DELAY_AFTER_AGREE_SYNC_WEIGHT = 30
ENGINE_STATUS_POLL_INTERVAL = 5


def _extract_local_swarm_port(server_url: str) -> int:
    parsed = urlparse(server_url if "://" in server_url else f"http://{server_url}")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            f"auto_start_swarm_server only supports local server_url, got {server_url!r}"
        )
    if parsed.port is None:
        raise ValueError(f"server_url must include an explicit port, got {server_url!r}")
    return parsed.port


def _auto_start_local_swarm_server(server_url: str):
    from ajet.utils.smart_daemon import LaunchCommandWhenAbsent

    swarm_port = _extract_local_swarm_port(server_url)
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            "ajet-swarm",
            "start",
            f"--swarm-port={swarm_port}",
        ],
        dir="./",
        tag=f"swarm_server_{swarm_port}",
    )
    companion.launch(
        launch_wait_time=60,
        success_std_string="Interchange server is running in blocking mode",
        env_dict=os.environ.copy(),
    )


def raise_for_status_with_detail(resp):
    """
    Raise an exception with detailed error information if the response indicates an error.

    Args:
        resp: The httpx response object to check.

    Raises:
        RuntimeError: If the response status code indicates an error.
    """
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Read response text first (can only read body once)
        response_text = resp.text
        try:
            # Try to parse as JSON
            error_detail = json.loads(response_text)
            logger.error(f"SwarmClient error {resp.status_code}: {error_detail}")
            raise RuntimeError(f"SwarmClient error {resp.status_code}: {error_detail}") from e
        except (json.JSONDecodeError, ValueError):
            # Failed to parse JSON response
            logger.error(f"SwarmClient error {resp.status_code} with non-JSON response: {response_text}")
            raise RuntimeError(f"SwarmClient error {resp.status_code} with non-JSON response: {response_text}") from e


class SwarmServerOfflineError(Exception): ...


class SwarmClientBase(object):
    """HTTP client plus a background thread that keeps engine status cached."""

    SLOW_POLL = 1.0
    FAST_POLL = 0.33
    FAST_POLL_WINDOW = 10.0
    REFRESH_TRIGGER_KEYWORDS = (
        "broken pipe", "disconnected", "connection reset",
        "connection closed", "connection aborted", "bad file descriptor",
    )
    # Force-refresh the http client after this many consecutive poll failures,
    # even if none of the error messages match REFRESH_TRIGGER_KEYWORDS. Covers
    # cases where httpx wedges in ways our keyword heuristic can't detect
    # (HTTP/2 protocol stalls, sticky timeouts, stale pools).
    POLL_FORCE_REFRESH_AFTER = 3

    def __init__(self, server_url: str, verbose: bool = True):
        """
        Initialize the SwarmClientBase.

        Args:
            server_url: The URL of the swarm server.
            verbose: If True, enable verbose logging output.
        """
        register_console()
        self.server_url = server_url
        self.verbose = verbose
        self.client_uuid = str(uuid.uuid4())
        # http client
        self._last_second_print_buffer: dict[str, float] = {}
        self._http_client_lock = threading.Lock()
        self._http_client = self._refresh_http_client()

        # engine-status cache (written by the poll thread; readers wait on _engine_status_ready)
        self._engine_status_cache: tuple[str, dict] | None = None
        self._engine_status_ready = threading.Event()
        self._engine_status_last_error_log_time = 0.0
        self._engine_status_poll_interval = self.SLOW_POLL
        # consecutive failures since the last successful poll. Used to force an
        # http client refresh when the keyword-based heuristic in
        # `_should_refresh_client_on_error` misses a wedged connection.
        self._engine_status_consecutive_failures = 0

        # fast-poll window: True for FAST_POLL_WINDOW seconds after each get_engine_status() call
        self._high_freq_update_status = False
        self._high_freq_update_expiry = 0.0

        # callbacks fired once per fresh transition into ENGINE.WEIGHT_SYNCING
        self._entering_weight_sync_callbacks: list = []
        self._last_observed_engine_status: str | None = None
        self._engine_status_callback_lock = threading.Lock()

        # background polling thread
        self._engine_status_poll_stop = threading.Event()
        self._engine_status_poll_thread = threading.Thread(
            target=self._engine_status_poll_loop,
            daemon=True,
            name=f"SwarmClient-EngineStatusPoll-{self.client_uuid[:8]}",
        )
        self._engine_status_poll_thread.start()

    # ---- logging ------------------------------------------------------

    def logger_info(self, message):
        """logger.info with 1s de-duplication to avoid flooding."""
        now = time.time()
        last = self._last_second_print_buffer.get(message)
        if last is not None and now - last < 1:
            return
        self._last_second_print_buffer[message] = now
        logger.info(message)
        # keep the dict small
        for k in [k for k, ts in self._last_second_print_buffer.items() if now - ts > 1]:
            del self._last_second_print_buffer[k]

    # ---- http client --------------------------------------------------

    def _refresh_http_client(self):
        """Close the existing http client and create a fresh one.

        HTTP/1.1 only on purpose: swarm endpoints are small, low-frequency
        polls/heartbeats, so multiplexing buys nothing — and HTTP/2 has a class
        of stall failures (flow-control deadlock, GOAWAY mishandling, ping
        timeouts, HPACK desync, server-restart-behind-LB) where the TCP
        connection stays "alive" but every stream hangs without ever raising
        connection-reset/broken-pipe. That makes the keyword-based refresh
        heuristic miss them. Plain HTTP/1.1 fails loudly on the same
        scenarios, which our refresh logic can detect.
        """
        with self._http_client_lock:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = httpx.Client(timeout=GENERAL_TIMEOUT, http2=False)
            logger.warning("swarm client httpx client refreshed.")
            return self._http_client

    def _should_refresh_client_on_error(self, error: Exception) -> bool:
        """
        Check if the HTTP client should be refreshed based on the error message.

        Args:
            error: The exception that occurred during an HTTP request.

        Returns:
            True if the error message contains keywords indicating a connection issue.
        """
        msg = str(error).lower()
        return any(k in msg for k in self.REFRESH_TRIGGER_KEYWORDS)

    # ---- weight-sync transition callbacks -----------------------------

    def add_entering_weight_sync_callback(self, callback):
        """Fire `callback()` once each time engine status enters ENGINE.WEIGHT_SYNCING."""
        with self._engine_status_callback_lock:
            self._entering_weight_sync_callbacks.append(callback)

    def _observe_engine_status(self, new_status: str):
        """
        Observe engine status changes and fire callbacks on transitions.

        Args:
            new_status: The new engine status string.
        """
        with self._engine_status_callback_lock:
            fresh_entry = (
                new_status == "ENGINE.WEIGHT_SYNCING"
                and self._last_observed_engine_status != "ENGINE.WEIGHT_SYNCING"
            )
            self._last_observed_engine_status = new_status
            callbacks = list(self._entering_weight_sync_callbacks) if fresh_entry else ()
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.exception(f"Error in entering_weight_sync callback: {e}")

    # ---- engine status: public reader + background poller -------------

    def get_engine_status(self) -> Tuple[str, dict]:
        """Return the latest cached (status, json). Extends the fast-poll window."""
        self._high_freq_update_status = True
        self._engine_status_poll_interval = self.FAST_POLL
        self._high_freq_update_expiry = time.time() + self.FAST_POLL_WINDOW

        if not self._engine_status_ready.is_set():
            self._engine_status_ready.wait(timeout=15)
        return self._engine_status_cache or ("ENGINE.CANNOT_CONNECT", {})

    def get_global_step(self) -> int:
        """Return the current global training step from the swarm server."""
        _, status_json = self.get_engine_status()
        return status_json.get("global_step", 0)

    def _engine_status_poll_loop(self):
        """Background thread: fetch engine status at _engine_status_poll_interval.

        Top-level try/except is a final safety net: if it dies the cache freezes
        forever and only a process restart recovers — exactly the failure mode
        this whole module is trying to prevent.
        """
        while not self._engine_status_poll_stop.is_set():
            try:
                if self._high_freq_update_status and time.time() >= self._high_freq_update_expiry:
                    self._high_freq_update_status = False
                    self._engine_status_poll_interval = self.SLOW_POLL
                self._poll_engine_status_once()
            except Exception as e:
                now = time.time()
                if now - self._engine_status_last_error_log_time > 30:
                    logger.exception(f"Unexpected error in engine_status poll loop (continuing): {e}")
                    self._engine_status_last_error_log_time = now
            self._engine_status_poll_stop.wait(self._engine_status_poll_interval)

    def _poll_engine_status_once(self):
        """Fetch engine status from the server once and update the cache."""
        try:
            resp = self._http_client.get(f"{self.server_url}/get_engine_status", timeout=10)
            raise_for_status_with_detail(resp)
            resp_json = resp.json()
            status = resp_json.get("engine_status", "unknown")
            if status == "unknown":
                logger.warning(f"get_engine_status: {resp_json}")
            self._engine_status_cache = (status, resp_json)
            self._engine_status_ready.set()
            self._engine_status_consecutive_failures = 0
            self._observe_engine_status(status)
        except Exception as e:
            self._engine_status_consecutive_failures += 1
            # Refresh on either: (a) a known-transient error pattern, or
            # (b) sustained failure even if the error doesn't match — httpx can
            # wedge in ways the keyword heuristic doesn't catch, and without
            # this the same broken connection keeps failing forever and the
            # cached status stays stale until the process is restarted.
            try:
                if (
                    self._should_refresh_client_on_error(e)
                    or self._engine_status_consecutive_failures >= self.POLL_FORCE_REFRESH_AFTER
                ):
                    self._refresh_http_client()
                    self._engine_status_consecutive_failures = 0
            except Exception as refresh_err:
                logger.error(f"engine_status poll: http client refresh failed: {refresh_err}")
            if self._engine_status_cache is None:
                # unblock waiters on the very first call when the server is unreachable
                self._engine_status_cache = ("ENGINE.CANNOT_CONNECT", {})
                self._engine_status_ready.set()
            now = time.time()
            if now - self._engine_status_last_error_log_time > 30:
                logger.error(
                    f"Error getting engine status in poll loop "
                    f"(consecutive failures: {self._engine_status_consecutive_failures}): {e}"
                )
                self._engine_status_last_error_log_time = now

    def _wait_until_status_change_to(self, desired_status=None, desired_status_list=None, verbose=True):
        """Block until engine status reaches desired_status or one of desired_status_list, reporting every 30s."""
        if desired_status is not None:
            desired_status_list = [desired_status]
        assert desired_status_list, "Must specify desired_status or non-empty desired_status_list"

        if verbose:
            self.logger_info(f"Polling engine status until {desired_status_list}...")

        start = time.time()
        last_report = start
        ever_see_non_offline_state = False

        while True:
            try:
                current_status, _ = self.get_engine_status()
                now = time.time()

                if current_status != "ENGINE.OFFLINE":
                    ever_see_non_offline_state = True

                if ever_see_non_offline_state and (current_status == "ENGINE.OFFLINE") and ("ENGINE.OFFLINE" not in desired_status_list):
                    raise SwarmServerOfflineError(f"Engine status is OFFLINE while waiting for {desired_status_list}. This may indicate an error in the engine. Please check the swarm server logs for details.")

                if current_status in desired_status_list:
                    if verbose:
                        self.logger_info(f"Engine status is {current_status}.")
                    return

                if verbose and (now - last_report >= 30):
                    self.logger_info(f"Current engine status (already waited {int(now - start)}s): {current_status}")
                    last_report = now

                time.sleep(ENGINE_STATUS_POLL_INTERVAL)
            except SwarmServerOfflineError:
                raise
            except Exception as e:
                logger.error(f"Error polling engine status: {e}")
                time.sleep(ENGINE_STATUS_POLL_INTERVAL)



class SwarmClient(SwarmClientBase):
    """HTTP client for interacting with the Swarm server for distributed RL training."""

    def __init__(
        self,
        server_url: str,
        verbose: bool = True,
        agentjet_job=None,
        auto_start_swarm_server: bool = False,
    ):
        """
        Initialize the SwarmClient.

        Args:
            server_url: The URL of the swarm server.
            verbose: If True, enable verbose logging output.
            agentjet_job: The training parameters.
            auto_start_swarm_server: If True, automatically start the swarm server.
        """
        if auto_start_swarm_server:
            _auto_start_local_swarm_server(server_url)
        super().__init__(server_url=server_url, verbose=verbose)
        self.previous_warning_time = 0
        self.record_episode_expire_time = {}
        self.auto_batching_tasks = []
        self._begin_episode_lock = threading.Lock()
        # record last registered AgentJetJob
        self._agentjet_job = agentjet_job
        # throttle
        self._recent_seen_tasks = []

    def _clean_up_expired_records(self):
        """Remove episode records that have expired beyond CLEAN_RECORD_TIMEOUT seconds."""
        current_time = time.time()
        expired_episodes = [
            episode_uuid for episode_uuid, expire_time in self.record_episode_expire_time.items()
            if expire_time < current_time - CLEAN_RECORD_TIMEOUT
        ]
        for episode_uuid in expired_episodes:
            self.record_episode_expire_time.pop(episode_uuid, None)
        return


    def _check_throttle_policy(self, throttle_policy: SwarmThrottlePolicy, pool_info: CurrentBatchRolloutPoolInformation) -> Tuple[bool, str]:
        """
        Check if the client should throttle based on the throttle policy.
        Returns: (should_throttle, reason)
        """
        assert throttle_policy is not None, "Throttle policy must be provided."

        if self._agentjet_job:
            # check and raise early errors when possible
            assert self._agentjet_job.swarm_mode_sample_collection_method == "rollout_until_finish_enough_tasks", \
                f"Current sample collection method ({self._agentjet_job.swarm_mode_sample_collection_method}) does not support throttle policy."

        # only_this_client_uuid = throttle_policy.throttle_method in ["Task_Ratio_Limit"]
        only_this_client_uuid = True

        current_task_id = throttle_policy.current_task_id
        if not current_task_id:
            raise RuntimeError("Task_Ratio_Limit requires current_task_id to be set.")

        # loop completed_tasks, count how many task show up (consider this uuid only if only_this_client_uuid is True)
        task_set = set()
        task_set_with_alien_client_uuid = set()
        task_episode_count = {}
        for task_id, client_uuid_list in pool_info.completed_tasks_client_uuids.items():
            for cuuid in client_uuid_list:
                task_episode_count[task_id] = task_episode_count.get(task_id, 0) + 1
                if cuuid != self.client_uuid:
                    task_set_with_alien_client_uuid.add(task_id)
                if (not only_this_client_uuid) or (cuuid == self.client_uuid):
                    task_set.add(task_id)
                    break

        # loop running episodes, count how many task show up (consider this uuid only if only_this_client_uuid is True)
        if pool_info.running_episode_details is not None:
            for episode_uuid, episode_detail in pool_info.running_episode_details.items():
                cuuid = episode_detail.get("client_uuid", "")
                task_id = episode_detail.get("optional_task_id", "")
                task_episode_count[task_id] = task_episode_count.get(task_id, 0) + 1
                if cuuid != self.client_uuid:
                    task_set_with_alien_client_uuid.add(task_id)
                if task_id and ((not only_this_client_uuid) or (cuuid == self.client_uuid)):
                    task_set.add(task_id)
        else:
            # no running episode
            # get the number of totally completed tasks (task_episode_count >= expected_num_repeat)
            total_completed_tasks = sum(1 for count in task_episode_count.values() if count >= throttle_policy.expected_num_repeat)
            if total_completed_tasks < throttle_policy.expected_batch_size:
                # logger.debug(f"Throttling check for task_id {current_task_id}: there are only {total_completed_tasks} completed tasks in the batch, which is below the expected_batch_size of {throttle_policy.expected_batch_size}. ")
                return False, ""

        if current_task_id in self._recent_seen_tasks:
            # logger.debug(f"This task is already seen before, not throttling. ")
            return False, ""

        if throttle_policy.current_task_id in task_set:
            # logger.debug(f"Throttling check for task_id {current_task_id}: already has the same task_id in the batch. ")
            return False, ""

        if throttle_policy.current_task_id in task_set_with_alien_client_uuid:
            # logger.debug(f"Throttling check for task_id {current_task_id}: already has the same task_id from other client_uuid in the batch. ")
            return False, ""

        # task_set - task_set_with_alien_client_uuid to get the number of unique tasks that are not from other client_uuid, which is the real number of unique tasks that may cause throttling for this task_id
        real_unique_task = task_set - task_set_with_alien_client_uuid
        n_unique_task = len(real_unique_task)

        # is above threshold?
        _max = throttle_policy.expected_batch_size * throttle_policy.ratio
        # logger.debug(f"Throttling: there are currently {n_unique_task} / (max: {_max}) unique tasks. ")
        if n_unique_task >= _max:
            reason = f"Throttling because there are already {n_unique_task} unique tasks in the batch, which meets/exceeds the threshold of {_max} for task_id {current_task_id}."
            return True, reason
        else:
            return False, ""

    def _remember_seen_task(self, task_id: str, batch_size, num_repeat):
        """
        Record a task_id as recently seen for throttle policy tracking.

        Args:
            task_id: The task ID to remember.
            batch_size: Expected batch size, used to calculate buffer limit.
            num_repeat: Expected number of repeats per task, used to calculate buffer limit.
        """
        MAX_SEEN_TASK_BUFFER_SIZE = batch_size*num_repeat*3  # keep buffer size manageable, can be tuned
        if task_id not in self._recent_seen_tasks:
            self._recent_seen_tasks.append(task_id)
            if len(self._recent_seen_tasks) > MAX_SEEN_TASK_BUFFER_SIZE:
                self._recent_seen_tasks = self._recent_seen_tasks[-MAX_SEEN_TASK_BUFFER_SIZE:]

    def _should_throttle(self, throttle_policy: SwarmThrottlePolicy, pool_info: CurrentBatchRolloutPoolInformation) -> bool:
        """
        Determine if the client should throttle based on the throttle policy.

        Args:
            throttle_policy: The throttle policy configuration.
            pool_info: Current batch rollout pool information from the server.

        Returns:
            True if the client should throttle and delay starting a new episode.
        """
        should_throttle, throttle_reason = self._check_throttle_policy(throttle_policy, pool_info)
        if not should_throttle:
            # direct start this episode
            self._remember_seen_task(throttle_policy.current_task_id, throttle_policy.expected_batch_size, throttle_policy.expected_num_repeat)
        return should_throttle

    def begin_episode(self, discard_episode_timeout=240, episode_type="train", throttle_policy: SwarmThrottlePolicy|None = None) -> Tuple[str, OpenaiBaseUrlAndApiKey]:
        """
        Block until an episode is claimed.
        Argument:
            - discard_episode_timeout: when an episode is **idle** (idle means no llm request) for X seconds, it will be terminated by swarm server **remotely**
            - episode_type:
                - train: data will be fed to training pipeline
                - eval: data will NOT be fed to training pipeline
            - throttle_policy:        when there are multiple clients running different tasks (e.g. math + coding), you may need to arrange the percentage of different tasks in each batch (e.g. 40% math + 60% coding).
                                      But of course, you can set up your own logic and ignore this argument, the choice is all yours.
        Return:
            (episode_uuid, openai_base_url, openai_api_key)
        """
        return self._begin_episode_auto_retry(discard_episode_timeout, episode_type, throttle_policy)

    def _begin_episode_auto_retry(self, discard_episode_timeout=240, episode_type="train", throttle_policy: SwarmThrottlePolicy|None = None) -> Tuple[str, OpenaiBaseUrlAndApiKey]:
        """
        Internal method to claim an episode with automatic retry logic.

        Args:
            discard_episode_timeout: Idle timeout in seconds before the server discards the episode.
            episode_type: Type of episode, either "train" or "eval".
            throttle_policy: Optional throttle policy for task distribution control.

        Returns:
            A tuple of (episode_uuid, OpenaiBaseUrlAndApiKey).

        Raises:
            SwarmServerOfflineError: If the server goes offline during the operation.
        """
        # max_episode_time: when an episode has **lasted** for more than X seconds, it will be terminated **locally** by client (call `end_episode` will be re-route to `abort_episode`)
        max_episode_time = 8*discard_episode_timeout
        status, status_json = self.get_engine_status()  # warm up connection and log the status
        if status not in ["ENGINE.ROLLING"]:
            self.logger_info(f"Engine status is {status}. Waiting until ENGINE.ROLLING...")
            self._wait_until_status_change_to(desired_status="ENGINE.ROLLING", verbose=False)

        retry_delay = 0

        while True:

            # if not first attempt, sleep for a while before retrying
            if retry_delay > 0:
                time.sleep(retry_delay)
                status, status_json = self.get_engine_status()  # warm up connection and log the status
                if status not in ["ENGINE.ROLLING"]:
                    self.logger_info(f"Engine status is {status}. Waiting until ENGINE.ROLLING...")
                    self._wait_until_status_change_to(desired_status="ENGINE.ROLLING", verbose=False)

            # when throttle_policy is set, acquire lock to prevent multiple threads from claiming episode at the same time and causing throttle policy to fail
            if throttle_policy is not None:
                self._begin_episode_lock.acquire()

            try:
                # Check throttle policy before claiming episode (only for train episodes)
                if (throttle_policy is not None) and (episode_type == "train"):
                    pool_info = self.get_rollout_stat()
                    should_throttle = self._should_throttle(throttle_policy, pool_info)
                    if should_throttle:
                        self.logger_info(f"Throttle policy is active, delaying episode ...")
                        retry_delay = TROTTLE_EPISODE_RETRY_DELAY
                        continue

                # connect remote server to claim an episode
                req_obj = ClaimEpisodeRequest(
                    client_uuid=self.client_uuid,
                    episode_type=episode_type,
                    discard_episode_timeout=discard_episode_timeout,
                    throttle_policy=throttle_policy
                )
                resp = self._http_client.post(
                    f"{self.server_url}/claim_episode",
                    json=req_obj.model_dump()
                )
                raise_for_status_with_detail(resp)
                data = ClaimEpisodeResponse.model_validate(resp.json())
                episode_uuid = data.episode_uuid
                self.record_episode_expire_time[episode_uuid] = time.time() + max_episode_time
                self._clean_up_expired_records()

                if data.success:
                    episode_uuid = data.episode_uuid
                    openai_base_url = data.openai_base_url
                    openai_api_key = data.openai_api_key

                    # force replace openai_base_url host with self.server_url
                    openai_base_url = re.sub(r'^https?://[^/]+', self.server_url, openai_base_url)

                    if self.verbose:
                        self.logger_info(f"Claimed episode {episode_uuid}, current global step: {status_json.get('global_step', 'unknown')}")
                    return episode_uuid, OpenaiBaseUrlAndApiKey(
                        base_url=openai_base_url,
                        api_key=openai_api_key,
                        episode_uuid=episode_uuid
                    )
                else:
                    need_snap_scenarios =[
                        "Engine is syncing weights",
                        "Engine is in post-rolling phase",
                    ]
                    need_wait_scenarios =[
                        "No available episodes to claim.",
                    ]
                    if any(scenario in data.fail_cause for scenario in need_snap_scenarios):
                        if time.time() - self.previous_warning_time > 60:
                            self.logger_info(f"{data.fail_cause}. Retrying ...")
                            self.previous_warning_time = time.time()
                        retry_delay = START_EPISODE_RETRY_DELAY
                        continue
                    elif any(scenario in data.fail_cause for scenario in need_wait_scenarios):
                        retry_delay = WAIT_MORE_AVAIL_EPISODE_RETRY_DELAY
                        continue
                    else:
                        logger.warning(f"Failed to claim episode: {data.fail_cause}. Retrying ...")
                        retry_delay = START_EPISODE_RETRY_DELAY
                        continue

            except SwarmServerOfflineError:
                # exit immediately without retrying when server is offline, to avoid flooding the logs with errors
                raise

            except Exception as e:

                if self._should_refresh_client_on_error(e):
                    self._refresh_http_client()
                logger.error(f"Error claiming episode: {e}. Retrying ...")
                retry_delay = START_EPISODE_RETRY_DELAY
                continue

            finally:
                if throttle_policy is not None:
                    if self._begin_episode_lock.locked():
                        self._begin_episode_lock.release()

    def end_episode(self, task:Task, episode_uuid: str, workflow_output: WorkflowOutput, declare_client_active: bool = True):
        """
        End an episode and submit the workflow output to the server.

        Args:
            task: The task associated with this episode.
            episode_uuid: The UUID of the episode to end.
            workflow_output: The workflow output containing reward and metadata.
            declare_client_active: If True, register this client as active on the server.
                This is only useful when you select `rollout_until_all_clients_agree_sync_weight`,
                because in this case the server has to know how many client nodes are active.

        Raises:
            RuntimeError: If the server fails to end the episode.
        """
        if not episode_uuid:
            logger.error("No episode to end.")
            return

        if episode_uuid in self.record_episode_expire_time:
            remain_time = self.record_episode_expire_time.pop(episode_uuid, 0) - time.time()
            if remain_time < 0:
                logger.warning(f"Episode {episode_uuid} has expired (expired {-remain_time} seconds ago). Please use a larger `discard_episode_timeout` when `begin_episode`. Skipping end_episode.")
                # send abort signal to server to clean up episode
                self.abort_episode(episode_uuid)
                return
        else:
            # send abort signal to server to clean up episode
            logger.warning(f"Episode {episode_uuid} has expired (expired at least {CLEAN_RECORD_TIMEOUT} seconds ago). Please use a larger `discard_episode_timeout` when `begin_episode`. Skipping end_episode.")
            self.abort_episode(episode_uuid)
            return

        task_id = task.task_id
        assert task_id, "task.task_id must be valid!"
        workflow_output.metadata["task_id"] = task_id
        req_obj = EndEpisodeRequest(
            client_uuid=self.client_uuid,
            episode_uuid=episode_uuid,
            workflow_output=workflow_output,
            task_id=task_id,
            declare_client_active=declare_client_active
        )

        resp = self._http_client.post(
            f"{self.server_url}/end_episode",
            json=req_obj.model_dump()
        )
        # Special handling: when engine is WEIGHT_SYNCING, just warn instead of raising error
        if resp.status_code == 400:
            try:
                error_detail = resp.json()
                if "WEIGHT_SYNCING" in str(error_detail.get("detail", "")):
                    logger.warning(f"Engine is in WEIGHT_SYNCING state, episode {episode_uuid} will be discarded. This is expected during weight sync.")
                    return
            except (json.JSONDecodeError, ValueError):
                pass
        raise_for_status_with_detail(resp)
        data = EndEpisodeResponse.model_validate(resp.json())

        if data.success:
            if self.verbose:
                self.logger_info(f"Ended episode {episode_uuid}")
        else:
            logger.error(f"Failed to end episode {episode_uuid}")
            raise RuntimeError(f"Failed to end episode {episode_uuid}")


    def abort_episode(self, episode_uuid: str, declare_client_active: bool = True):
        """
        Abort an episode without submitting a valid workflow output.

        Args:
            episode_uuid: The UUID of the episode to abort.
            declare_client_active: If True, register this client as active on the server.
                This is only useful when you select `rollout_until_all_clients_agree_sync_weight`,
                because in this case the server has to know how many client nodes are active.
        """
        if not episode_uuid:
            logger.error("No episode to end.")
            return

        try:
            workflow_output = WorkflowOutput(reward=0.0, metadata={})
            req_obj = EndEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid,
                workflow_output=workflow_output,
                task_id="",
                declare_client_active=declare_client_active
            )

            resp = self._http_client.post(
                f"{self.server_url}/abort_episode",
                json=req_obj.model_dump()
            )
            raise_for_status_with_detail(resp)
            data = EndEpisodeResponse.model_validate(resp.json())

            if data.success:
                if self.verbose:
                    self.logger_info(f"Aborted episode {episode_uuid}")
            else:
                logger.error(f"Failed to end episode {episode_uuid}")

        except Exception as e:
            if self._should_refresh_client_on_error(e):
                self._refresh_http_client()
            logger.error(f"Error ending episode: {e}")

    def sync_train_config(self, agentjet_job: AgentJetJob):
        """
        Sync training configuration to the Swarm server.
        This sends the AgentJetJob config as YAML to the remote server.
        """
        # try get init status
        current_status, _ = self.get_engine_status()
        self._agentjet_job = agentjet_job
        if current_status != "ENGINE.OFFLINE":
            raise RuntimeError(f"Cannot sync train config when engine is NOT ENGINE.OFFLINE. (current status: {current_status})")

        try:
            config_dict = agentjet_job.config.to_dict()
            yaml_str = yaml.safe_dump(config_dict, sort_keys=False)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_str)
                logger.warning(f"Sync new training configuration: {f.name}")
            req_obj = SyncTrainConfigRequest(yaml_as_string=yaml_str)

            resp = self._http_client.post(
                f"{self.server_url}/sync_train_config",
                json=req_obj.model_dump()
            )
            raise_for_status_with_detail(resp)
            self.logger_info("Synced train config to Swarm server")
        except Exception as e:
            logger.error(f"Error syncing train config: {e}")
            raise

    def sync_train_code(self, zip_file_path: str):
        """
        Sync an ajet/ source zip to the Swarm server.
        The zip must contain a top-level ajet/ directory.
        """
        current_status, _ = self.get_engine_status()
        if current_status != "ENGINE.OFFLINE":
            raise RuntimeError(
                "Cannot sync train code when engine is NOT ENGINE.OFFLINE. "
                f"(current status: {current_status})"
            )
        if not os.path.isfile(zip_file_path):
            raise FileNotFoundError(f"Training code zip file does not exist: {zip_file_path}")

        try:
            with open(zip_file_path, "rb") as f:
                resp = self._http_client.post(
                    f"{self.server_url}/sync_train_code",
                    content=f.read(),
                    headers={"Content-Type": "application/zip"},
                    timeout=600,
                )
            raise_for_status_with_detail(resp)
            result = resp.json()
            if not result.get("success"):
                raise RuntimeError(
                    result.get("message")
                    or result.get("error")
                    or "Failed to sync train code"
                )
            self.logger_info(
                f"Synced train code to Swarm server: {result.get('temp_ajet_code_path')}"
            )
            return result
        except Exception as e:
            logger.error(f"Error syncing train code: {e}")
            raise

    def sync_train_code_from_dir(self, directory_path: str):
        """
        Create a tracked-only ajet/ source zip from a directory, then sync it.
        The directory must contain an ajet/ folder and be inside a Git work tree.
        """
        zip_file_path, file_count = create_tracked_ajet_zip_from_dir(directory_path)
        logger.warning(f"Created tracked-only training code zip: {zip_file_path}")
        result = self.sync_train_code(zip_file_path)
        result["local_zip_file_path"] = zip_file_path
        result["local_zip_file_count"] = file_count
        return result

    def start_engine(self):
        """
        Start the training engine on the Swarm server.
        This triggers the server to begin the training process.
        Polls until engine status is "ENGINE.ROLLING".
        """
        # try get init status
        current_status, _ = self.get_engine_status()
        if current_status != "ENGINE.OFFLINE":
            raise RuntimeError(f"Cannot start engine when engine is NOT ENGINE.OFFLINE. (current status: {current_status})")

        # Send start engine request
        resp = self._http_client.post(
            f"{self.server_url}/start_engine",
            json={},
            timeout=600
        )
        raise_for_status_with_detail(resp)
        result = resp.json()
        if result.get("success"):
            self.logger_info("Successfully started training engine on Swarm server (current model global step)")
        else:
            logger.error("Failed to start training engine")
            raise RuntimeError("Failed to start training engine")

        # Poll until engine status is "ENGINE.ROLLING"
        self._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
        logger.success("Training engine is now ROLLING and ready.")


    def can_continue_episode(self, episode_uuid: str) -> bool:
        """
        Check if an episode can continue (still claimed and engine is rolling).

        Args:
            episode_uuid: The UUID of the episode to check.

        Returns:
            True if the episode can continue, False otherwise.
        """
        if not episode_uuid:
            return False
        try:
            req_obj = CanContinueEpisodeRequest(
                client_uuid=self.client_uuid,
                episode_uuid=episode_uuid
            )
            resp = self._http_client.post(
                f"{self.server_url}/can_continue_episode",
                json=req_obj.model_dump(),
                timeout=10
            )
            raise_for_status_with_detail(resp)
            data = CanContinueEpisodeResponse.model_validate(resp.json())
            return data.can_continue
        except Exception as e:
            if self._should_refresh_client_on_error(e):
                self._refresh_http_client()
            logger.error(f"Error checking can_continue_episode: {e}")
            return False

    def get_episode_buffer(self) -> List[EpisodeStatus]:
        """
        Get the current episode buffer from the server.

        Returns:
            A list of EpisodeStatus objects representing all active episodes.
        """
        try:
            resp = self._http_client.post(
                f"{self.server_url}/get_episode_buffer",
                json={},
                timeout=10
            )
            raise_for_status_with_detail(resp)
            data = EpisodeBufferResponse.model_validate(resp.json())
            return data.buffer
        except Exception as e:
            if self._should_refresh_client_on_error(e):
                self._refresh_http_client()
            logger.error(f"Error getting episode buffer: {e}")
            return []

    def auto_sync_train_config_and_start_engine(self, agentjet_job: AgentJetJob, force_restart=False, _retry_once=True) -> None:
        """
        Automatically sync training configuration and start the engine if needed.
        This checks the current engine status and performs actions accordingly.

        Args:
            - agentjet_job: The AgentJetJob configuration to sync.
            - force_restart: If True, forces a restart of the engine.
        """
        if force_restart:
            logger.warning("Force restarting the engine...")
            self.stop_engine()
            time.sleep(8)

        if agentjet_job.ensure_new_experiment and not force_restart:
            logger.warning("ensure_new_experiment is set to True, but force_restart is not set! Will still continue the experiment on the current model version!")
            time.sleep(8)

        logger.success(f"--------------------------------------------------------------------------------------------------")
        logger.success(f"Run `python -m ajet.launcher --swarm-overwatch={self.server_url}` to monitor the training process.")
        logger.success(f"--------------------------------------------------------------------------------------------------")

        current_status, _ = self.get_engine_status()
        if current_status == "ENGINE.OFFLINE":
            self.logger_info("Engine is OFFLINE. Syncing train config and starting engine...")
            self.sync_train_config(agentjet_job)
            self.start_engine()
        elif current_status == "ENGINE.ROLLING":
            self.logger_info("Engine is already ROLLING. No action needed.")
        elif current_status == "ENGINE.ROLLING_POST":
            self.logger_info("Engine is already ROLLING. No action needed.")
        elif current_status in ["ENGINE.CANNOT_CONNECT"]:
            logger.error("Unable to connect to swarm server.")
            if _retry_once:
                time.sleep(16)
                return self.auto_sync_train_config_and_start_engine(agentjet_job, force_restart=force_restart, _retry_once=False)
            raise RuntimeError(f"Unable to connect to swarm server.")
        elif current_status in ["ENGINE.BOOTING", "ENGINE.WEIGHT_SYNCING"]:
            self.logger_info(f"Engine is {current_status}. Waiting until it becomes ROLLING...")
            self._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
            logger.success("Training engine is now ROLLING and ready.")
        else:
            raise RuntimeError(f"Cannot sync train config or start engine when engine is in status: {current_status}")

    @staticmethod
    def async_and_start_multi_engine(client_job_pairs: List[Tuple["SwarmClient", AgentJetJob]], force_restart=False):
        """
        Run `auto_sync_train_config_and_start_engine` on multiple (client, job) pairs in parallel.
        """
        threads = []
        errors = []

        def _worker(client: "SwarmClient", job: AgentJetJob):
            try:
                client.auto_sync_train_config_and_start_engine(job, force_restart=force_restart)
            except Exception as e:
                logger.exception(f"Error starting engine on {client.server_url}: {e}")
                errors.append((client.server_url, e))

        for client, job in client_job_pairs:
            t = threading.Thread(target=_worker, args=(client, job), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Failed to start {len(errors)} engine(s): {errors}")

    def stop_engine(self):
        """
        Stop the training engine on the Swarm server.
        This triggers the server to stop the training process.
        """
        current_status, _ = self.get_engine_status()
        if current_status == "ENGINE.OFFLINE":
            self.logger_info("Engine is already OFFLINE. No action needed.")
            return

        resp = self._http_client.post(
            f"{self.server_url}/stop_engine",
            json={},
            timeout=600
        )
        raise_for_status_with_detail(resp)
        result = resp.json()
        if result and result.get("success"):
            self.logger_info("Successfully stopped training engine on Swarm server")
        else:
            logger.error("Failed to stop training engine")
            raise RuntimeError("Failed to stop training engine")
        self._wait_until_status_change_to(desired_status="ENGINE.OFFLINE")

    def server_experiment_dir(self) -> str:
        """
        Fetch the absolute experiment directory from the Swarm server.
        Returns None if the engine has not started yet (no experiment dir is set).
        """
        try:
            resp = self._http_client.get(
                f"{self.server_url}/get_server_experiment_dir",
                timeout=10
            )
            raise_for_status_with_detail(resp)
            return resp.json().get("server_experiment_dir", None)
        except Exception as e:
            return "saved_experiments"

    def agree_sync_weight(self) -> bool:
        """Notify the swarm server that this client agrees to a weight sync.

        The server only accepts the agreement if this client is in its
        active-client list (i.e. has end_episode'd at least one rewarded
        episode since the last sync). Used together with the
        `rollout_until_any_client_agree_sync_weight` /
        `rollout_until_all_clients_agree_sync_weight` stop conditions so the
        client can decide for itself when its current batch is "good enough".

        Important: this call retries indefinitely on rejection or error. A
        dropped agreement can stall the trainer (e.g. under "all clients
        agree"), and the most common rejection -- "client not yet in active
        list" -- clears itself once the just-finished end_episode propagates.
        The only early exit is if the engine has left ROLLING/ROLLING_POST,
        since the agreement would be wiped by the server-side reset anyway.

        Returns: True once the agreement was registered, False if the engine
            left rolling state before agreement.
        """

        assert self._agentjet_job, "Please call sync_train_config with a valid AgentJetJob before starting the engine."
        assert self._agentjet_job.swarm_mode_sample_collection_method in ["rollout_until_any_client_agree_sync_weight", "rollout_until_all_clients_agree_sync_weight"], \
            "agree_sync_weight is only applicable when swarm_mode_sample_collection_method is set to rollout_until_any_client_agree_sync_weight or rollout_until_all_clients_agree_sync_weight."

        while True:
            engine_status, _ = self.get_engine_status()
            if engine_status not in ("ENGINE.ROLLING", "ENGINE.ROLLING_POST"):
                logger.warning(
                    f"agree_sync_weight: engine is {engine_status}, abandoning "
                    f"agreement (would be reset by server-side cleanup anyway)."
                )
                return False
            try:
                req_obj = AgreeSyncWeightRequest(client_uuid=self.client_uuid)
                resp = self._http_client.post(
                    f"{self.server_url}/agree_sync_weight",
                    json=req_obj.model_dump(),
                    timeout=10,
                )
                raise_for_status_with_detail(resp)
                data = BoolResponse.model_validate(resp.json())
                if data.success:
                    if self.verbose:
                        self.logger_info("agree_sync_weight: registered with server")
                    self._wait_until_status_change_to(desired_status_list=["ENGINE.ROLLING_POST", "ENGINE.WEIGHT_SYNCING"])
                    return True
                logger.warning(f"agree_sync_weight rejected: {data.failure_reason}. Retrying in {AGREE_SYNC_WEIGHT_RETRY_DELAY}s...")
            except SwarmServerOfflineError:
                raise
            except Exception as e:
                if self._should_refresh_client_on_error(e):
                    self._refresh_http_client()
                logger.error(f"agree_sync_weight errored: {e}. Retrying in {AGREE_SYNC_WEIGHT_RETRY_DELAY}s...")
            time.sleep(AGREE_SYNC_WEIGHT_RETRY_DELAY)

    def get_rollout_stat(self) -> CurrentBatchRolloutPoolInformation:
        """
        Get the current batch rollout pool information from the Swarm server.
        Returns statistics about completed episodes, tasks, and progress.
        """
        try:
            resp = self._http_client.get(
                f"{self.server_url}/get_current_batch_rollout_pool_information",
                timeout=10
            )
            raise_for_status_with_detail(resp)
            data = CurrentBatchRolloutPoolInformation.model_validate(resp.json())
            return data
        except Exception as e:
            if self._should_refresh_client_on_error(e):
                self._refresh_http_client()
            logger.error(f"Error getting rollout statistics: {e}")
            return CurrentBatchRolloutPoolInformation()

    def print_rollout_stat(self):
        """
        Print the current batch rollout pool information in a human-readable format.
        """
        try:
            stat = self.get_rollout_stat().model_dump()
            completed_tasks_details = stat.pop("completed_tasks_details", None)
            episodes_per_task = []
            stat["average_episodes_per_task"] = 0
            task_buffer = ""
            for task_id, episode_list in completed_tasks_details.items():
                episodes_per_task += [len(episode_list)]
                task_buffer += f"Task-{task_id} ({len(episode_list)})  "
            stat["average_episodes_per_task"] = sum(episodes_per_task) / len(episodes_per_task) if episodes_per_task else 0.0
            stat = {
                "Completed tasks: (current) / (required)": f"{stat.get('completed_tasks', 0)} / {stat.get('completed_task_target', 0)}",
                "Completed episodes: (current) / (required)": f"{stat.get('completed_episodes', 0)} / {stat.get('completed_episode_target', 0)}",
                "Average episodes per task: (current) / (expected)": f"{stat.get('average_episodes_per_task', 0):.2f} / {stat.get('task_expected_num_repeat', 0)}",
                "Completed num-dummy tasks: (current) / (required)": f"{stat.get('completed_non_dummy_tasks', 0)} / {stat.get('completed_task_target', 0)}",
                "Tasks (Number of episodes completed for each task)": task_buffer,
                "Hint": f"Please run `ajet-swarm overwatch --swarm-url={self.server_url}` to get more details."
            }
            print_dict(stat, mod="console", header="Current Swarm Rollout Pool Information")
        except:
            pass


def auto_train_with_dataset(dataset, swarm_worker: SwarmClient, execute_agent, local_grpo_n=2, remote_batch_size=8):
    """
    Automatically train with a dataset using the swarm worker.

    Args:
        dataset: The dataset providing training tasks via generate_training_tasks().
        swarm_worker: The SwarmClient instance for communication with the server.
        execute_agent: A callable that executes the agent on a task and returns WorkflowOutput.
        local_grpo_n: Number of local GRPO repeats per task.
        remote_batch_size: Number of parallel remote workers.
    """
    from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

    def rollout(task) -> float | None:
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode()
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        # print global rollout status across the swarm
        swarm_worker.print_rollout_stat()
        return workflow_output.reward

    executor = PeriodicDrainThreadPoolExecutor(workers=remote_batch_size * local_grpo_n, max_parallel=64, auto_retry=True)
    for _, task in enumerate(dataset.generate_training_tasks()):
        for _ in range(local_grpo_n):
            executor.submit_with_periodic_drain(fn=rollout, task=task)
