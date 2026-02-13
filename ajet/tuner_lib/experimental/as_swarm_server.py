import multiprocessing
import time
import zmq
import os
import asyncio
import threading
from loguru import logger
from functools import lru_cache
from types import SimpleNamespace
from fastapi import FastAPI, HTTPException
from multiprocessing.managers import DictProxy
from typing import Coroutine, Optional, Tuple, List
from ajet.utils.process_killer import kill_process_tree
from ajet.tuner_lib.experimental.swarm_overwatch_utils import CurrentBatchRolloutPoolInformation
from ajet.tuner_lib.experimental.interchange_utils import DEBUG, VERBOSE
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
    BoolResponse,
    RegisterEpisodeRequest,
    UpdateEngineStatusRequest,
    VALID_STATUSES,
)

RCVTIMEO = 2 * 1000
RCVTIMEO_OUT = 300 * 1000
RCVTIMEO_WAIT_N = RCVTIMEO_OUT // RCVTIMEO


def is_key_epsisode_status(key: str) -> bool:
    return key.startswith("episodes-")


@lru_cache(maxsize=128)
def ep_key(episode_uuid: str) -> str:
    return f"episodes-{episode_uuid}"


def register_enable_swarm_mode_routes(
    app,
    zmq_context,
    shared_mem_dict: DictProxy,
    shared_mem_dict_lock: threading.Lock,
) -> Tuple[FastAPI, Optional[Coroutine]]:
    if "episodes" not in shared_mem_dict:
        shared_mem_dict["episodes"] = {}

    if "unclaimed_episodes" not in shared_mem_dict:
        shared_mem_dict["unclaimed_episodes"] = []

    if "current_batch_rollout_pool_information" not in shared_mem_dict:
        shared_mem_dict["current_batch_rollout_pool_information"] = CurrentBatchRolloutPoolInformation()

    # ------------------------------------------------------------------------------------------------
    # ------ Recycle claimed episodes that client failed to complete in (promised) time --------------
    # ---------------------------------  claimed -> unclaimed ----------------------------------------
    # ------------------------------------------------------------------------------------------------

    async def find_claimed_episodes_that_need_to_be_unclaimed() -> List[str]:
        to_unclaim_episodes = []
        current_time = time.time()

        for k, v in shared_mem_dict.items():
            if is_key_epsisode_status(k):
                es: EpisodeStatus = v
                if es.episode_status == "claimed":
                    if (current_time - es.latest_activity_timestamp) > es.discard_episode_timeout:
                        to_unclaim_episodes.append(es.episode_uuid)

        for episode_uuid in to_unclaim_episodes:
            await _revert_episode_to_unclaimed(episode_uuid, shared_mem_dict, shared_mem_dict_lock)

        return to_unclaim_episodes

    def _context_tracker_reset_blocking(episode_uuid, shared_mem_dict):  # must async
        # send message to context tracker
        assert "episodes" in shared_mem_dict
        zmq_addr = shared_mem_dict[ep_key(episode_uuid)].zmq_listen_result_addr
        socket = zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, RCVTIMEO)  # 2 seconds recv timeout
        socket.connect(zmq_addr)

        # <send to>
        #   <to_sourcefile>: ajet/task_runner/swarm_runner.py
        #   <to_code>: message = zmq_socket.recv_string()
        socket.send_string("RUNNER.SPECIAL.RESET_CONTEXT_TRACKER")

        # <wait for ack>
        for _ in range(RCVTIMEO_WAIT_N):  # max 5 minutes wait
            try:
                if DEBUG:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string begin.")
                # <wait for>:
                #   <from_sourcefile>: ajet/task_runner/swarm_runner.py
                #   <from_code>: zmq_socket.send_string("ack")
                #   <expect>: "ack"
                socket.recv_string()
                break
            except zmq.Again as e:
                if DEBUG:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string timeout, retrying.")

                if shared_mem_dict["engine_status"] not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | Engine is no longer rolling, aborting wait for ack.")
                    raise RuntimeError("Engine is no longer rolling, aborting wait for ack.")
                continue

    async def _revert_episode_to_unclaimed(episode_uuid: str, shared_mem_dict, shared_mem_dict_lock):
        # check status again, because other thread may have changed it
        if shared_mem_dict[ep_key(episode_uuid)].episode_status != "claimed":
            if episode_uuid in shared_mem_dict["unclaimed_episodes"]:
                pass
            else:
                shared_mem_dict["unclaimed_episodes"] += [episode_uuid]
            return

        # reset context tracker
        # _context_tracker_reset_blocking(episode_uuid, shared_mem_dict)   # must async
        await asyncio.to_thread(_context_tracker_reset_blocking, episode_uuid, shared_mem_dict)

        # revert
        logger.warning(f"Reverting episode {episode_uuid} to unclaimed due to client timeout.")
        if ep_key(episode_uuid) in shared_mem_dict:
            es: EpisodeStatus = shared_mem_dict[ep_key(episode_uuid)]
            es.episode_status = "registered"
            es.client_uuid = ""
            es.latest_activity_timestamp = time.time()
            es.discard_episode_timeout = -1
            with shared_mem_dict_lock:
                shared_mem_dict[ep_key(episode_uuid)] = es
                if episode_uuid in shared_mem_dict["unclaimed_episodes"]:
                    pass
                else:
                    shared_mem_dict["unclaimed_episodes"] += [episode_uuid]

    def _delete_episode_record(episode_uuid: str, shared_mem_dict, shared_mem_dict_lock):
        with shared_mem_dict_lock:
            # remove episode record
            if ep_key(episode_uuid) in shared_mem_dict:
                del shared_mem_dict[ep_key(episode_uuid)]  # RM--
                logger.info(f"Deleted episode record for {episode_uuid}.")
            # remove from unclaimed list if present
            if episode_uuid in shared_mem_dict["unclaimed_episodes"]:
                shared_mem_dict["unclaimed_episodes"].remove(episode_uuid)

    # --------------------------------------------------------------------------------------
    # -------------------------- return workflow output ------------------------------------
    # --------------------------------------------------------------------------------------

    def _register_final_episode_output_blocking(episode_uuid, workflow_output, shared_mem_dict, shared_mem_dict_lock):  # must async
        # begin send workflow_output
        zmq_addr = shared_mem_dict[ep_key(episode_uuid)].zmq_listen_result_addr
        if DEBUG:
            logger.info(f"[server] episode_uuid: {episode_uuid} | Received new chat completion request")
        socket = zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, RCVTIMEO)  # 2 seconds recv timeout
        socket.connect(zmq_addr)
        if DEBUG:
            logger.info(f"[server] episode_uuid: {episode_uuid} | connect done")
        socket.send_string(workflow_output.model_dump_json())
        if DEBUG:
            logger.info(f"[server] episode_uuid: {episode_uuid} | send_string")
        # wait for ack
        for _ in range(RCVTIMEO_WAIT_N):  # max 5 minutes wait
            try:
                if DEBUG:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string begin.")
                # <wait for>:
                #   <from_sourcefile>: ajet/task_runner/swarm_runner.py
                #   <from_code>: zmq_socket.send_string("ack")
                #   <expect>: "ack"
                socket.recv_string()
                break
            except zmq.Again as e:
                if DEBUG:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string timeout, retrying.")
                if shared_mem_dict["engine_status"] not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
                    logger.info(f"[server] episode_uuid: {episode_uuid} | Engine is no longer rolling, aborting wait for ack.")
                    # raise RuntimeError("Engine is no longer rolling, aborting wait for ack.")
                    break
                continue
        # clean up episode records
        with shared_mem_dict_lock:
            del shared_mem_dict[ep_key(episode_uuid)]
            if episode_uuid in shared_mem_dict["unclaimed_episodes"]:
                shared_mem_dict["unclaimed_episodes"].remove(episode_uuid)

    # --------------------------------------------------------------------------------------
    # -------------------------- status monitor --------------------------------------------
    # --------------------------------------------------------------------------------------

    async def register_episode_ready_listener():
        while True:
            await asyncio.sleep(10)  # check every 10 seconds
            await find_claimed_episodes_that_need_to_be_unclaimed()
            # read_all_episode_status()
            if DEBUG:
                _write_swarm_server_dynamic_log(shared_mem_dict)

    def read_all_episode_status() -> Optional[EpisodeStatus]:
        group_by_status = {}

        for k, v in shared_mem_dict.items():
            if is_key_epsisode_status(k):
                es: EpisodeStatus = v
                if es.episode_status not in group_by_status:
                    group_by_status[es.episode_status] = []
                group_by_status[es.episode_status].append(es)

        print_buffer_str = f"Registered: {len(group_by_status.get('registered', []))}, Claimed: {len(group_by_status.get('claimed', []))}"
        logger.info(f"Current engine status: [{shared_mem_dict['engine_status']}], " + print_buffer_str)

        return None

    def _write_swarm_server_dynamic_log(shared_mem_dict):
        if DEBUG:
            fp = "./swarm_server.dynamic.log"
            string_buffer = ""

            for k, v in shared_mem_dict.items():
                if is_key_epsisode_status(k):
                    es: EpisodeStatus = v
                    p = es.model_dump_json()
                    string_buffer += f"{p}\n"

            with open(fp, "w") as f:
                f.write(string_buffer)
        return

    # --------------------------------------------------------------------------------------
    # -------------------------- engine status op ------------------------------------------
    # --------------------------------------------------------------------------------------
    shared_mem_dict["engine_status"] = "ENGINE.OFFLINE"  # initial status
    def _clean_up_engine_status(shared_mem_dict_lock, shared_mem_dict):
        with shared_mem_dict_lock:
            episode_keys = [k for k in shared_mem_dict.keys() if is_key_epsisode_status(k)]
            # remove all episodes
            for key in episode_keys:
                del shared_mem_dict[key]
                logger.info(f"[_clean_up_engine_status] Removed episode: {key}")

            # clear unclaimed episodes list
            if "unclaimed_episodes" in shared_mem_dict:
                num_unclaimed = len(shared_mem_dict["unclaimed_episodes"])
                shared_mem_dict["unclaimed_episodes"] = []
                logger.info(f"[_clean_up_engine_status] Cleared {num_unclaimed} unclaimed episodes")

    # --------------------------------------------------------------------------------------
    # -------------------------- fastapi routes --------------------------------------------
    # --------------------------------------------------------------------------------------

    @app.post("/sync_train_config")
    async def sync_train_config(req: SyncTrainConfigRequest):
        """
        Receive training configuration from client as YAML string.
        Store it in shared memory for later use by start_engine.
        """
        if VERBOSE:
            logger.info(f"Running: /sync_train_config")

        if shared_mem_dict["engine_status"] != "ENGINE.OFFLINE":
            raise HTTPException(
                status_code=400,
                detail="Engine is already started. Call `stop_engine` first before syncing new training configuration.",
            )

        try:
            yaml_str = req.yaml_as_string
            logger.info("[sync_train_config] Received training configuration")
            if DEBUG:
                logger.debug(f"[sync_train_config] YAML content:\n{yaml_str}...")

            # Store the YAML config in shared memory for start_engine to use
            with shared_mem_dict_lock:
                shared_mem_dict["train_config_yaml"] = yaml_str

            logger.info("[sync_train_config] Successfully stored training configuration")
            return {"success": True}
        except Exception as e:
            logger.error(f"[sync_train_config] Error: {e}")
            return {"success": False, "error": str(e)}

    @app.post("/start_engine")
    async def start_engine():
        """
        Start the training engine using the previously synced configuration.
        This creates a temporary YAML file and spawns a training process.
        """
        if VERBOSE:
            logger.info(f"Running: /start_engine")
        try:
            import ray
            import tempfile
            import yaml as yaml_module
            from ajet.utils.launch_utils import execute_training_process
            from ajet.utils.config_utils import prepare_experiment_config
            from ajet.launcher import get_backbone_target, setup_environment_vars

            # Check if config has been synced
            if "train_config_yaml" not in shared_mem_dict:
                logger.error("[start_engine] No training config found. Please call sync_train_config first.")
                return {"success": False, "error": "No training config found"}
            with shared_mem_dict_lock:
                shared_mem_dict["engine_status"] = "ENGINE.BOOTING"
                shared_mem_dict["booting_start_time"] = time.time()
            # Parse YAML to get backbone
            yaml_str = shared_mem_dict["train_config_yaml"]
            config_dict = yaml_module.safe_load(yaml_str)
            backbone = config_dict.get("ajet", {}).get("backbone", "verl")
            DEFAULT_DIR = "saved_experiments"
            exp_dir_final = config_dict.get("ajet", {}).get("experiment_dir", DEFAULT_DIR)
            if exp_dir_final != DEFAULT_DIR:
                # remove last dir level if possible
                exp_dir_final = os.path.dirname(exp_dir_final)


            # Save YAML to temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
                temp_file.write(yaml_str)
                main_yaml_fp = temp_file.name
            logger.info(f"[start_engine] Saved config to temporary file: {main_yaml_fp}")

            # Create args namespace
            args = SimpleNamespace(
                conf=main_yaml_fp,
                backbone=backbone,
                exp_dir=exp_dir_final,
                with_logview=False,
                debug=False,
            )
            # get debug param
            should_debug = os.environ.get("RAY_DEBUG_POST_MORTEM", "0") == "1"
            debug_tags = os.environ.get("DEBUG_TAGS", "")
            if should_debug:
                args.debug = debug_tags

            def override_param_callback(config):
                config["ajet"]["interchange_server"]["already_started"] = True
                config["ajet"]["interchange_server"]["interchange_server_port"] = int(os.getenv("AJET_DAT_INTERCHANGE_PORT"))  # type: ignore
                return config

            # Finalize experiment config
            main_yaml_fp, exe_exp_base, exp_name, exp_config = prepare_experiment_config(main_yaml_fp, exp_dir_final, backbone, override_param_callback)

            # Setup environment variables
            env, exp_config = setup_environment_vars(args, exp_config, main_yaml_fp)

            # Start ray if not already started
            if not ray.is_initialized():
                from ajet.utils.launch_utils import start_ray_service

                logger.info("[start_engine] Starting Ray service...")
                start_ray_service(args, env)
            else:
                logger.info("[start_engine] Ray already initialized")

            # Start training process in a separate process
            p = multiprocessing.Process(
                target=execute_training_process,
                args=(
                    args,
                    get_backbone_target(args.backbone),
                    main_yaml_fp,
                    exe_exp_base,
                    main_yaml_fp,
                    env,
                    exp_config,
                ),
            )
            p.daemon = True
            p.start()

            # wait until p.pid is available
            while not isinstance(p.pid, int):
                time.sleep(1)

            # set new process group
            os.setpgid(p.pid, p.pid)

            # Store process info in shared memory
            _clean_up_engine_status(shared_mem_dict_lock, shared_mem_dict)
            with shared_mem_dict_lock:
                shared_mem_dict["training_process_pid"] = p.pid
                shared_mem_dict["engine_status"] = "ENGINE.BOOTING"
                shared_mem_dict["booting_start_time"] = time.time()

            logger.info(f"[start_engine] Successfully started training process (PID: {p.pid})")
            return {"success": True, "pid": p.pid}

        except Exception as e:
            logger.error(f"[start_engine] Error starting engine: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    @app.post("/update_engine_status", response_model=BoolResponse)
    async def update_engine_status(req: UpdateEngineStatusRequest):
        """Update the current engine status."""
        if VERBOSE:
            logger.info(f"Running /update_engine_status")
        if req.engine_status not in VALID_STATUSES:
            return BoolResponse(success=False, failure_reason="Invalid engine status")
        previous_status = shared_mem_dict["engine_status"]
        shared_mem_dict["engine_status"] = req.engine_status
        if previous_status in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"] and req.engine_status not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            _clean_up_engine_status(shared_mem_dict_lock, shared_mem_dict)

        # Clear booting_start_time when transitioning away from BOOTING
        if previous_status == "ENGINE.BOOTING" and req.engine_status != "ENGINE.BOOTING":
            shared_mem_dict["booting_start_time"] = None

        engine_status_detail = req.engine_status_detail
        global_step = req.global_step
        if global_step is not None:
            shared_mem_dict["global_step"] = global_step
        if engine_status_detail is not None:
            shared_mem_dict["engine_status_detail"] = engine_status_detail
        logger.info(f"[update_engine_status] Engine status set to {req.engine_status}")
        return BoolResponse(success=True)

    @app.get("/get_engine_status")
    async def get_engine_status():
        """Get the current engine status."""
        status = shared_mem_dict["engine_status"]
        engine_status_detail = shared_mem_dict.get("engine_status_detail", None)
        global_step = shared_mem_dict.get("global_step", None)
        return {
            "engine_status": status,
            "engine_status_detail": engine_status_detail,
            "global_step": global_step,
        }

    # --- episode status ---
    @app.post("/register_episode", response_model=BoolResponse)
    async def register_episode(req: RegisterEpisodeRequest):
        """(From task_runner) Register a new episode as ready to roll."""
        engine_status = shared_mem_dict["engine_status"]
        if engine_status not in ["ENGINE.ROLLING"]:
            return BoolResponse(
                success=False,
                failure_reason=f"Engine is not in rolling state. Cannot register episode.",
            )

        episode_uuid = req.episode_uuid
        if VERBOSE: logger.info(f"Running [{episode_uuid}]: /register_episode")

        es = EpisodeStatus(
            episode_uuid=req.episode_uuid,
            openai_base_url=req.openai_base_url,
            openai_api_key=req.openai_api_key,
            episode_status="registered",
            zmq_listen_result_addr=req.zmq_listen_result_addr,
            discard_episode_timeout=-1,
        )
        es.latest_activity_timestamp = time.time()

        with shared_mem_dict_lock:
            shared_mem_dict[ep_key(episode_uuid)] = es
            shared_mem_dict["unclaimed_episodes"] += [req.episode_uuid]

        return BoolResponse(success=True)

    @app.post("/claim_episode", response_model=ClaimEpisodeResponse)
    async def claim_episode(req: ClaimEpisodeRequest):
        """(From client) Claim an available episode to rollout."""
        # find_claimed_episodes_that_need_to_be_unclaimed()

        engine_status = shared_mem_dict["engine_status"]

        if engine_status != "ENGINE.ROLLING":
            fail_cause = f"Engine not ready. Current status: [{engine_status}]."
            advise = ""
            if engine_status == "ENGINE.OFFLINE":
                advise = "Please start the engine first. Please use one of the client to run `client.sync_train_config() + client.start_engine()` to start the engine."
            elif engine_status == "ENGINE.BOOTING":
                advise = "Please wait until the engine is fully booted. Try again (maybe 1 minute) later."
            elif engine_status == "ENGINE.WEIGHT_SYNCING":
                advise = "Engine is syncing weights. Try again (maybe 1 minute) later."
            elif engine_status == "ENGINE.WEIGHT_EXPORTING":
                advise = "Engine is exporting weights (fsdp -> hf safetensor). Try again (maybe 1 minute) later."
            elif engine_status == "ENGINE.ROLLING_POST":
                advise = "Engine is in post-rolling phase. Try again (maybe 1 minute) later."
            return ClaimEpisodeResponse(
                success=False,
                client_uuid=req.client_uuid,
                episode_uuid="",
                openai_base_url="",
                openai_api_key="",
                fail_cause=fail_cause + " " + advise,
            )

        if req.episode_type == "train" or req.episode_type == "eval":
            with shared_mem_dict_lock:
                if len(shared_mem_dict["unclaimed_episodes"]) <= 0:
                    return ClaimEpisodeResponse(
                        success=False,
                        client_uuid=req.client_uuid,
                        episode_uuid="",
                        openai_base_url="",
                        openai_api_key="",
                        fail_cause="No available episodes to claim. Try again (maybe 1 minute) later.",
                    )

                # Hint: do NOT optimize these two lines
                episode_uuid = shared_mem_dict["unclaimed_episodes"][0]
                shared_mem_dict["unclaimed_episodes"] = shared_mem_dict["unclaimed_episodes"][1:]

                # get episode
                es: EpisodeStatus = shared_mem_dict[ep_key(episode_uuid)]
                es.episode_status = "claimed"
                es.episode_type = req.episode_type
                es.client_uuid = req.client_uuid
                es.latest_activity_timestamp = time.time()
                es.discard_episode_timeout = req.discard_episode_timeout

                shared_mem_dict[ep_key(episode_uuid)] = es
                openai_base_url = es.openai_base_url
                openai_api_key = es.openai_api_key

            if VERBOSE:
                logger.info(f"Running [{episode_uuid}]: /claim_episode")

            return ClaimEpisodeResponse(
                success=True,
                client_uuid=req.client_uuid,
                episode_uuid=episode_uuid,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                fail_cause="",
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown episode_type: {req.episode_type}")

    @app.post("/end_episode", response_model=EndEpisodeResponse)
    async def end_episode(req: EndEpisodeRequest):
        engine_status = shared_mem_dict["engine_status"]
        if engine_status not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            raise HTTPException(
                status_code=400,
                detail=f"Engine is not in rolling state. Current status: [{engine_status}]. Cannot end episode.",
            )

        # receive workflow output data
        client_uuid = req.client_uuid
        episode_uuid = req.episode_uuid
        workflow_output = req.workflow_output
        task_id = req.task_id

        if VERBOSE:
            logger.info(f"Running [{episode_uuid}]: /end_episode")

        assert "task_id" in workflow_output.metadata, "workflow_output.metadata must contain task_id"
        assert workflow_output.metadata["task_id"] == task_id, "workflow_output.metadata.task_id must match req.task_id"

        if "episodes" not in shared_mem_dict:
            logger.error(f"[server] No episodes registered yet.")
            raise HTTPException(status_code=400, detail=f"No episodes registered yet.")

        if (ep_key(episode_uuid)) not in shared_mem_dict:
            logger.error(f"[server] Episode {episode_uuid} not found.")
            raise HTTPException(status_code=400, detail=f"Episode {episode_uuid} not found.")

        # send workflow_output to zmq
        assert "episodes" in shared_mem_dict
        ep_stat = shared_mem_dict[ep_key(episode_uuid)]
        episode_type = ep_stat.episode_type
        episode_status = ep_stat.episode_status
        client_uuid_recorded = ep_stat.client_uuid

        if episode_status != "claimed":
            logger.error(f"[server] Episode {episode_uuid} is not in claimed status.")
            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_uuid} is not in claimed status, maybe you take **too long** to submit the workflow output, try increase `discard_episode_timeout` when `begin_episode`.",
            )

        if client_uuid_recorded != client_uuid:
            logger.error(f"[server] Episode {episode_uuid} is claimed by different client: {client_uuid_recorded}, but got {client_uuid}.")
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_uuid} is claimed by different client: {client_uuid_recorded}, but got {client_uuid}.",
            )

        if episode_type == "train":
            await asyncio.to_thread(
                _register_final_episode_output_blocking,
                episode_uuid,
                workflow_output,
                shared_mem_dict,
                shared_mem_dict_lock,
            )

        elif episode_type == "eval":
            if engine_status in ["ENGINE.ROLLING"]:
                await _revert_episode_to_unclaimed(episode_uuid, shared_mem_dict, shared_mem_dict_lock)
            else:
                _delete_episode_record(episode_uuid, shared_mem_dict, shared_mem_dict_lock)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown episode_type: {episode_type}")

        # return success
        return EndEpisodeResponse(success=True)

    @app.post("/abort_episode", response_model=EndEpisodeResponse)
    async def abort_episode(req: EndEpisodeRequest):
        engine_status = shared_mem_dict["engine_status"]
        if engine_status not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            return EndEpisodeResponse(success=True)

        # receive workflow output data
        episode_uuid = req.episode_uuid
        workflow_output = req.workflow_output
        task_id = req.task_id

        if VERBOSE:
            logger.info(f"Running [{episode_uuid}]: /abort_episode")

        # assert "task_id" in workflow_output.metadata, "workflow_output.metadata must contain task_id"
        # assert workflow_output.metadata["task_id"] == task_id, "workflow_output.metadata.task_id must match req.task_id"

        if "episodes" not in shared_mem_dict:
            logger.error(f"[server] No episodes registered yet.")
            return EndEpisodeResponse(success=True)

        if (ep_key(episode_uuid)) not in shared_mem_dict:
            logger.error(f"[server] Episode {episode_uuid} not found.")
            return EndEpisodeResponse(success=True)

        if engine_status in ["ENGINE.ROLLING"]:
            await _revert_episode_to_unclaimed(episode_uuid, shared_mem_dict, shared_mem_dict_lock)
        else:
            _delete_episode_record(episode_uuid, shared_mem_dict, shared_mem_dict_lock)

        return EndEpisodeResponse(success=True)

    @app.post("/can_continue_episode", response_model=CanContinueEpisodeResponse)
    async def can_continue_episode(req: CanContinueEpisodeRequest):
        engine_status = shared_mem_dict["engine_status"]
        if engine_status not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            return CanContinueEpisodeResponse(can_continue=False)

        can_continue = ep_key(req.episode_uuid) in shared_mem_dict
        can_continue = can_continue and shared_mem_dict[ep_key(req.episode_uuid)].episode_status == "claimed"

        return CanContinueEpisodeResponse(can_continue=can_continue)

    @app.post("/is_episode_claimed", response_model=BoolResponse)
    async def is_episode_claimed(req: CanContinueEpisodeRequest):
        engine_status = shared_mem_dict["engine_status"]
        if engine_status not in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]:
            return BoolResponse(success=False)
        if ep_key(req.episode_uuid) not in shared_mem_dict:
            return BoolResponse(success=False)
        es = shared_mem_dict[ep_key(req.episode_uuid)]
        if not es:
            return BoolResponse(success=False)
        if es.episode_status == "claimed":
            return BoolResponse(success=True)
        else:
            return BoolResponse(success=False)

    @app.post("/get_episode_buffer", response_model=EpisodeBufferResponse)
    async def get_episode_buffer():
        result = [v for k, v in shared_mem_dict.items() if is_key_epsisode_status(k)]
        return EpisodeBufferResponse(buffer=result)

    @app.post("/update_current_batch_rollout_pool_information", response_model=BoolResponse)
    async def update_current_batch_rollout_pool_information(req: CurrentBatchRolloutPoolInformation):
        """Update the current batch rollout pool information."""
        if VERBOSE:
            logger.info(f"Running /update_current_batch_rollout_pool_information")
        try:
            with shared_mem_dict_lock:
                # Ignore fields that are only maintained in shared_mem_dict
                req.running_episode_details = None
                req.engine_status = None
                req.global_step = None
                shared_mem_dict["current_batch_rollout_pool_information"] = req
            return BoolResponse(success=True)
        except Exception as e:
            logger.error(f"Error updating current batch rollout pool information: {e}")
            return BoolResponse(success=False, failure_reason=str(e))

    @app.get("/get_current_batch_rollout_pool_information", response_model=CurrentBatchRolloutPoolInformation)
    async def get_current_batch_rollout_pool_information():
        """Get the current batch rollout pool information."""
        try:
            pool_info = shared_mem_dict.get(
                "current_batch_rollout_pool_information",
                CurrentBatchRolloutPoolInformation(),
            )
            # Fetch additional fields from shared_mem_dict
            pool_info.engine_status = shared_mem_dict.get("engine_status", None)
            pool_info.global_step = shared_mem_dict.get("global_step", None)
            pool_info.booting_start_time = shared_mem_dict.get("booting_start_time", None)

            # Build running_episode_details for claimed episodes
            running_episode_details = {}
            current_time = time.time()
            for k, v in shared_mem_dict.items():
                if is_key_epsisode_status(k):
                    es: EpisodeStatus = v
                    if es.episode_status == "claimed":
                        time_since_last_activity = current_time - es.latest_activity_timestamp
                        running_episode_details[es.episode_uuid] = {
                            "episode_status": es.episode_status,
                            "time_since_last_activity": f"{time_since_last_activity:.1f}s",
                            "discard_episode_timeout": f"{es.discard_episode_timeout:.1f}s",
                        }
            pool_info.running_episode_details = running_episode_details if running_episode_details else None

            return pool_info
        except Exception as e:
            logger.error(f"Error getting current batch rollout pool information: {e}")
            return CurrentBatchRolloutPoolInformation()

    # --------------------------------------------------------------------
    # ------------ bring engine back to ENGINE.OFFLINE -------------------
    # --------------------------------------------------------------------
    @app.post("/stop_engine")
    async def stop_engine():
        """
        Terminate the training engine and reset all state.
        This will:
        - Kill the training process and all its subprocesses (forcefully if necessary)
        - Set engine status to OFFLINE
        - Remove all episodes (registered, claimed, and unclaimed)
        - Clean up shared memory state
        """
        kill_process_tree(shared_mem_dict_lock, shared_mem_dict)

    return app, register_episode_ready_listener()
