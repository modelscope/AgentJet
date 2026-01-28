import multiprocessing
import time
from multiprocessing.managers import DictProxy
import threading
from types import SimpleNamespace

import zmq
import asyncio

from loguru import logger
from fastapi import FastAPI, HTTPException
from typing import List

from typing import Coroutine, Optional, Tuple
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import (
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
)

DEBUG = True

def register_enable_tinkerscript_mode_routes(
        app,
        zmq_context,
        shared_mem_dict:DictProxy,
        shared_mem_dict_lock:threading.Lock,
    ) -> Tuple[FastAPI, Optional[Coroutine]]:

    if 'episodes' not in shared_mem_dict:
        shared_mem_dict["episodes"] = {}

    if 'unclaimed_episodes' not in shared_mem_dict:
        shared_mem_dict['unclaimed_episodes'] = []

    @app.post("/sync_train_config")
    async def sync_train_config(req: SyncTrainConfigRequest):
        """
        Receive training configuration from client as YAML string.
        Store it in shared memory for later use by start_engine.
        """
        try:
            yaml_str = req.yaml_as_string
            logger.info("[sync_train_config] Received training configuration")
            if DEBUG:
                logger.debug(f"[sync_train_config] YAML content:\n{yaml_str}...")

            # Store the YAML config in shared memory for start_engine to use
            with shared_mem_dict_lock:
                shared_mem_dict['train_config_yaml'] = yaml_str

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
        try:
            from ajet.utils.launch_utils import execute_training_process
            from ajet.launcher import (
                get_backbone_target,
                setup_environment_vars,
            )
            from ajet.utils.config_utils import (
                prepare_experiment_config,
            )
            import ray
            import tempfile
            import yaml as yaml_module

            # Check if config has been synced
            if 'train_config_yaml' not in shared_mem_dict:
                logger.error("[start_engine] No training config found. Please call sync_train_config first.")
                return {"success": False, "error": "No training config found"}

            yaml_str = shared_mem_dict['train_config_yaml']

            # Parse YAML to get backbone
            config_dict = yaml_module.safe_load(yaml_str)
            backbone = config_dict.get('ajet', {}).get('backbone', 'verl')
            exp_dir_final = config_dict.get('ajet', {}).get('experiment_dir', 'saved_experiments')

            # Save YAML to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
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

            # Finalize experiment config
            main_yaml_fp, exe_exp_base, exp_name, exp_config = prepare_experiment_config(
                main_yaml_fp, exp_dir_final, backbone
            )

            # Setup environment variables
            env = setup_environment_vars(args, exp_config, main_yaml_fp)

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
                )
            )
            p.daemon = True
            p.start()

            # Store process info in shared memory
            with shared_mem_dict_lock:
                shared_mem_dict['training_process_pid'] = p.pid
                shared_mem_dict['engine_status'] = "running"

            logger.info(f"[start_engine] Successfully started training process (PID: {p.pid})")
            return {"success": True, "pid": p.pid}

        except Exception as e:
            logger.error(f"[start_engine] Error starting engine: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


    # --- engine status ---
    shared_mem_dict['engine_status'] = "booting"
    @app.post("/update_engine_status", response_model=BoolResponse)
    async def update_engine_status(req: UpdateEngineStatusRequest):
        shared_mem_dict['engine_status'] = req.engine_status
        return BoolResponse(success=True)


    @app.get("/get_engine_status")
    async def get_engine_status():
        status = shared_mem_dict['engine_status']
        return {"engine_status": status}


    # --- episode status ---
    @app.post("/register_episode", response_model=BoolResponse)
    async def register_episode(req: RegisterEpisodeRequest):

        episode_uuid = req.episode_uuid
        es = EpisodeStatus(
            episode_uuid=req.episode_uuid,
            openai_base_url=req.openai_base_url,
            openai_api_key=req.openai_api_key,
            episode_status="registered",
            zmq_listen_result_addr=req.zmq_listen_result_addr,
            allow_discard_timeout=-1,
        )
        es.latest_activity_timestamp = time.time()

        with shared_mem_dict_lock:
            shared_mem_dict[f"episodes-{episode_uuid}"] = es
            shared_mem_dict['unclaimed_episodes'] += [req.episode_uuid]

        return BoolResponse(
            success=True,
        )


    @app.post("/claim_episode", response_model=ClaimEpisodeResponse)
    async def claim_episode(req: ClaimEpisodeRequest):
        find_claimed_episodes_that_need_to_be_unclaimed()

        with shared_mem_dict_lock:
            if len(shared_mem_dict['unclaimed_episodes']) <= 0:
                return ClaimEpisodeResponse(
                    success=False,
                    client_uuid=req.client_uuid,
                    episode_uuid="",
                    openai_base_url="",
                    openai_api_key="",
                    fail_cause="No available episodes to claim. Try again (maybe 1 minute) later.",
                )

            # hint: do not optimize this
            episode_uuid = shared_mem_dict['unclaimed_episodes'][0]
            shared_mem_dict['unclaimed_episodes'] = shared_mem_dict['unclaimed_episodes'][1:]

            # get episode
            es:EpisodeStatus = shared_mem_dict[f"episodes-{episode_uuid}"]
            es.episode_status = "claimed"
            es.client_uuid = req.client_uuid
            es.latest_activity_timestamp = time.time()
            es.allow_discard_timeout = req.allow_discard_timeout

            shared_mem_dict[f"episodes-{episode_uuid}"] = es
            openai_base_url = es.openai_base_url
            openai_api_key = es.openai_api_key

        return ClaimEpisodeResponse(
            success=True,
            client_uuid=req.client_uuid,
            episode_uuid=episode_uuid,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            fail_cause="",
        )


    def find_claimed_episodes_that_need_to_be_unclaimed() -> List[str]:
        result = []
        current_time = time.time()

        for k, v in shared_mem_dict.items():
            if k.startswith("episodes-"):
                es:EpisodeStatus = v
                if es.episode_status == "claimed":
                    if (current_time - es.latest_activity_timestamp) > es.allow_discard_timeout:
                        result.append(es.episode_uuid)

        for episode_uuid in result:
            _revert_episode_to_unclaimed(episode_uuid)

        return result


    def _revert_episode_to_unclaimed(episode_uuid: str):
        with shared_mem_dict_lock:
            # check status again, because other thread may have changed it
            if shared_mem_dict[f"episodes-{episode_uuid}"].episode_status != "claimed":
                return

            # revert
            logger.info(f"Reverting episode {episode_uuid} to unclaimed due to client timeout.")
            if f"episodes-{episode_uuid}" in shared_mem_dict:
                es:EpisodeStatus = shared_mem_dict[f"episodes-{episode_uuid}"]
                es.episode_status = "registered"
                es.client_uuid = ""
                es.latest_activity_timestamp = time.time()
                es.allow_discard_timeout = -1
                shared_mem_dict[f"episodes-{episode_uuid}"] = es
                shared_mem_dict['unclaimed_episodes'] += [episode_uuid]


    @app.post("/end_episode", response_model=EndEpisodeResponse)
    async def end_episode(req: EndEpisodeRequest):
        # receive workflow output data
        client_uuid = req.client_uuid
        episode_uuid = req.episode_uuid
        workflow_output = req.workflow_output

        if 'episodes' not in shared_mem_dict:
            logger.error(f"[server] No episodes registered yet.")
            raise HTTPException(status_code=400, detail=f"No episodes registered yet.")
        if (f"episodes-{episode_uuid}") not in shared_mem_dict:
            logger.error(f"[server] Episode {episode_uuid} not found.")
            raise HTTPException(status_code=400, detail=f"Episode {episode_uuid} not found.")

        # send workflow_output to zmq
        assert 'episodes' in shared_mem_dict
        zmq_addr = shared_mem_dict[f"episodes-{episode_uuid}"].zmq_listen_result_addr
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | Received new chat completion request (inside thread)")
        socket = zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 60*1000)  # 1 minute recv timeout
        socket.connect(zmq_addr)
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | connect done")
        socket.send_string(workflow_output.model_dump_json())
        if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | send_string")

        # wait for ack
        for _ in range(5):  # max 5 minutes wait
            try:
                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string begin.")
                result_str = socket.recv_string()
                break
            except zmq.Again as e:
                if DEBUG: logger.info(f"[server] episode_uuid: {episode_uuid} | recv_string timeout, retrying.")
                continue

        # clean up episode records
        with shared_mem_dict_lock:
            del shared_mem_dict[f"episodes-{episode_uuid}"]
            if episode_uuid in shared_mem_dict['unclaimed_episodes']:
                shared_mem_dict['unclaimed_episodes'].remove(episode_uuid)

        # return success
        return EndEpisodeResponse(success=True)



    @app.post("/can_continue_episode", response_model=CanContinueEpisodeResponse)
    async def can_continue_episode(req: CanContinueEpisodeRequest):
        can_continue = (f"episodes-{req.episode_uuid}" in shared_mem_dict)
        can_continue = can_continue and shared_mem_dict[f"episodes-{req.episode_uuid}"]["episode_status"] == "claimed"
        return CanContinueEpisodeResponse(can_continue=can_continue)



    @app.post("/get_episode_buffer", response_model=EpisodeBufferResponse)
    async def get_episode_buffer():
        result = [
            v for k, v in shared_mem_dict.items() if k.startswith("episodes-")
        ]
        return EpisodeBufferResponse(buffer=result)



    async def register_episode_ready_listener():
        pass


    return app, register_episode_ready_listener()
