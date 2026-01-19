from multiprocessing.managers import DictProxy
import threading

import zmq

from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import List

from typing import Coroutine, Optional, Tuple
from ajet.schema.task import WorkflowOutput


class SyncTrainConfigRequest(BaseModel):
    yaml_as_string: str

class ClaimEpisodeRequest(BaseModel):
    client_uuid: str
    episode_type: str

class ClaimEpisodeResponse(BaseModel):
    success: bool
    client_uuid: str
    episode_uuid: str
    openai_base_url: str = ""
    openai_api_key: str = ""
    fail_cause: str = ""

class CanContinueEpisodeRequest(BaseModel):
    client_uuid: str
    episode_uuid: str

class CanContinueEpisodeResponse(BaseModel):
    can_continue: bool

class EndEpisodeRequest(BaseModel):
    client_uuid: str
    episode_uuid: str
    workflow_output: WorkflowOutput

class EndEpisodeResponse(BaseModel):
    success: bool


class EpisodeStatus(BaseModel):
    episode_uuid: str
    episode_status: str = "rolling"
    openai_base_url: str = ""
    openai_api_key: str = ""
    client_uuid: str = ""
    zmq_listen_result_addr: str = ""

class EpisodeBufferResponse(BaseModel):
    buffer: List[EpisodeStatus]


class BoolResponse(BaseModel):
    success: bool

class RegisterEpisodeRequest(BaseModel):
    episode_uuid: str
    openai_base_url: str = ""
    openai_api_key: str = ""
    zmq_listen_result_addr: str = ""


class UpdateEngineStatusRequest(BaseModel):
    engine_status: str = ""


DEBUG = True

def register_enable_tinkerscript_mode_routes(app, zmq_context, shared_mem_dict:DictProxy, shared_mem_dict_lock:threading.Lock) -> Tuple[FastAPI, Optional[Coroutine]]:

    if 'episodes' not in shared_mem_dict:
        shared_mem_dict["episodes"] = {}

    if 'unclaimed_episodes' not in shared_mem_dict:
        shared_mem_dict['unclaimed_episodes'] = []

    @app.post("/sync_train_config")
    async def sync_train_config(req: SyncTrainConfigRequest):
        # dummy: just print the yaml string
        try:
            print("[sync_train_config] received yaml:", req.yaml_as_string)
        except Exception:
            pass
        return {"success": True}


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
        )

        with shared_mem_dict_lock:
            shared_mem_dict[f"episodes-{episode_uuid}"] = es
            shared_mem_dict['unclaimed_episodes'] += [req.episode_uuid]

        return BoolResponse(
            success=True,
        )


    @app.post("/claim_episode", response_model=ClaimEpisodeResponse)
    async def claim_episode(req: ClaimEpisodeRequest):
        # placeholder implementation — real logic should check episode_semaphore

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



    @app.post("/end_episode", response_model=EndEpisodeResponse)
    async def end_episode(req: EndEpisodeRequest):
        # receive workflow output data
        client_uuid = req.client_uuid
        episode_uuid = req.episode_uuid
        workflow_output = req.workflow_output

        if 'episodes' not in shared_mem_dict:
            raise HTTPException(status_code=400, detail=f"No episodes registered yet.")
        if (f"episodes-{episode_uuid}") not in shared_mem_dict:
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
