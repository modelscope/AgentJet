"""End-to-end smoke test for POST /v1/messages (real uvicorn server).

Stands up:
  - a ZMQ REP stub worker replying with canned ChatCompletion JSON
  - the REAL get_app() FastAPI server on 127.0.0.1 via uvicorn
then drives /v1/messages with the REAL anthropic SDK as a client.

No GPU / no swarm engine — only the LLM reply is stubbed.
"""
import base64
import json
import os
import threading
import time
import uuid
import zmq

import uvicorn

from ajet.tuner_lib.experimental.oai_model_server import get_app


def make_chat_completion(*, text=None, tool_call=None, finish="stop"):
    message = {"role": "assistant", "content": text}
    if tool_call:
        message["tool_calls"] = [tool_call]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "claude-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }


def run_zmq_stub(addr: str, replies: list, holder: dict):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(addr)
    holder["ready"] = True
    for cc in replies:
        try:
            _req = sock.recv_string()
            sock.send_string(json.dumps(cc))
        except zmq.ZMQError:
            break
    time.sleep(3)
    sock.close()
    ctx.term()


def make_auth(episode_address: str) -> str:
    payload = {
        "agent_name": "smoke-agent",
        "target_tag": "smoke-tag",
        "episode_uuid": "ep-" + uuid.uuid4().hex[:8],
        "episode_address": episode_address,
    }
    return "Bearer " + base64.b64encode(json.dumps(payload).encode()).decode()


app, _ = get_app(max_fastapi_threads=4, enable_swarm_mode=False)
PORT = 5800 + (os.getpid() % 500)
BASE = f"http://127.0.0.1:{PORT}"


def wait_port(port, timeout=15):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def main():
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    t_srv = threading.Thread(target=server.run, daemon=True)
    t_srv.start()
    assert wait_port(PORT), "uvicorn did not come up"
    print(f"[setup] server up at {BASE}")

    from anthropic import Anthropic

    # ---- Test 1: plain text round-trip -----------------------------------
    zmq_addr1 = f"tcp://127.0.0.1:{6800 + (os.getpid() % 500)}"
    holder = {"ready": False}
    t1 = threading.Thread(
        target=run_zmq_stub,
        args=(zmq_addr1, [make_chat_completion(text="The answer is \\boxed{42}.")], holder),
        daemon=True,
    )
    t1.start()
    while not holder["ready"]:
        time.sleep(0.01)

    c1 = Anthropic(base_url=BASE, api_key=make_auth(zmq_addr1), max_retries=0, timeout=30)
    resp = c1.messages.create(
        model="claude-test", max_tokens=1024, system="You are a math assistant.",
        messages=[{"role": "user", "content": "What is 6*7?"}],
    )
    print(f"[test1] type={resp.type} stop_reason={resp.stop_reason}")
    print(f"[test1] blocks={[(b.type, getattr(b,'text',None)) for b in resp.content]}")
    print(f"[test1] usage={resp.usage.input_tokens}/{resp.usage.output_tokens}")
    assert resp.type == "message" and resp.stop_reason == "end_turn"
    assert any(b.type == "text" and "42" in (b.text or "") for b in resp.content)
    print("[test1] PASS\n")

    # ---- Test 2: tool_use round-trip -------------------------------------
    zmq_addr2 = zmq_addr1 + "2"
    tool = {
        "name": "python_code_with_standard_io", "description": "Run python.",
        "input_schema": {"type": "object",
                         "properties": {"code": {"type": "string"}, "input": {"type": "string"}},
                         "required": ["code", "input"]},
    }
    holder2 = {"ready": False}
    t2 = threading.Thread(
        target=run_zmq_stub,
        args=(zmq_addr2, [make_chat_completion(
            text="Let me compute.",
            tool_call={"id": "call_abc", "type": "function",
                       "function": {"name": "python_code_with_standard_io",
                                    "arguments": json.dumps({"code": "print(6*7)", "input": ""})}},
            finish="tool_calls")], holder2),
        daemon=True,
    )
    t2.start()
    while not holder2["ready"]:
        time.sleep(0.01)

    c2 = Anthropic(base_url=BASE, api_key=make_auth(zmq_addr2), max_retries=0, timeout=30)
    resp2 = c2.messages.create(
        model="claude-test", max_tokens=1024, system="You are a math assistant.",
        messages=[{"role": "user", "content": "Compute 6*7 with the tool."}],
        tools=[tool],
    )
    print(f"[test2] stop_reason={resp2.stop_reason}")
    print(f"[test2] blocks={[(b.type, getattr(b,'text',None), getattr(b,'name',None), getattr(b,'input',None)) for b in resp2.content]}")
    tu = next((b for b in resp2.content if b.type == "tool_use"), None)
    assert resp2.stop_reason == "tool_use", resp2.stop_reason
    assert tu is not None and tu.name == "python_code_with_standard_io"
    assert tu.input == {"code": "print(6*7)", "input": ""}, tu.input
    assert tu.id == "call_abc"
    print("[test2] PASS\n")

    # ---- Test 3: streaming ------------------------------------------------
    zmq_addr3 = zmq_addr1 + "3"
    holder3 = {"ready": False}
    t3 = threading.Thread(
        target=run_zmq_stub,
        args=(zmq_addr3, [make_chat_completion(text="\\boxed{99}")], holder3),
        daemon=True,
    )
    t3.start()
    while not holder3["ready"]:
        time.sleep(0.01)
    c3 = Anthropic(base_url=BASE, api_key=make_auth(zmq_addr3), max_retries=0, timeout=30)
    events = []
    with c3.messages.stream(model="claude-test", max_tokens=1024,
                            messages=[{"role": "user", "content": "stream test"}]) as stream:
        for ev in stream:
            events.append(ev.type)
        final = stream.get_final_message()
    print(f"[test3] events={events}")
    assert "message_start" in events and "message_stop" in events and "content_block_delta" in events
    assert final.stop_reason == "end_turn"
    print("[test3] PASS\n")

    print("ALL_SMOKE_TESTS_PASSED")
    server.should_exit = True


if __name__ == "__main__":
    main()
