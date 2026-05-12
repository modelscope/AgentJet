"""Lightweight tokenizer cache service.

A companion process holds an HF tokenizer and serves ``encode``, ``decode``
and ``apply_chat_template`` calls over a ZMQ ``ipc://`` socket with an LRU
cache. The caller keeps a local tokenizer for everything else (attributes,
``__call__``, etc.) — only those three hot methods cross the wire.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Optional

import msgpack
import zmq


_CACHE_OPS = ("encode", "decode", "apply_chat_template")
_DEFAULT_CACHE_SIZE = 4096
_DEFAULT_RECV_TIMEOUT_MS = 120_000


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _serve(
    model_path: str,
    ipc_path: str,
    *,
    trust_remote_code: bool,
    cache_size: int,
    ready_file: Optional[str],
) -> None:
    from loguru import logger
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    cache: "OrderedDict[bytes, bytes]" = OrderedDict()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"ipc://{ipc_path}")

    if ready_file:
        try:
            with open(ready_file, "w") as fh:
                fh.write(str(os.getpid()))
        except OSError as exc:
            logger.warning(f"failed to write ready file {ready_file}: {exc}")

    logger.info(f"Tokenizer cache service ready at ipc://{ipc_path} (pid={os.getpid()})")

    hits = misses = 0
    while True:
        try:
            raw = sock.recv()
        except (zmq.ContextTerminated, KeyboardInterrupt):
            break

        # The request bytes are a stable cache key — msgpack is deterministic
        # for our payloads (lists/dicts of primitives), so identical calls
        # produce identical raw frames.
        if raw in cache:
            cache.move_to_end(raw)
            hits += 1
            sock.send(cache[raw])
            continue

        try:
            req = msgpack.unpackb(raw, raw=False)
            op = req.get("op")
            args = req.get("args") or []
            kwargs = req.get("kwargs") or {}
        except Exception as exc:
            sock.send(msgpack.packb({"ok": False, "error": f"bad request: {exc}"}, use_bin_type=True))
            continue

        if op == "shutdown":
            sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
            break
        if op == "stats":
            total = hits + misses
            payload = {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total else 0.0,
                "size": len(cache),
                "max_size": cache_size,
            }
            sock.send(msgpack.packb({"ok": True, "result": payload}, use_bin_type=True))
            continue

        try:
            if op == "encode":
                result = tokenizer.encode(*args, **kwargs)
            elif op == "decode":
                result = tokenizer.decode(*args, **kwargs)
            elif op == "apply_chat_template":
                result = tokenizer.apply_chat_template(*args, **kwargs)
            else:
                sock.send(
                    msgpack.packb({"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True)
                )
                continue
        except Exception as exc:
            import traceback

            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            sock.send(msgpack.packb({"ok": False, "error": err}, use_bin_type=True))
            continue

        misses += 1
        payload = msgpack.packb({"ok": True, "result": result}, use_bin_type=True)
        if op in _CACHE_OPS:
            cache[raw] = payload
            while len(cache) > cache_size:
                cache.popitem(last=False)
        sock.send(payload)

    sock.close(linger=0)
    ctx.term()
    logger.info("Tokenizer cache service stopped")


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class CachedTokenizer:
    """Wraps a local HF tokenizer; routes ``encode`` / ``decode`` /
    ``apply_chat_template`` through a cache process, falls back to the local
    instance for every other attribute."""

    def __init__(
        self,
        local_tokenizer,
        ipc_path: str,
        *,
        recv_timeout_ms: int = _DEFAULT_RECV_TIMEOUT_MS,
    ):
        self._local = local_tokenizer
        self._ipc_path = ipc_path
        self._recv_timeout_ms = recv_timeout_ms
        self._tls = threading.local()  # one REQ socket per thread

    def _socket(self) -> zmq.Socket:
        sock = getattr(self._tls, "sock", None)
        if sock is not None:
            return sock
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        sock.connect(f"ipc://{self._ipc_path}")
        self._tls.sock = sock
        return sock

    def _call(self, op: str, args, kwargs):
        sock = self._socket()
        sock.send(msgpack.packb({"op": op, "args": list(args), "kwargs": kwargs}, use_bin_type=True))
        try:
            raw = sock.recv()
        except zmq.Again as exc:
            # REQ socket is stuck after a timeout; recreate on next call.
            try:
                sock.close(linger=0)
            except zmq.ZMQError:
                pass
            self._tls.sock = None
            raise RuntimeError(f"tokenizer service timeout (op={op})") from exc

        resp = msgpack.unpackb(raw, raw=False)
        if not resp.get("ok"):
            raise RuntimeError(f"tokenizer service error (op={op}): {resp.get('error')}")
        return resp.get("result")

    # -- the three cached methods ----------------------------------------

    def encode(self, text, **kwargs):
        return self._call("encode", [text], kwargs)

    def decode(self, ids, skip_special_tokens: bool = False, **kwargs):
        # HF accepts both ``decode(123)`` and ``decode([123, 456])``. Normalize
        # to a list so the cache key is stable and msgpack-serializable.
        if isinstance(ids, int):
            ids = [ids]
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
            if isinstance(ids, int):  # 0-d tensor
                ids = [ids]
        else:
            ids = list(ids)
        return self._call("decode", [ids], {"skip_special_tokens": skip_special_tokens, **kwargs})

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        kw = {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize, **kwargs}
        if tools is not None:
            kw["tools"] = tools
        return self._call("apply_chat_template", [conversation], kw)

    # -- everything else: defer to local tokenizer -----------------------

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._local, name)

    def __call__(self, *args, **kwargs):
        return self._local(*args, **kwargs)

    def stats(self) -> dict:
        return self._call("stats", [], {})

    def shutdown_service(self) -> None:
        try:
            self._call("shutdown", [], {})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def start_tokenizer_service(
    local_tokenizer,
    model_path: str,
    *,
    trust_remote_code: bool = False,
    ipc_path: Optional[str] = None,
    cache_size: int = _DEFAULT_CACHE_SIZE,
    ready_timeout_s: float = 180.0,
    recv_timeout_ms: int = _DEFAULT_RECV_TIMEOUT_MS,
) -> CachedTokenizer:
    """Spawn the cache service subprocess and return a wrapper around the
    caller-supplied local tokenizer.
    """
    from loguru import logger

    cleanup_dir: Optional[str] = None
    if ipc_path is None:
        cleanup_dir = tempfile.mkdtemp(prefix="ajet-tok-")
        ipc_path = os.path.join(cleanup_dir, "sock")

    ready_file = ipc_path + ".ready"
    if os.path.exists(ready_file):
        try:
            os.remove(ready_file)
        except OSError:
            pass

    cmd = [
        sys.executable, "-m", "ajet.tokenizer.service", "serve",
        "--model-path", model_path,
        "--ipc-path", ipc_path,
        "--cache-size", str(cache_size),
        "--ready-file", ready_file,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    logger.info(f"Launching tokenizer cache service: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=os.environ.copy())

    def _cleanup() -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception:
                pass
        if cleanup_dir and os.path.isdir(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    atexit.register(_cleanup)

    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"tokenizer service exited with code {proc.returncode}")
        if os.path.exists(ready_file):
            break
        time.sleep(0.1)
    else:
        _cleanup()
        raise RuntimeError(
            f"tokenizer service at ipc://{ipc_path} did not become ready within {ready_timeout_s}s"
        )

    return CachedTokenizer(local_tokenizer, ipc_path, recv_timeout_ms=recv_timeout_ms)


# ---------------------------------------------------------------------------
# CLI entry point (used by the subprocess)
# ---------------------------------------------------------------------------


def _main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="ZMQ IPC tokenizer cache service")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--model-path", required=True)
    p_serve.add_argument("--ipc-path", required=True)
    p_serve.add_argument("--cache-size", type=int, default=_DEFAULT_CACHE_SIZE)
    p_serve.add_argument("--ready-file", default=None)
    p_serve.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        def _on_term(*_):
            os._exit(0)

        signal.signal(signal.SIGTERM, _on_term)
        _serve(
            model_path=args.model_path,
            ipc_path=args.ipc_path,
            trust_remote_code=args.trust_remote_code,
            cache_size=args.cache_size,
            ready_file=args.ready_file,
        )


if __name__ == "__main__":
    _main()
