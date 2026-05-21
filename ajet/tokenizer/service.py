"""Lightweight tokenizer cache service.

A companion process holds an HF tokenizer and serves ``encode``, ``decode``,
``batch_decode`` and ``apply_chat_template`` calls over a ZMQ ``ipc://`` socket
with an LRU cache. The caller keeps a local tokenizer for everything else
(attributes, ``__call__``, etc.) — only those hot methods cross the wire.
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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import msgpack
import zmq


_CACHE_OPS = ("encode", "decode", "batch_decode", "apply_chat_template")
_DEFAULT_CACHE_SIZE = 8192
_DEFAULT_RECV_TIMEOUT_MS = 120_000
_DEFAULT_WORKERS = 4


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
    workers: int = _DEFAULT_WORKERS,
) -> None:
    from loguru import logger
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    _serve_tokenizer(
        tokenizer=tokenizer,
        ipc_path=ipc_path,
        cache_size=cache_size,
        ready_file=ready_file,
        workers=workers,
        logger=logger,
    )


def _serve_tokenizer(
    tokenizer,
    ipc_path: str,
    *,
    cache_size: int,
    ready_file: Optional[str],
    workers: int = _DEFAULT_WORKERS,
    logger=None,
) -> None:
    workers = max(1, workers)

    cache: "OrderedDict[bytes, bytes]" = OrderedDict()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"ipc://{ipc_path}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    if ready_file:
        try:
            with open(ready_file, "w") as fh:
                fh.write(str(os.getpid()))
        except OSError as exc:
            if logger:
                logger.warning(f"failed to write ready file {ready_file}: {exc}")

    if logger:
        logger.info(
            f"Tokenizer cache service ready at ipc://{ipc_path} "
            f"(pid={os.getpid()}, workers={workers})"
        )

    hits = misses = 0
    pending: dict[Future, tuple[list[bytes], bytes, str]] = {}

    def _dispatch(raw: bytes) -> tuple[bytes, str]:
        try:
            req = msgpack.unpackb(raw, raw=False)
            op = req.get("op")
            args = req.get("args") or []
            kwargs = req.get("kwargs") or {}
        except Exception as exc:
            return msgpack.packb(
                {"ok": False, "error": f"bad request: {exc}"}, use_bin_type=True
            ), ""

        try:
            if op == "encode":
                result = tokenizer.encode(*args, **kwargs)
            elif op == "decode":
                result = tokenizer.decode(*args, **kwargs)
            elif op == "batch_decode":
                result = tokenizer.batch_decode(*args, **kwargs)
            elif op == "apply_chat_template":
                result = tokenizer.apply_chat_template(*args, **kwargs)
            else:
                return msgpack.packb(
                    {"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True
                ), str(op)
        except Exception as exc:
            import traceback

            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            return msgpack.packb({"ok": False, "error": err}, use_bin_type=True), str(op)

        return msgpack.packb({"ok": True, "result": result}, use_bin_type=True), str(op)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            try:
                events = dict(poller.poll(timeout=10))
            except (zmq.ContextTerminated, KeyboardInterrupt):
                break

            if sock in events:
                frames = sock.recv_multipart()
                route = frames[:-1]
                raw = frames[-1]

                try:
                    req = msgpack.unpackb(raw, raw=False)
                    op = req.get("op")
                except Exception:
                    op = None

                if op == "shutdown":
                    sock.send_multipart(route + [msgpack.packb({"ok": True}, use_bin_type=True)])
                    break
                if op == "stats":
                    total = hits + misses
                    payload = {
                        "hits": hits,
                        "misses": misses,
                        "hit_rate": hits / total if total else 0.0,
                        "size": len(cache),
                        "max_size": cache_size,
                        "workers": workers,
                        "pending": len(pending),
                    }
                    sock.send_multipart(
                        route + [msgpack.packb({"ok": True, "result": payload}, use_bin_type=True)]
                    )
                    continue

                # The request bytes are a stable cache key — msgpack is deterministic
                # for our payloads (lists/dicts of primitives), so identical calls
                # produce identical raw frames.
                if raw in cache:
                    cache.move_to_end(raw)
                    hits += 1
                    sock.send_multipart(route + [cache[raw]])
                    continue

                future = executor.submit(_dispatch, raw)
                pending[future] = (route, raw, str(op))

            done = [future for future in pending if future.done()]
            for future in done:
                route, raw, op = pending.pop(future)
                payload, result_op = future.result()
                misses += 1
                cache_op = result_op or op
                if cache_op in _CACHE_OPS:
                    cache[raw] = payload
                    while len(cache) > cache_size:
                        cache.popitem(last=False)
                sock.send_multipart(route + [payload])

        for future in pending:
            future.cancel()

    sock.close(linger=0)
    ctx.term()
    if logger:
        logger.info("Tokenizer cache service stopped")


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class CachedTokenizer:
    """Wraps a local HF tokenizer; routes ``encode`` / ``decode`` /
    ``batch_decode`` / ``apply_chat_template`` through a cache process, falls
    back to the local instance for every other attribute."""

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

    # -- cached methods ---------------------------------------------------

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

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        sequences = [
            [sequence] if isinstance(sequence, int) else list(sequence)
            for sequence in sequences
        ]
        return self._call(
            "batch_decode",
            [sequences],
            {"skip_special_tokens": skip_special_tokens, **kwargs},
        )

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
    workers: int = _DEFAULT_WORKERS,
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
        "--workers", str(workers),
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
    p_serve.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
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
            workers=args.workers,
        )


if __name__ == "__main__":
    _main()
