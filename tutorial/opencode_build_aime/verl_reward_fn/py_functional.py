# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small python utility functions
"""

import atexit
import importlib
import multiprocessing
import os
import pickle
import queue  # Import the queue module for exception type hint
import signal
import threading
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional


# --- Persistent worker pool for timed calls ---
# A single long-lived child process handles all timed calls from this Python process.
# Avoids per-call fork/spawn overhead, which under heavy concurrent load can itself
# exceed the configured timeout on trivial inputs.


def _pool_worker_loop(in_q: "multiprocessing.Queue", out_q: "multiprocessing.Queue") -> None:
    """Child-side loop: read tasks, run with SIGALRM timeout, push result."""
    while True:
        task = in_q.get()
        if task is None:  # poison pill
            return
        func, args, kwargs, timeout = task

        def _handler(signum, frame):
            fname = getattr(func, "__name__", "target")
            raise TimeoutError(f"Function {fname} timed out after {timeout} seconds (persistent worker)!")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            result = func(*args, **kwargs)
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                out_q.put((True, result))
            except Exception:
                out_q.put((False, RuntimeError("Result is not pickleable")))
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                pickle.dumps(e)
                out_q.put((False, e))
            except (pickle.PicklingError, TypeError):
                out_q.put(
                    (False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}"))
                )
        finally:
            signal.signal(signal.SIGALRM, old)


_TIMEOUT_POOL_SIZE = int(os.environ.get("TIMEOUT_POOL_SIZE", "64"))


class _Worker:
    """Wraps one long-lived child process with dedicated in/out queues."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._in_q = ctx.Queue()
        self._out_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_pool_worker_loop, args=(self._in_q, self._out_q), daemon=True
        )
        self._proc.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def submit(self, func: Callable, args: tuple, kwargs: dict, timeout: float) -> None:
        self._in_q.put((func, args, kwargs, timeout))

    def get_result(self, timeout: float):
        return self._out_q.get(timeout=timeout)

    def kill(self) -> None:
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.5)
            if self._proc is not None and self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=0.5)
        except Exception:
            pass
        for q in (self._in_q, self._out_q):
            try:
                if q is not None:
                    q.close()
                    q.join_thread()
            except Exception:
                pass
        self._proc = None
        self._in_q = None
        self._out_q = None

    def shutdown(self) -> None:
        try:
            if self._in_q is not None:
                self._in_q.put(None)
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=0.5)
        except Exception:
            pass
        self.kill()


class _PersistentTimeoutPool:
    """N-worker persistent pool serving this process.

    Up to `size` long-lived child workers are spawned lazily. Each call acquires
    a free worker, submits the task, waits for the result with a small grace
    over the configured timeout, and returns the worker to the free list. If a
    worker becomes unresponsive, it is terminated and a replacement is created
    on the next acquire.
    """

    def __init__(self, size: int) -> None:
        self._size = max(1, int(size))
        self._ctx = multiprocessing.get_context("fork") if os.name == "posix" else multiprocessing.get_context()
        self._cond = threading.Condition()
        self._free: list[_Worker] = []
        self._alive_count = 0
        self._owner_pid = os.getpid()

    def _reset_if_forked(self) -> None:
        if os.getpid() != self._owner_pid:
            # A fork happened after pool init — child inherits dead handles; start over.
            self._free = []
            self._alive_count = 0
            self._owner_pid = os.getpid()

    def _acquire(self) -> _Worker:
        with self._cond:
            while True:
                self._reset_if_forked()
                # Reuse an already-spawned free worker.
                while self._free:
                    w = self._free.pop()
                    if w.is_alive():
                        return w
                    # Stale (died in background) — drop it.
                    self._alive_count -= 1
                # Room to spawn a new one?
                if self._alive_count < self._size:
                    w = _Worker(self._ctx)
                    self._alive_count += 1
                    return w
                # All workers busy; wait for one to be released.
                self._cond.wait()

    def _release(self, w: _Worker, keep: bool) -> None:
        with self._cond:
            if keep and w.is_alive():
                self._free.append(w)
            else:
                self._alive_count -= 1
                w.kill()
            self._cond.notify()

    def call(self, func: Callable, args: tuple, kwargs: dict, timeout: float) -> Any:
        w = self._acquire()
        keep = True
        try:
            w.submit(func, args, kwargs, timeout)
            try:
                success, payload = w.get_result(timeout=timeout + 1.0)
            except queue.Empty:
                keep = False
                w.kill()
                fname = getattr(func, "__name__", "target")
                raise TimeoutError(
                    f"Function {fname} timed out after {timeout} seconds (persistent worker, unresponsive)!"
                )
        except BaseException:
            # If we failed mid-conversation (e.g., queue put error), don't reuse this worker.
            keep = False
            raise
        finally:
            self._release(w, keep=keep and w.is_alive())

        if success:
            return payload
        raise payload

    def shutdown(self) -> None:
        with self._cond:
            workers = list(self._free)
            self._free.clear()
        for w in workers:
            w.shutdown()


_GLOBAL_TIMEOUT_POOL = _PersistentTimeoutPool(size=_TIMEOUT_POOL_SIZE)
atexit.register(_GLOBAL_TIMEOUT_POOL.shutdown)


# Renamed the function from timeout to timeout_limit
def timeout_limit(seconds: float, use_signals: bool = False):
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).
    """

    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            # Issue deprecation warning if use_signals is explicitly True
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                # Use setitimer for float seconds support, alarm only supports integers
                signal.setitimer(signal.ITIMER_REAL, seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    # Reset timer and handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result

            return wrapper_signal
        else:
            # --- Persistent-worker-pool based timeout ---
            # Dispatches to a single long-lived child process per parent process,
            # which runs each call with a SIGALRM timeout internally. Avoids the
            # per-call process startup overhead of the old implementation.
            @wraps(func)
            def wrapper_pool(*args, **kwargs):
                return _GLOBAL_TIMEOUT_POOL.call(func, args, kwargs, seconds)

            return wrapper_pool

    return decorator


def union_two_dict(dict1: dict, dict2: dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def append_to_dict(data: dict, new_data: dict):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class NestedNamespace(SimpleNamespace):
    """A nested version of SimpleNamespace that recursively converts dictionaries to namespaces.

    This class allows for dot notation access to nested dictionary structures by recursively
    converting dictionaries to NestedNamespace objects.

    Example:
        config_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = NestedNamespace(config_dict)
        # Access with: config.a, config.b.c, config.b.d

    Args:
        dictionary: The dictionary to convert to a nested namespace.
        **kwargs: Additional attributes to set on the namespace.
    """

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


class DynamicEnumMeta(type):
    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._registry.values())

    def __contains__(cls, item: Any) -> bool:
        # allow `name in EnumClass` or `member in EnumClass`
        if isinstance(item, str):
            return item in cls._registry
        return item in cls._registry.values()

    def __getitem__(cls, name: str) -> Any:
        return cls._registry[name]

    def __reduce_ex__(cls, protocol):
        # Always load the existing module and grab the class
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)

    def names(cls):
        return list(cls._registry.keys())

    def values(cls):
        return list(cls._registry.values())


class DynamicEnum(metaclass=DynamicEnumMeta):
    _registry: dict[str, "DynamicEnum"] = {}
    _next_value: int = 0

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __reduce_ex__(self, protocol):
        """
        Unpickle via: getattr(import_module(module).Dispatch, 'ONE_TO_ALL')
        so the existing class is reused instead of re-executed.
        """
        module = importlib.import_module(self.__class__.__module__)
        enum_cls = getattr(module, self.__class__.__name__)
        return getattr, (enum_cls, self.name)

    @classmethod
    def register(cls, name: str) -> "DynamicEnum":
        key = name.upper()
        if key in cls._registry:
            raise ValueError(f"{key} already registered")
        member = cls(key, cls._next_value)
        cls._registry[key] = member
        setattr(cls, key, member)
        cls._next_value += 1
        return member

    @classmethod
    def remove(cls, name: str):
        key = name.upper()
        member = cls._registry.pop(key)
        delattr(cls, key)
        return member

    @classmethod
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:
        return cls._registry.get(name.upper())


@contextmanager
def temp_env_var(key: str, value: str):
    """Context manager for temporarily setting an environment variable.

    This context manager ensures that environment variables are properly set and restored,
    even if an exception occurs during the execution of the code block.

    Args:
        key: Environment variable name to set
        value: Value to set the environment variable to

    Yields:
        None

    Example:
        >>> with temp_env_var("MY_VAR", "test_value"):
        ...     # MY_VAR is set to "test_value"
        ...     do_something()
        ... # MY_VAR is restored to its original value or removed if it didn't exist
    """
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, ListConfig | DictConfig):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, list | tuple):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj
