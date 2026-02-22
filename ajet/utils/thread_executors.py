from concurrent.futures import ThreadPoolExecutor
from ajet.utils.sington import singleton
from loguru import logger
import threading


@singleton
class SharedInterchangeThreadExecutor:
    def __init__(self, max_workers=64):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_shared_executor(self) -> ThreadPoolExecutor:
        return self.executor



@singleton
class SharedInferenceTrackerThreadExecutor:
    def __init__(self, max_workers=64):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_shared_executor(self) -> ThreadPoolExecutor:
        return self.executor


class BoundedThreadPoolExecutor:
    def __init__(self, max_workers, max_queue_size=100):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = threading.Semaphore(max_queue_size)

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()

        def wrapped_fn(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                self.semaphore.release()

        return self.executor.submit(wrapped_fn, *args, **kwargs)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)

class PeriodicDrainThreadPoolExecutor:
    """A ThreadPoolExecutor that bounds the number of pending tasks via a semaphore."""

    def __init__(self, workers=100, auto_retry=True):
        self._max_workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._submitted_count = 0
        self._auto_retry = auto_retry

    def submit(self, fn, *args, **kwargs):
        """Submit a task, blocking if the pending queue is full."""

        def retry_wrapper(func, arg):
            while True:
                try:
                    return func(arg)
                except Exception as e:
                    logger.exception(f"[run_episodes_until_all_complete] Error executing episode: {e}. Retrying...")

        if self._auto_retry:
            return self._executor.submit(retry_wrapper, fn, *args, **kwargs)
        else:
            return self._executor.submit(fn, *args, **kwargs)

    def submit_with_periodic_drain(self, fn, *args, **kwargs):
        """Submit a task, draining all in-flight work every `drain_every_n_job` submissions."""
        drain_every_n_job = self._max_workers
        if self._submitted_count > 0 and self._submitted_count % drain_every_n_job == 0:
            self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        self._submitted_count += 1
        return self.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        """Shut down the underlying executor."""
        self._executor.shutdown(wait=wait)