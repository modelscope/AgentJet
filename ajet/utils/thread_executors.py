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

    def __init__(self, workers=100, max_parallel=None, auto_retry=True, block_first_run=False):
        self._max_workers = workers
        if max_parallel is None:
            self._max_parallel = workers
        else:
            self._max_parallel = max_parallel
        self._executor = ThreadPoolExecutor(max_workers=self._max_parallel)
        self._submitted_count = 0
        self._auto_retry = auto_retry
        self.current_futures = []
        self._slow_first_run = block_first_run

    def submit(self, fn, *args, **kwargs):
        """Submit a task, blocking if the pending queue is full."""

        def retry_wrapper(fn, *args, **kwargs):
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"[run_episodes_until_all_complete] Error executing episode: {e}. Retrying...")

        if self._auto_retry:
            future = self._executor.submit(retry_wrapper, fn, *args, **kwargs)
        else:
            future = self._executor.submit(fn, *args, **kwargs)

        if self._slow_first_run:
            self._slow_first_run = False
            future.result()  # Wait for the first run to complete before allowing more tasks to be submitted

        return future

    def submit_with_periodic_drain(self, fn, *args, **kwargs):
        """Submit a task, draining all in-flight work every `drain_every_n_job` submissions."""
        drain_every_n_job = self._max_workers
        results = []
        if self._submitted_count > 0 and self._submitted_count % drain_every_n_job == 0:
            for future in self.current_futures:
                try:
                    results += [future.result()]  # Wait for the task to complete and raise exceptions if any
                except Exception as e:
                    logger.exception(f"Error in task execution: {e}")
            self.current_futures = []

        self._submitted_count += 1
        future = self.submit(fn, *args, **kwargs)
        self.current_futures.append(future)
        return future, results

    def shutdown(self, wait=True):
        """Shut down the underlying executor."""
        self._executor.shutdown(wait=wait)
