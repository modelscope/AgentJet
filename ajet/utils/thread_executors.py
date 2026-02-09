from concurrent.futures import ThreadPoolExecutor
from ajet.utils.sington import singleton
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