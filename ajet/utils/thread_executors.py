from ajet.utils.sington import singleton
import concurrent.futures


@singleton
class SharedInterchangeThreadExecutor:
    def __init__(self, max_workers=64):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def get_shared_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self.executor


@singleton
class SharedInferenceTrackerThreadExecutor:
    def __init__(self, max_workers=64):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def get_shared_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self.executor
