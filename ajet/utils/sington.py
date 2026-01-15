import concurrent.futures
from ajet.utils.testing_utils import singleton

@singleton
class ThreadExecutorSingleton:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)

    def get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self.executor


@singleton
class ThreadExecutorContextTrackerSingleton:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)

    def get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self.executor