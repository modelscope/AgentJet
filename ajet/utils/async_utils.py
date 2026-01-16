import asyncio
import concurrent.futures
from typing import Any

def run_async_coroutine_with_timeout(coro, timeout: int = 3600) -> Any:
    """
    Run an async coroutine with a timeout, supporting both inside and outside event loops.
    Args:
        coro: The coroutine to run.
        timeout (int): Timeout in seconds. Default is 3600.
    Returns:
        Any: The result of the coroutine.
    Raises:
        concurrent.futures.TimeoutError: If the coroutine does not finish in time.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        final_res = asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            try:
                final_res = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
            except Exception:
                raise
    return final_res


def apply_httpx_aclose_patch():
    try:
        from openai._base_client import AsyncHttpxClientWrapper

        _original_init = AsyncHttpxClientWrapper.__init__

        def _patched_init(self, *args, **kwargs):
            try:
                self._created_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._created_loop = None
            _original_init(self, *args, **kwargs)

        def _patched_del(self) -> None:
            if self.is_closed:
                return

            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                return

            if getattr(self, "_created_loop", None) is not None and current_loop is not self._created_loop:
                return

            try:
                current_loop.create_task(self.aclose())
            except Exception:
                pass

        AsyncHttpxClientWrapper.__init__ = _patched_init
        AsyncHttpxClientWrapper.__del__ = _patched_del
        print("Applied httpx aclose patch.")
    except ImportError:
        pass
