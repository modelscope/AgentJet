import asyncio
import concurrent.futures
import logging
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


def suppress_httpx_aclose_exception():
    """
    Suppress the 'Task exception was never retrieved' error from httpx AsyncClient.aclose().
    This error occurs when the event loop is closed before the AsyncClient is properly closed.
    """
    # Custom exception handler for asyncio
    def custom_exception_handler(loop, context):
        exception = context.get('exception')
        message = context.get('message', '')

        # Check if this is the specific httpx aclose RuntimeError we want to suppress
        if exception is not None:
            if isinstance(exception, RuntimeError):
                exc_str = str(exception)
                if 'unable to perform operation on' in exc_str and 'the handler is closed' in exc_str:
                    return  # Suppress this specific error
                if 'TCPTransport' in exc_str and 'closed' in exc_str:
                    return  # Suppress this specific error

        # For other exceptions, use the default handler
        loop.default_exception_handler(context)

    # Apply custom exception handler to current or new event loop
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_exception_handler)
    except RuntimeError:
        # No running loop, will be applied when loop starts
        pass

    # Also filter the logging output for this specific error
    class HttpxAcloseFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if 'Task exception was never retrieved' in msg and 'aclose' in msg:
                return False
            if 'unable to perform operation on' in msg and 'the handler is closed' in msg:
                return False
            if 'TCPTransport' in msg and 'closed' in msg:
                return False
            return True

    # Apply filter to root logger and asyncio logger
    logging.getLogger().addFilter(HttpxAcloseFilter())
    logging.getLogger('asyncio').addFilter(HttpxAcloseFilter())
