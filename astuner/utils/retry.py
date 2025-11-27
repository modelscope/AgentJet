import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from astuner.utils.testing_utils import GoodbyeException, TestFailException

T = TypeVar("T")


def retry_with_backoff(
    max_retry: int = 3,
    backoff_fn: Optional[Callable[[int], float]] = None,
    max_retry_attr: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff and structured logging."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            target_max_retry = max_retry
            if max_retry_attr and args:
                candidate = getattr(args[0], max_retry_attr, None)
                if isinstance(candidate, int) and candidate > 0:
                    target_max_retry = candidate
            if target_max_retry < 1:
                target_max_retry = 1

            for attempt in range(target_max_retry):
                try:
                    return func(*args, **kwargs)
                except GoodbyeException as exc:  # noqa: BLE001
                    raise exc
                except TestFailException as exc:  # noqa: BLE001
                    raise exc
                except Exception as exc:  # noqa: BLE001
                    if attempt < target_max_retry - 1:
                        logger.bind(exception=True).exception(
                            f"{func.__name__} error: {exc.args}, retrying {attempt + 1}/{target_max_retry}"
                        )
                        sleep_seconds = backoff_fn(attempt) if backoff_fn else 2**attempt
                        time.sleep(sleep_seconds)
                    else:
                        logger.bind(exception=True).exception(
                            f"{func.__name__} failed after {target_max_retry} retries: {exc.args}"
                        )
                        raise

            raise RuntimeError("retry_with_backoff exhausted attempts")

        return wrapper

    return decorator
