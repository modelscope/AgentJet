import time
from typing import Dict, Callable, Any
from functools import wraps

def cache_with_ttl(ttl: float):
    """
    Decorator that caches function results for a specified time-to-live (TTL).

    Args:
        ttl: Time-to-live in seconds for cached results
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, tuple[Any, float]] = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            current_time = time.time()

            # Check if cached result exists and is still valid
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if current_time - cached_time < ttl:
                    return cached_result

            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            return result

        return wrapper
    return decorator
