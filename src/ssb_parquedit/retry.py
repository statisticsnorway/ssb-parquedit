"""Retry decorator for DuckLake lock conflicts."""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_LOCK_CONFLICT_MARKERS = ("conflict", "lock", "concurrent", "serialization", "cannot rollback",)


def _is_lock_conflict(e: Exception) -> bool:
    """Return True if the exception looks like a DuckLake lock/snapshot conflict.

    Args:
        e: The exception to inspect.

    Returns:
        True if the exception message contains known conflict markers.
    """
    msg = str(e).lower()
    return any(marker in msg for marker in _LOCK_CONFLICT_MARKERS)


def with_retry(
    max_retries: int = 5,
    retry_delay_seconds: float = 1.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that retries a function on DuckLake lock conflicts.

    Uses exponential backoff with jitter between attempts. If the final
    attempt also fails with a lock conflict, the exception is re-raised.
    Non-lock exceptions are always re-raised immediately without retrying.

    Args:
        max_retries: Maximum number of attempts before giving up. Defaults to 5.
        retry_delay_seconds: Base delay in seconds for the first retry.
            Subsequent retries use exponential backoff. Defaults to 1.0.

    Returns:
        A decorator that wraps the target function with retry logic.

    Example:
        >>> @with_retry(max_retries=3, retry_delay_seconds=0.5)
        ... def my_func():
        ...     ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    is_last_attempt = attempt == max_retries - 1
                    if _is_lock_conflict(e) and not is_last_attempt:
                        wait = retry_delay_seconds * (2**attempt) + random.uniform(0, 0.5)
                        logger.info(
                            "Lock conflict in '%s' (attempt %d/%d), retrying in %.1fs...",
                            func.__qualname__,
                            attempt + 1,
                            max_retries,
                            wait,
                        )
                        time.sleep(wait)
                    else:
                        raise
            # unreachable, but satisfies type checker
            raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover

        return wrapper
    return decorator