import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Type

class AdaptiveRetry:
    """
    Implements adaptive retry logic with dynamic delays and error handling.
    """

    def __init__(self, 
                 max_retries: int = 5,
                 initial_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 max_delay: float = 60.0,
                 jitter: float = 0.1,
                 error_map: Optional[Dict[Type[Exception], Dict[str, Any]]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initializes the AdaptiveRetry mechanism.

        Args:
            max_retries: Maximum number of retry attempts.
            initial_delay: Initial delay in seconds before the first retry.
            backoff_factor: Factor by which the delay increases (e.g., 2 for exponential backoff).
            max_delay: Maximum delay in seconds between retries.
            jitter: Random jitter factor to prevent thundering herd problem (0.0 to 1.0).
            error_map: A dictionary mapping exception types to specific retry configurations.
                       e.g., {NetworkError: {"max_retries": 10, "initial_delay": 0.5}}
            logger: Optional logger instance.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.error_map = error_map or {}
        self.logger = logger or logging.getLogger(__name__)

    async def retry_async(self, 
                          func: Callable[..., Any],
                          *args, 
                          **kwargs) -> Any:
        """
        Retries an asynchronous function with adaptive backoff.

        Args:
            func: The asynchronous function to retry.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            Exception: If the function fails after all retry attempts.
        """
        current_retries = 0
        current_delay = self.initial_delay

        while current_retries < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                current_retries += 1
                
                # Apply error-specific retry configuration if available
                retry_config = self._get_error_specific_config(type(e))
                
                effective_max_retries = retry_config.get("max_retries", self.max_retries)
                effective_initial_delay = retry_config.get("initial_delay", self.initial_delay)
                effective_backoff_factor = retry_config.get("backoff_factor", self.backoff_factor)
                effective_max_delay = retry_config.get("max_delay", self.max_delay)

                if current_retries >= effective_max_retries:
                    self.logger.error(f"Function {func.__name__} failed after {current_retries} retries. Last error: {e}")
                    raise

                delay = min(effective_max_delay, effective_initial_delay * (effective_backoff_factor ** (current_retries - 1)))
                delay += (self.jitter * delay * (2 * random.random() - 1)) # Add jitter
                delay = max(0.1, delay) # Ensure minimum delay

                self.logger.warning(f"Function {func.__name__} failed ({e}). Retrying in {delay:.2f} seconds (attempt {current_retries}/{effective_max_retries}).")
                await asyncio.sleep(delay)
        
        # This part should ideally not be reached if the loop condition is correct
        raise Exception(f"Function {func.__name__} failed unexpectedly after all retries.")

    def _get_error_specific_config(self, error_type: Type[Exception]) -> Dict[str, Any]:
        """
        Retrieves retry configuration specific to an exception type.
        """
        for exc_type, config in self.error_map.items():
            if issubclass(error_type, exc_type):
                return config
        return {}

import random

# Example Usage (for testing)
async def flaky_operation(should_fail_count: int) -> str:
    global call_count
    call_count += 1
    if call_count <= should_fail_count:
        if random.random() < 0.5:
            raise ConnectionError("Simulated network error")
        else:
            raise ValueError("Simulated data error")
    return "Operation successful!"

async def main():
    global call_count
    call_count = 0

    # Default retry
    retry_handler = AdaptiveRetry(max_retries=3, initial_delay=0.5)
    try:
        print("\n--- Testing default retry ---")
        result = await retry_handler.retry_async(flaky_operation, 2) # Fails 2 times, succeeds on 3rd
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    call_count = 0
    # Retry with custom error map
    error_specific_retry = AdaptiveRetry(
        max_retries=5,
        initial_delay=1.0,
        error_map={
            ConnectionError: {"max_retries": 10, "initial_delay": 0.2, "backoff_factor": 1.5},
            ValueError: {"max_retries": 2} # Will fail faster for ValueErrors
        }
    )
    try:
        print("\n--- Testing error-specific retry (ConnectionError) ---")
        result = await error_specific_retry.retry_async(flaky_operation, 7) # Fails 7 times
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    call_count = 0
    try:
        print("\n--- Testing error-specific retry (ValueError, fails fast) ---")
        result = await error_specific_retry.retry_async(flaky_operation, 3) # Fails 3 times
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())

