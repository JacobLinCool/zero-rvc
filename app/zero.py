import os
import logging

logger = logging.getLogger(__name__)

zero_is_available = "SPACES_ZERO_GPU" in os.environ

if zero_is_available:
    import spaces  # type: ignore

    logger.info("ZeroGPU is available")
else:
    logger.info("ZeroGPU is not available")


# a decorator that applies the spaces.GPU decorator if zero is available
def zero(duration=60):
    def wrapper(func):
        if zero_is_available:
            return spaces.GPU(func, duration=duration)
        else:
            return func

    return wrapper
