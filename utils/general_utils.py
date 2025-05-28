# utils/general_utils.py
import logging
import sys
import time

def setup_logger(name="DanzarVLM", level_str="INFO"):
    # Basic logger setup, can be expanded
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers if called more than once
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class RateLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.call_timestamps = []

    def allow_call(self) -> bool:
        now = time.time()
        # Remove timestamps older than the period
        self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < self.period_seconds]

        if len(self.call_timestamps) < self.max_calls:
            self.call_timestamps.append(now)
            return True
        return False

if __name__ == '__main__':
    logger = setup_logger(level_str="DEBUG")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")

    limiter = RateLimiter(max_calls=2, period_seconds=5)
    for i in range(5):
        if limiter.allow_call():
            print(f"Call {i+1} allowed at {time.time()}")
        else:
            print(f"Call {i+1} rate limited at {time.time()}")
        time.sleep(1)
