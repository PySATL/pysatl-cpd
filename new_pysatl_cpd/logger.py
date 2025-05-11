import logging
from functools import wraps
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import Any, Callable


def setup_logger() -> Logger:
    logger = logging.getLogger("cpd_logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler_debug = RotatingFileHandler(
        "pysatl_cpd_debug.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler_debug.setLevel(logging.DEBUG)
    file_handler_debug.setFormatter(formatter)

    file_handler_info = RotatingFileHandler(
        "pysatl_cpd_info.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler_info.setLevel(logging.INFO)
    file_handler_info.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler_info)

    return logger


def log_exceptions(func: Callable[[Any], Any]) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in function {func.__name__}: {e}")
            raise

    return wrapper


logger = setup_logger()
