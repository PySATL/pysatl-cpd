import logging
from functools import wraps
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import Any, Callable


def setup_logger(rewrite_logs: bool = True) -> Logger:
    """
    Initialize and configure a logger with console and file handlers.

    Creates a logger with:
    - Two file handlers (DEBUG and INFO levels)
    - One console handler (INFO level)
    - Configurable log rotation/rewrite behavior

    :param rewrite_logs: If True, log files will be rewritten on each run.
                         If False, enables log rotation (max 3 backups, 5MB each).
    :return: Configured logger instance with name 'cpd_logger'

    :note: Log files are created in 'new_pysatl_cpd/execution_logs/' directory
    :note: Debug logs go to 'pysatl_cpd_debug.log'
    :note: Info logs go to 'pysatl_cpd_info.log'
    :note: Console output shows INFO level and above

    :example:
        >>> logger = setup_logger(rewrite_logs=False)
        >>> logger.info("Test message")

    .. rubric:: Log Format Details

    File and console output uses format::

        %(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s

    With date format::

        %Y-%m-%d %H:%M:%S
    """
    path = "new_pysatl_cpd/execution_logs/"

    logger = logging.getLogger("cpd_logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    settings: dict[str, Any] = (
        {"encoding": "utf-8", "mode": "w"}
        if rewrite_logs
        else {"maxBytes": 5 * 1024 * 1024, "backupCount": 3, "encoding": "utf-8"}
    )

    file_handler_debug = RotatingFileHandler(f"{path}pysatl_cpd_debug.log", **settings)
    file_handler_debug.setLevel(logging.DEBUG)
    file_handler_debug.setFormatter(formatter)

    file_handler_info = RotatingFileHandler(f"{path}pysatl_cpd_info.log", **settings)
    file_handler_info.setLevel(logging.INFO)
    file_handler_info.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler_info)

    return logger


def log_exceptions(func: Callable[[Any], Any]) -> Any:
    """
    Decorator to log exceptions occurring in wrapped functions.

    This decorator catches any exceptions raised by the decorated function,
    logs them with traceback information using the global 'cpd_logger',
    and re-raises the exception.

    :param func: The function to be decorated
    :return: The wrapped function

    :raises Exception: Re-raises any exception that occurs in the decorated function

    :note: Uses the global logger named 'cpd_logger'
    :note: Preserves original function metadata using functools.wraps

    :example:
        >>> @log_exceptions
        ... def risky_operation():
        ...     return 1 / 0
        >>> try:
        ...     risky_operation()
        ... except ZeroDivisionError:
        ...     print("Caught exception")  # Exception will be logged first
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            cpd_logger.exception(f"Exception in function {func.__name__}: {e}")
            raise

    return wrapper


cpd_logger = setup_logger()
