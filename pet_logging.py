import logging
import sys


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    formatter: str = "%(name)s %(asctime)s %(message)s",
) -> logging.Logger:
    """
    Get or create a logger with the specified name and log level.

    Args:
        name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        formatter (str): The log message format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return logger
