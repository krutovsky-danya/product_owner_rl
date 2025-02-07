import logging
import sys

_logger = None


def get_logger(name: str):
    global _logger
    if _logger is not None:
        return _logger
    _logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)

    return _logger
