import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def get_logger(name):
    return logging.getLogger(name)
