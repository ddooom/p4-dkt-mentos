import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%Y-%m-%d %X]", handlers=[RichHandler()])


def get_logger(name):
    return logging.getLogger(name)
