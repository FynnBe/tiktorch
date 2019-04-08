import multiprocessing as mp
import logging.config
import pprint

CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "%(asctime)s.%(msecs)03d [%(processName)s/%(threadName)s] %(levelname)s %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        }
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO", "propagate": True},
        "tiktorch": {"handlers": ["default"], "level": "DEBUG", "propagate": False},
    },
}
logging.config.dictConfig(CONFIG)


def configure(queue: mp.Queue) -> None:
    config = {
        **CONFIG,
        "handlers": {"default": {"level": "DEBUG", "class": "logging.handlers.QueueHandler", "queue": queue}},
    }
    logging.config.dictConfig(config)
