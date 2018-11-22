#!/usr/bin/env python3

"""Basic logging setup for instaGRAAL.

Logging level can be set by the user and determines the verbosity of the
whole program.
"""


import logging

from logging import FileHandler

# Log level is modified by other modules according to the entry (user set)
# value
CURRENT_LOG_LEVEL = logging.DEBUG

logging.captureWarnings(True)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logfile_formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(message)s"
)
stdout_formatter = logging.Formatter("%(levelname)s :: %(message)s")

file_handler = FileHandler("instagraal.log", "a")

file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logfile_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(stdout_formatter)
logger.addHandler(stream_handler)
