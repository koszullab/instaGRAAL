#!/usr/bin/env python3

"""Basic logging setup for instaGRAAL.

Logging level can be set by the user and determines the verbosity of the
whole program.
"""

import datetime
import logging
from logging.handlers import RotatingFileHandler

# Log level is modified by other modules according to the entry (user set)
# value
CURRENT_LOG_LEVEL = logging.DEBUG

logging.captureWarnings(True)

# Use a named logger to avoid hijacking the root logger and interfering
# with third-party libraries that also emit log records.
logger = logging.getLogger("instagraal")
logger.setLevel(logging.DEBUG)

logfile_path = f"instagraal-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logfile_formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
stdout_formatter = logging.Formatter("%(levelname)s :: %(message)s")

# Rotate the log file at 10 MB, keeping up to 5 backups, so the file
# never grows unboundedly across long runs.
file_handler = RotatingFileHandler(
    logfile_path,
    mode="a",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logfile_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(stdout_formatter)
logger.addHandler(stream_handler)
