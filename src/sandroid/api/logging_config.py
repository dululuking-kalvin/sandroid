"""Process-wide logging configuration.

Plain text by default (developer ergonomic). Set ``SANDROID_LOG_FORMAT=json``
to switch to JSON lines suitable for ELK / Loki ingestion. ``SANDROID_LOG_LEVEL``
sets the root level (default INFO).

Called exactly once from ``create_app``. Safe to call again — handlers are
replaced, not appended, so reloads don't stack duplicate lines.
"""

from __future__ import annotations

import logging
import os
import sys

from pythonjsonlogger.json import JsonFormatter

LOG_FORMAT_ENV = "SANDROID_LOG_FORMAT"  # "plain" (default) | "json"
LOG_LEVEL_ENV = "SANDROID_LOG_LEVEL"

_PLAIN_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_JSON_FIELDS = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = os.environ.get(LOG_FORMAT_ENV, "plain").lower()
    handler = logging.StreamHandler(stream=sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter(_JSON_FIELDS))
    else:
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Tame the noisy children — uvicorn.access spams per request, we have
    # /metrics instead. Errors still come through.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
