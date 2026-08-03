from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(message)s",
    )
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(
        json.dumps(
            {"event": event, "timestamp": datetime.now(UTC).isoformat(), **fields},
            default=str,
        )
    )
