import json
import logging
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict

TEXT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


class JsonLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        # Include extra fields passed via the `extra` dict
        for key in ("tool_name", "duration_s"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/agno_agent_api.log",
    log_format: str = "text",
) -> None:
    """Configure application-wide logging with console and rotating file handlers."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(TEXT_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s, file=%s, format=%s",
        log_level,
        log_path,
        log_format,
    )


def logger_hook(
    function_name: str, function_call: Callable, arguments: Dict[str, Any]
) -> Any:
    """Agno tool hook that logs function call duration and details."""
    hook_logger = logging.getLogger("app.tools")
    start_time = time.time()

    result = function_call(**arguments)

    duration = time.time() - start_time
    hook_logger.info(
        "Tool %s executed in %.2fs | args=%s",
        function_name,
        duration,
        str(arguments)[:500],
        extra={"tool_name": function_name, "duration_s": round(duration, 3)},
    )
    hook_logger.debug("Tool %s returned: %s", function_name, str(result)[:1000])

    return result
