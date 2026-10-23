from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + ".{:03d}Z".format(int(record.msecs)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in ("request_id", "method", "route", "status", "duration_seconds", "job_name", "run_id", "interval_seconds", "records_processed", "model_version", "model_status"):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False)


def configure_logging(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "app.jsonl"
    logger = logging.getLogger("robotics-demo")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_path


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[Tuple[str, str, int], int] = defaultdict(int)
        self._durations: Dict[Tuple[str, str, int], float] = defaultdict(float)
        self._duration_counts: Dict[Tuple[str, str, int, float], int] = defaultdict(int)
        self._duration_bounds = (0.05, 0.1, 0.5, 1.0, 5.0, float("inf"))

    def observe(self, method: str, route: str, status: int, duration_seconds: float) -> None:
        key = (method, route, status)
        with self._lock:
            self._counts[key] += 1
            self._durations[key] += duration_seconds
            for bound in self._duration_bounds:
                if duration_seconds <= bound:
                    self._duration_counts[key + (bound,)] += 1

    def prometheus_metrics(self) -> str:
        with self._lock:
            counts = dict(self._counts)
            durations = dict(self._durations)
            duration_counts = dict(self._duration_counts)
        lines = [
            "# HELP robotics_http_requests_total Total HTTP requests.",
            "# TYPE robotics_http_requests_total counter",
        ]
        for (method, route, status), count in sorted(counts.items()):
            lines.append(
                'robotics_http_requests_total{{method="{}",route="{}",status="{}"}} {}'.format(
                    method, route, status, count
                )
            )
        lines.extend(
            [
                "# HELP robotics_http_request_duration_seconds HTTP request duration in seconds.",
                "# TYPE robotics_http_request_duration_seconds histogram",
            ]
        )
        for (method, route, status), _ in sorted(counts.items()):
            for bound in self._duration_bounds:
                bound_value = "+Inf" if bound == float("inf") else str(bound)
                bucket_count = duration_counts.get((method, route, status, bound), 0)
                lines.append(
                    'robotics_http_request_duration_seconds_bucket{{method="{}",route="{}",status="{}",le="{}"}} {}'.format(
                        method, route, status, bound_value, bucket_count
                    )
                )
        for (method, route, status), duration in sorted(durations.items()):
            lines.append(
                'robotics_http_request_duration_seconds_sum{{method="{}",route="{}",status="{}"}} {:.6f}'.format(
                    method, route, status, duration
                )
            )
            lines.append(
                'robotics_http_request_duration_seconds_count{{method="{}",route="{}",status="{}"}} {}'.format(
                    method, route, status, counts[(method, route, status)]
                )
            )
        return "\n".join(lines) + "\n"


def read_recent_logs(path: Path, limit: int = 50) -> list:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events[-limit:]
