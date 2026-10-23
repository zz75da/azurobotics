from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from robot_data import analyze_text, make_sample_telemetry_csv


JOB_NAME = "sample-telemetry-analysis"
MIN_INTERVAL_SECONDS = 60
MAX_INTERVAL_SECONDS = 86_400
RUN_RETENTION = 500
LOG = logging.getLogger("robotics-demo.orchestration")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineOrchestrator:
    def __init__(
        self,
        database_path: Path,
        model_trainer: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        self.database_path = database_path
        self.model_trainer = model_trainer
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    name TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    interval_seconds INTEGER NOT NULL DEFAULT 300,
                    next_run_at REAL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scheduled_at TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL,
                    records_processed INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_job_started ON runs(job_name, started_at DESC)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO jobs(name, enabled, interval_seconds, updated_at) VALUES (?, 0, 300, ?)",
                (JOB_NAME, utc_now()),
            )
            connection.execute(
                """
                UPDATE runs SET status = 'failed', finished_at = ?, error = ?
                WHERE status = 'running'
                """,
                (utc_now(), "Arrêt inattendu du processus pendant l'exécution."),
            )
            connection.execute(
                "DELETE FROM runs WHERE id IN (SELECT id FROM runs ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
                (RUN_RETENTION,),
            )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, name="pipeline-scheduler", daemon=True)
        self._thread.start()
        LOG.info("scheduler_started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        LOG.info("scheduler_stopped")

    def _scheduler_loop(self) -> None:
        while not self._stop_event.wait(1):
            try:
                self.run_due_once()
            except Exception:
                LOG.exception("scheduler_tick_failed")

    def configure_schedule(self, interval_seconds: int) -> Dict[str, Any]:
        if isinstance(interval_seconds, bool) or not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS:
            raise ValueError(
                "La fréquence doit être comprise entre {} et {} secondes.".format(
                    MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS
                )
            )
        next_run = time.time() + interval_seconds
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET enabled = 1, interval_seconds = ?, next_run_at = ?, updated_at = ?
                WHERE name = ?
                """,
                (interval_seconds, next_run, utc_now(), JOB_NAME),
            )
        LOG.info(
            "schedule_enabled",
            extra={"job_name": JOB_NAME, "interval_seconds": interval_seconds},
        )
        return self.snapshot()

    def disable_schedule(self) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET enabled = 0, next_run_at = NULL, updated_at = ? WHERE name = ?",
                (utc_now(), JOB_NAME),
            )
        LOG.info("schedule_disabled", extra={"job_name": JOB_NAME})
        return self.snapshot()

    def run_due_once(self) -> bool:
        run_id = self._claim_run(force=False)
        if run_id is None:
            return False
        self._execute(run_id)
        return True

    def run_now(self) -> Optional[bool]:
        run_id = self._claim_run(force=True)
        if run_id is None:
            return None
        return self._execute(run_id)

    def _claim_run(self, force: bool) -> Optional[str]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE name = ?", (JOB_NAME,)).fetchone()
            if job is None:
                raise RuntimeError("Le pipeline local n'est pas configuré.")
            active = connection.execute(
                "SELECT 1 FROM runs WHERE job_name = ? AND status = 'running' LIMIT 1", (JOB_NAME,)
            ).fetchone()
            if active:
                return None
            if not force and (not job["enabled"] or job["next_run_at"] is None or job["next_run_at"] > now):
                return None
            run_id = uuid.uuid4().hex
            scheduled_at = utc_now() if not force else None
            connection.execute(
                """
                INSERT INTO runs(id, job_name, status, scheduled_at, started_at)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, JOB_NAME, scheduled_at, utc_now()),
            )
            if job["enabled"]:
                connection.execute(
                    "UPDATE jobs SET next_run_at = ?, updated_at = ? WHERE name = ?",
                    (now + job["interval_seconds"], utc_now(), JOB_NAME),
                )
            connection.execute(
                "DELETE FROM runs WHERE id IN (SELECT id FROM runs WHERE status != 'running' ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
                (RUN_RETENTION - 1,),
            )
            return run_id

    def _execute(self, run_id: str) -> bool:
        started = time.monotonic()
        try:
            content = make_sample_telemetry_csv()
            result = analyze_text(content, ".csv")
            model_result = self.model_trainer(content) if self.model_trainer else None
            summary = _summary(result)
            if model_result is not None:
                summary["model_evaluation"] = {
                    "version": model_result["version"],
                    "status": model_result["status"],
                    "metrics": model_result["metrics"],
                }
            duration = time.monotonic() - started
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE runs SET status = 'success', finished_at = ?, duration_seconds = ?,
                        records_processed = ?, result_json = ? WHERE id = ?
                    """,
                    (
                        utc_now(),
                        duration,
                        result["events_total"],
                        json.dumps(summary, ensure_ascii=False),
                        run_id,
                    ),
                )
            LOG.info(
                "pipeline_run_succeeded",
                extra={
                    "job_name": JOB_NAME,
                    "run_id": run_id,
                    "duration_seconds": round(duration, 4),
                    "records_processed": result["events_total"],
                },
            )
            return True
        except Exception as error:
            duration = time.monotonic() - started
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE runs SET status = 'failed', finished_at = ?, duration_seconds = ?, error = ?
                    WHERE id = ?
                    """,
                    (utc_now(), duration, str(error)[:500], run_id),
                )
            LOG.exception(
                "pipeline_run_failed",
                extra={"job_name": JOB_NAME, "run_id": run_id, "duration_seconds": round(duration, 4)},
            )
            return False

    def snapshot(self, run_limit: int = 20) -> Dict[str, Any]:
        with self._connect() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE name = ?", (JOB_NAME,)).fetchone()
            runs = connection.execute(
                "SELECT * FROM runs WHERE job_name = ? ORDER BY started_at DESC LIMIT ?",
                (JOB_NAME, run_limit),
            ).fetchall()
            counts = connection.execute(
                "SELECT status, COUNT(*) AS total FROM runs WHERE job_name = ? GROUP BY status",
                (JOB_NAME,),
            ).fetchall()
        job_value = dict(job)
        next_run = job_value.pop("next_run_at")
        job_value["next_run_at"] = datetime.fromtimestamp(next_run, timezone.utc).isoformat() if next_run else None
        run_values: List[Dict[str, Any]] = []
        for run in runs:
            value = dict(run)
            value["result"] = json.loads(value.pop("result_json")) if value.get("result_json") else None
            run_values.append(value)
        return {
            "job": job_value,
            "runs": run_values,
            "run_counts": {row["status"]: row["total"] for row in counts},
            "scheduler_running": bool(self._thread and self._thread.is_alive()),
        }

    def prometheus_metrics(self) -> str:
        snapshot = self.snapshot(run_limit=RUN_RETENTION)
        job = snapshot["job"]
        lines = [
            "# HELP robotics_pipeline_runs_total Completed and failed local pipeline runs.",
            "# TYPE robotics_pipeline_runs_total counter",
        ]
        for status in ("success", "failed"):
            lines.append(
                'robotics_pipeline_runs_total{{job="{}",status="{}"}} {}'.format(
                    JOB_NAME, status, snapshot["run_counts"].get(status, 0)
                )
            )
        last_success = next(
            (run for run in snapshot["runs"] if run["status"] == "success" and run["finished_at"]),
            None,
        )
        last_run = snapshot["runs"][0] if snapshot["runs"] else None
        success_timestamp = (
            datetime.fromisoformat(last_success["finished_at"]).timestamp() if last_success else 0
        )
        active = int(any(run["status"] == "running" for run in snapshot["runs"]))
        lines.extend(
            [
                "# HELP robotics_pipeline_last_success_timestamp_seconds Unix timestamp of the last successful pipeline run.",
                "# TYPE robotics_pipeline_last_success_timestamp_seconds gauge",
                'robotics_pipeline_last_success_timestamp_seconds{{job="{}"}} {}'.format(
                    JOB_NAME, success_timestamp
                ),
                "# HELP robotics_pipeline_running Whether the pipeline is currently running.",
                "# TYPE robotics_pipeline_running gauge",
                'robotics_pipeline_running{{job="{}"}} {}'.format(JOB_NAME, active),
                "# HELP robotics_scheduler_up Whether the local scheduler thread is alive.",
                "# TYPE robotics_scheduler_up gauge",
                "robotics_scheduler_up {}".format(int(snapshot["scheduler_running"])),
                "# HELP robotics_schedule_enabled Whether periodic scheduling is enabled.",
                "# TYPE robotics_schedule_enabled gauge",
                "robotics_schedule_enabled {}".format(int(job["enabled"])),
                "# HELP robotics_pipeline_last_run_duration_seconds Duration of the most recent run in seconds.",
                "# TYPE robotics_pipeline_last_run_duration_seconds gauge",
                'robotics_pipeline_last_run_duration_seconds{{job="{}"}} {}'.format(
                    JOB_NAME, last_run["duration_seconds"] if last_run and last_run["duration_seconds"] is not None else 0
                ),
                "# HELP robotics_pipeline_last_records_processed Number of records processed by the most recent run.",
                "# TYPE robotics_pipeline_last_records_processed gauge",
                'robotics_pipeline_last_records_processed{{job="{}"}} {}'.format(
                    JOB_NAME, last_run["records_processed"] if last_run else 0
                ),
            ]
        )
        return "\n".join(lines) + "\n"


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "events_total": result["events_total"],
        "robots_total": result["robots_total"],
        "errors_total": result["errors_total"],
        "error_rate_pct": result["error_rate_pct"],
        "battery_avg_pct": result["battery_avg_pct"],
        "duration_avg_ms": result["duration_avg_ms"],
        "quality_issues": result["quality_issues"],
    }
