from datetime import datetime
from pathlib import Path

import pytest

from operations import JOB_NAME, PipelineOrchestrator


def test_manual_run_persists_result_and_exports_metrics(tmp_path: Path):
    orchestrator = PipelineOrchestrator(tmp_path / "operations.sqlite3")
    orchestrator.start()
    try:
        assert orchestrator.run_now() is True
        snapshot = orchestrator.snapshot()
        assert snapshot["runs"][0]["status"] == "success"
        assert snapshot["runs"][0]["records_processed"] == 600
        assert snapshot["runs"][0]["result"]["robots_total"] == 8
        metrics = orchestrator.prometheus_metrics()
        assert (
            'robotics_pipeline_runs_total{job="' + JOB_NAME + '",status="success"} 1'
        ) in metrics
        assert "robotics_scheduler_up 1" in metrics
        assert 'robotics_pipeline_last_records_processed{job="' + JOB_NAME + '"} 600' in metrics
    finally:
        orchestrator.stop()


def test_orchestrated_run_invokes_model_training_and_persists_evaluation(tmp_path: Path):
    training_inputs = []

    def train_model(content):
        training_inputs.append(content)
        return {
            "version": "test-model",
            "status": "promoted",
            "metrics": {"roc_auc": 1.0, "f1": 1.0},
        }

    orchestrator = PipelineOrchestrator(
        tmp_path / "operations.sqlite3",
        model_trainer=train_model,
    )

    assert orchestrator.run_now() is True
    result = orchestrator.snapshot()["runs"][0]["result"]

    assert len(training_inputs) == 1
    assert result["model_evaluation"]["version"] == "test-model"
    assert result["model_evaluation"]["status"] == "promoted"


def test_schedule_validation_and_disable_persist(tmp_path: Path):
    database = tmp_path / "operations.sqlite3"
    orchestrator = PipelineOrchestrator(database)

    with pytest.raises(ValueError, match="fréquence"):
        orchestrator.configure_schedule(10)
    with pytest.raises(ValueError, match="fréquence"):
        orchestrator.configure_schedule(True)

    configured = orchestrator.configure_schedule(60)
    assert configured["job"]["enabled"] == 1
    assert configured["job"]["interval_seconds"] == 60
    assert datetime.fromisoformat(configured["job"]["next_run_at"])

    disabled = orchestrator.disable_schedule()
    assert disabled["job"]["enabled"] == 0
    assert disabled["job"]["next_run_at"] is None
    assert PipelineOrchestrator(database).snapshot()["job"]["enabled"] == 0


def test_scheduler_executes_a_due_run_once(tmp_path: Path):
    orchestrator = PipelineOrchestrator(tmp_path / "operations.sqlite3")
    orchestrator.configure_schedule(60)
    with orchestrator._connect() as connection:
        connection.execute("UPDATE jobs SET next_run_at = 0 WHERE name = ?", (JOB_NAME,))

    assert orchestrator.run_due_once() is True
    assert orchestrator.run_due_once() is False
    assert orchestrator.snapshot()["runs"][0]["status"] == "success"


def test_run_retention_is_bounded(tmp_path: Path):
    orchestrator = PipelineOrchestrator(tmp_path / "operations.sqlite3")
    with orchestrator._connect() as connection:
        for index in range(510):
            connection.execute(
                """
                INSERT INTO runs(id, job_name, status, started_at, finished_at)
                VALUES (?, ?, 'success', ?, ?)
                """,
                (str(index), JOB_NAME, str(index).zfill(4), str(index).zfill(4)),
            )

    orchestrator.run_now()

    with orchestrator._connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM runs WHERE job_name = ?", (JOB_NAME,)).fetchone()[0]
    assert count == 500


def test_failed_pipeline_is_persisted_and_not_reported_as_success(tmp_path: Path, monkeypatch):
    orchestrator = PipelineOrchestrator(tmp_path / "operations.sqlite3")

    def fail_generation():
        raise RuntimeError("synthetic test failure")

    monkeypatch.setattr("operations.make_sample_telemetry_csv", fail_generation)

    assert orchestrator.run_now() is False
    run = orchestrator.snapshot()["runs"][0]
    assert run["status"] == "failed"
    assert run["error"] == "synthetic test failure"
