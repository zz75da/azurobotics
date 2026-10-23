import pytest

from mlops import MLOpsService
from robot_data import make_sample_telemetry_csv


def test_training_prediction_and_evidently_report_are_persisted(tmp_path):
    service = MLOpsService(
        tmp_path / "mlops.sqlite3",
        tmp_path / "models",
        tmp_path / "evidently",
    )

    trained = service.train()

    assert trained["status"] == "promoted"
    assert trained["metrics"]["roc_auc"] >= 0.75
    assert trained["metrics"]["f1"] >= 0.5
    assert (tmp_path / "models" / (trained["version"] + ".joblib")).is_file()

    prediction = service.predict(make_sample_telemetry_csv())
    status = service.status()

    assert prediction["records_processed"] == 600
    assert prediction["model_version"] == trained["version"]
    assert len(prediction["predictions"]) == 100
    assert prediction["predictions_truncated"] is True
    assert len(status["prediction_batches"]) == 1
    assert "robotics_ml_model_promoted 1" in service.prometheus_metrics()
    assert "robotics_ml_drift_reports_total 2" in service.prometheus_metrics()
    run_ids = {
        str(run_id) for run_id in service._workspace.list_runs(service._evidently_project.id)
    }
    assert trained["evidently_snapshot_id"] in run_ids
    assert prediction["evidently_snapshot_id"] in run_ids


def test_training_rejects_csv_without_binary_labels(tmp_path):
    service = MLOpsService(
        tmp_path / "mlops.sqlite3",
        tmp_path / "models",
        tmp_path / "evidently",
    )
    csv_content = (
        "timestamp,robot_id,event,duration_ms,battery_pct,x_m,y_m\n"
        "2026-01-01T00:00:00Z,R1,move,1,80,2,3\n"
    )

    with pytest.raises(ValueError, match="is_anomaly"):
        service.train(csv_content)


def test_rejected_candidate_keeps_current_model_promoted(tmp_path, monkeypatch):
    service = MLOpsService(
        tmp_path / "mlops.sqlite3",
        tmp_path / "models",
        tmp_path / "evidently",
    )
    promoted = service.train()
    monkeypatch.setattr("mlops.PROMOTION_THRESHOLDS", {"roc_auc": 1.1, "f1": 0.5})

    rejected = service.train()

    assert promoted["status"] == "promoted"
    assert rejected["status"] == "rejected"
    assert service.status()["active_model"]["version"] == promoted["version"]
