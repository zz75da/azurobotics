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


def test_evidently_dashboard_is_created_once_and_keeps_manual_panels(tmp_path):
    def open_service():
        return MLOpsService(tmp_path / "m.sqlite3", tmp_path / "models", tmp_path / "ev")

    service = open_service()
    dashboard = service._evidently_project.dashboard
    assert len(dashboard.model().panels) == 4

    dashboard.delete_panel(dashboard.model().panels[0].title, dashboard.model().tabs[0].title)
    remaining = len(dashboard.model().panels)
    assert len(open_service()._evidently_project.dashboard.model().panels) == remaining


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


def _drifted_csv(rows=200):
    lines = ["timestamp,robot_id,event,duration_ms,battery_pct,x_m,y_m"]
    lines += ["t,r,move,{},{},{},{}".format(500_000 + i, 5_000 + i, 9_000 + i, 9_000 + i) for i in range(rows)]
    return "\n".join(lines) + "\n"


def _trained_service(tmp_path):
    service = MLOpsService(tmp_path / "m.sqlite3", tmp_path / "models", tmp_path / "ev")
    service.train()
    return service


def test_drift_dispatches_retrain_once_per_cooldown(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("mlops.dispatch_retrain", lambda share, snapshot: calls.append(share) or "dispatched")
    service = _trained_service(tmp_path)
    first = service.predict(_drifted_csv())
    second = service.predict(_drifted_csv())
    assert first["drift_detected"] and first["drift_share"] >= 0.5 and first["retrain"] == "dispatched"
    assert second["retrain"] == "cooldown" and len(calls) == 1
    assert "robotics_ml_retrain_triggers_total 1" in service.prometheus_metrics()


def test_failed_dispatch_does_not_start_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops.dispatch_retrain", lambda share, snapshot: "failed")
    service = _trained_service(tmp_path)
    assert service.predict(_drifted_csv())["retrain"] == "failed"
    assert service.predict(_drifted_csv())["retrain"] == "failed"


def test_small_or_stable_batches_do_not_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops.dispatch_retrain", lambda share, snapshot: pytest.fail("must not dispatch"))
    service = _trained_service(tmp_path)
    assert service.predict(_drifted_csv(rows=20))["retrain"] == "not_needed"
    assert service.predict(make_sample_telemetry_csv())["retrain"] == "not_needed"


def test_retrain_trigger_is_disabled_without_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_RETRAIN_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_RETRAIN_TOKEN", raising=False)
    assert _trained_service(tmp_path).predict(_drifted_csv())["retrain"] == "disabled"


def test_dispatch_posts_workflow_inputs(monkeypatch):
    import json

    import retrain_trigger

    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        seen.update(url=request.full_url, body=json.loads(request.data), auth=request.get_header("Authorization"))
        return Response()

    monkeypatch.setenv("GITHUB_RETRAIN_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RETRAIN_TOKEN", "secret")
    monkeypatch.setattr(retrain_trigger.urllib.request, "urlopen", fake_urlopen)
    assert retrain_trigger.dispatch_retrain(0.75, "snap") == "dispatched"
    assert seen["url"].endswith("/repos/owner/repo/actions/workflows/retrain.yml/dispatches")
    assert seen["body"]["ref"] == "main" and "0.75" in seen["body"]["inputs"]["description"]
    assert seen["auth"] == "Bearer secret"


def test_dispatch_reports_network_failure(monkeypatch):
    import retrain_trigger

    def fail(request, timeout):
        raise retrain_trigger.urllib.error.URLError("down")

    monkeypatch.setenv("GITHUB_RETRAIN_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RETRAIN_TOKEN", "secret")
    monkeypatch.setattr(retrain_trigger.urllib.request, "urlopen", fail)
    assert retrain_trigger.dispatch_retrain(0.75, "snap") == "failed"
