import json

import pytest

import mlops
from mlops import MLOpsService, RandomForestAnomalyDetector
from robot_data import analyze_text, make_sample_telemetry_csv


def _service(tmp_path):
    return MLOpsService(tmp_path / "m.sqlite3", tmp_path / "models", tmp_path / "ev")


def test_default_algorithm_stays_isolation_forest(tmp_path):
    service = _service(tmp_path)
    result = service.train()
    assert result["status"] == "promoted"
    model = mlops.joblib.load(tmp_path / "models" / (result["version"] + ".joblib"))["model"]
    assert type(model).__name__ == "IsolationForest"


def test_random_forest_trains_promotes_and_predicts(tmp_path):
    service = _service(tmp_path)
    result = service.train(hyperparameters={"algorithm": "random_forest"})
    assert result["status"] == "promoted"
    assert result["metrics"]["f1"] >= 0.5
    model = mlops.joblib.load(tmp_path / "models" / (result["version"] + ".joblib"))["model"]
    assert isinstance(model, RandomForestAnomalyDetector)
    prediction = service.predict(make_sample_telemetry_csv())
    assert prediction["records_processed"] == 600
    assert prediction["anomalies_detected"] > 0


def test_auto_picks_best_candidate_and_records_both(tmp_path):
    result = _service(tmp_path).train(hyperparameters={"algorithm": "auto"})
    assert result["status"] == "promoted"


def test_tuned_random_forest_settings_are_applied(tmp_path):
    service = _service(tmp_path)
    result = service.train(
        hyperparameters={
            "algorithm": "random_forest",
            "max_depth": 8,
            "min_samples_leaf": 5,
            "class_weight": None,
            "max_features": 0.5,
            "decision_threshold": 0.32,
        },
        tuning_results=[{"cv_f1": 0.5, "n_estimators": 10}],
    )
    model = mlops.joblib.load(tmp_path / "models" / (result["version"] + ".joblib"))["model"]
    assert model.max_depth == 8 and model.min_samples_leaf == 5 and model.class_weight is None
    assert model.decision_threshold == 0.32


@pytest.mark.parametrize(
    "bad",
    [{"max_depth": 0}, {"min_samples_leaf": 0}, {"class_weight": "x"}, {"decision_threshold": 1.5}],
)
def test_invalid_tuned_settings_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        _service(tmp_path).train(hyperparameters={"algorithm": "random_forest", **bad})


def test_unknown_algorithm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="algorithm"):
        _service(tmp_path).train(hyperparameters={"algorithm": "xgboost"})


def test_incident_groups_and_kpis_from_jsonl():
    lines = [
        {"timestamp": "2026-01-01T00:00:00Z", "robot_id": "R1", "code": "E-BATT", "severity": "error"},
        {"timestamp": "2026-01-01T01:00:00Z", "robot_id": "R2", "code": "E-BATT", "severity": "error"},
        {"timestamp": "2026-01-01T03:00:00Z", "robot_id": "R1", "code": "E-JAM", "severity": "warning"},
        {"timestamp": "2026-01-02T00:00:00Z", "robot_id": "R1", "event": "move"},
    ]
    result = analyze_text("\n".join(json.dumps(line) for line in lines), ".jsonl")
    top = result["incident_groups"][0]
    assert top["key"] == "E-BATT" and top["count"] == 2 and top["robots_affected"] == 2
    assert result["kpis"]["mean_time_between_incidents_min"] == 90.0
    assert result["kpis"]["worst_robots"][0]["robot_id"] == "R2"
    assert [d["day"] for d in result["kpis"]["daily"]] == ["2026-01-01", "2026-01-02"]


def test_kpis_tolerate_missing_and_invalid_fields():
    result = analyze_text("event,timestamp\nerror,not-a-date\nmove,\n", ".csv")
    assert result["incident_groups"][0]["count"] == 1
    assert result["kpis"]["mean_time_between_incidents_min"] is None
    assert result["kpis"]["daily"] == []
