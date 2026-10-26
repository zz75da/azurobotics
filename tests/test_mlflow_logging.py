import mlflow
import pytest
from mlflow import MlflowClient

from experiment_tracking import CHAMPION_ALIAS, EXPERIMENT_NAME
from mlops import MLOpsService


@pytest.mark.parametrize("algorithm", ["isolation_forest", "random_forest"])
def test_training_is_logged_to_a_real_mlflow_file_store(tmp_path, monkeypatch, algorithm):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    service = MLOpsService(tmp_path / "mlops.sqlite3", tmp_path / "models", tmp_path / "evidently")

    result = service.train(None, hyperparameters={"algorithm": algorithm}, tuning_results=[{"trial": 1}])

    run_id = result["mlflow_run_id"]
    assert run_id, "MLflow logging failed silently"
    client = MlflowClient(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["algorithm"]
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert run.info.experiment_id == experiment.experiment_id
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert {"MODEL_CARD.md", "rerun_config.json"} <= artifacts
    if result["status"] == "promoted":
        assert client.get_model_version_by_alias("robot-telemetry-isolation-forest", CHAMPION_ALIAS)
    mlflow.set_tracking_uri(tracking_uri)
    assert mlflow.sklearn.load_model("runs:/{}/model".format(run_id)) is not None
