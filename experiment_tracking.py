from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict

LOG = logging.getLogger("robotics-demo.tracking")
EXPERIMENT_NAME = "robotics-telemetry-anomaly"
CHAMPION_ALIAS = "champion"


def tracking_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def log_training_run(
    *,
    version: str,
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    promoted: bool,
    dataset_digest: str,
    dataset_rows: int,
    thresholds: Dict[str, float],
) -> str | None:
    """Best-effort MLflow logging; a tracking failure must never break training."""
    if not tracking_enabled():
        return None
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=version) as run:
            mlflow.log_params(params)
            mlflow.log_params({"dataset_sha256": dataset_digest[:16], "dataset_rows": dataset_rows})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.set_tags({
                "model_version": version,
                "promotion_status": "promoted" if promoted else "rejected",
                "promotion_thresholds": ",".join("{}>={}".format(k, v) for k, v in thresholds.items()),
            })
            info = mlflow.sklearn.log_model(
                model, name="model", registered_model_name=model_name if promoted else None
            )
        if promoted:
            client = MlflowClient()
            registered = getattr(info, "registered_model_version", None)
            if registered is not None:
                client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(registered))
        return run.info.run_id
    except Exception:
        LOG.exception("mlflow_logging_failed")
        return None


def digest_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
