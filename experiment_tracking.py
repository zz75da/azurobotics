from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import platform
from typing import Any, Dict

LOG = logging.getLogger("robotics-demo.tracking")
EXPERIMENT_NAME = "robotics-telemetry-anomaly"
CHAMPION_ALIAS = "champion"


def tracking_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _user_tags(extra: Dict[str, Any] | None) -> Dict[str, str]:
    if not isinstance(extra, dict):
        return {}
    tags: Dict[str, str] = {}
    for key, value in list(extra.items())[:20]:
        key = str(key).strip()[:100]
        if key and all(c.isalnum() or c in "_-. /" for c in key):
            tags["user." + key] = str(value)[:500]
    return tags


def _environment_tags() -> Dict[str, str]:
    tags = {"python_version": platform.python_version(), "app_version": os.environ.get("APP_VERSION", "unknown")}
    for module in ("sklearn", "pandas", "numpy", "mlflow"):
        try:
            tags[module + "_version"] = importlib.metadata.version("scikit-learn" if module == "sklearn" else module)
        except importlib.metadata.PackageNotFoundError:
            pass
    return tags


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
    data_source: str = "unknown",
    hyperparameters: Dict[str, Any] | None = None,
    description: str | None = None,
    extra_tags: Dict[str, Any] | None = None,
    class_counts: Dict[int, int] | None = None,
    train_sample: Any = None,
) -> str | None:
    """Best-effort MLflow logging; a tracking failure must never break training."""
    if not tracking_enabled():
        return None
    try:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(EXPERIMENT_NAME)
        client = MlflowClient()
        status = "promoted" if promoted else "rejected"
        summary = description or "Isolation Forest, {} ({} rows), {}: {}".format(
            data_source, dataset_rows, status,
            ", ".join("{}={}".format(k, metrics[k]) for k in ("roc_auc", "f1") if k in metrics),
        )
        rerun_config = {
            "data_source": data_source,
            "dataset_sha256": dataset_digest,
            "dataset_rows": dataset_rows,
            "hyperparameters": hyperparameters or {},
            "features": params.get("features"),
            "promotion_thresholds": thresholds,
            "rerun": (
                "POST /api/ml/train with a JSON body {\"hyperparameters\": <hyperparameters>, "
                "\"description\": \"...\"}; for uploaded-csv, re-send the same CSV (check dataset_sha256)."
            ),
        }
        with mlflow.start_run(run_name=version, description=summary) as run:
            mlflow.log_params(params)
            mlflow.log_params({"dataset_sha256": dataset_digest[:16], "dataset_rows": dataset_rows, "data_source": data_source})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if class_counts:
                mlflow.log_params({"class_{}_rows".format(k): v for k, v in class_counts.items()})
            mlflow.set_tags({
                "model_version": version,
                "promotion_status": status,
                "promotion_thresholds": ",".join("{}>={}".format(k, v) for k, v in thresholds.items()),
                "data_source": data_source,
                "dataset_sha256": dataset_digest,
                "task": "telemetry-anomaly-detection",
                **_environment_tags(),
                **_user_tags(extra_tags),
            })
            mlflow.log_dict(rerun_config, "rerun_config.json")
            signature = input_example = None
            if train_sample is not None:
                try:
                    input_example = train_sample.head(5)
                    signature = infer_signature(input_example, model.predict(input_example))
                    mlflow.log_text(train_sample.to_csv(index=False), "data/train_features.csv")
                except Exception:
                    LOG.exception("mlflow_signature_failed")
                    signature = input_example = None
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name if promoted else None,
            )
        if promoted:
            registered = getattr(info, "registered_model_version", None)
            if registered is not None:
                version_number = str(registered)
                client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version_number)
                # Optional metadata: the 3.1.x file store can fail to serialise model versions that carry metrics.
                for call, args in (
                    (client.update_registered_model, (model_name, "Isolation Forest anomaly detector for robot telemetry (features: {}).".format(params.get("features")))),
                    (client.set_model_version_tag, (model_name, version_number, "description", summary[:500])),
                    (client.set_model_version_tag, (model_name, version_number, "dataset_sha256", dataset_digest[:16])),
                ):
                    try:
                        call(*args)
                    except Exception:
                        LOG.warning("mlflow_registry_metadata_skipped", exc_info=True)
        return run.info.run_id
    except Exception:
        LOG.exception("mlflow_logging_failed")
        return None


def digest_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
