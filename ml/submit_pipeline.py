"""Submit the AI4I training pipeline to Azure ML and register the model if the quality gate passes.

prep -> train -> evaluate run on serverless compute (billed only while running). evaluate fails the
pipeline when F1 is below the threshold, so a model is registered only after a passing run, with
the dataset SHA-256, metrics and git commit as tags. Signs in with DefaultAzureCredential: the az
CLI login on a workstation, OIDC in GitHub Actions.

    python ml/submit_pipeline.py --workspace mlw-robotics-dev-99ff [--raw ai4i2020.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from azure.ai.ml import Input, MLClient, Output, command, dsl  # noqa: E402
from azure.ai.ml.constants import AssetTypes  # noqa: E402
from azure.ai.ml.entities import Environment, Model  # noqa: E402
from azure.identity import DefaultAzureCredential  # noqa: E402

MODEL_NAME = "robot-telemetry-anomaly"
ENVIRONMENT_NAME = "robotics-train"
EXPERIMENT_NAME = "robotics-ai4i-pipeline"
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"


def _git_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _raw_dataset(path: str | None) -> Path:
    if path:
        return Path(path)
    from tools.convert_ai4i import load_source

    target = Path(tempfile.mkdtemp()) / "ai4i2020.csv"
    target.write_text(load_source(None), encoding="utf-8", newline="")
    return target


def build_pipeline(environment: str, instance_type: str, f1_threshold: float, raw: Path):
    prep_step = command(
        name="prep",
        display_name="prep: AI4I to telemetry schema, train/test split",
        code=str(REPO),
        command="python ml/components/prep.py --raw ${{inputs.raw}} --output ${{outputs.prepared}}",
        inputs={"raw": Input(type=AssetTypes.URI_FILE)},
        outputs={"prepared": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
    )
    train_step = command(
        name="train",
        display_name="train: tuned AI4I random forest (mlops.py)",
        code=str(REPO),
        command="python ml/components/train.py --prepared ${{inputs.prepared}} --model-dir ${{outputs.model_dir}}",
        inputs={"prepared": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"model_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
    )
    evaluate_step = command(
        name="evaluate",
        display_name="evaluate: held-out F1 quality gate",
        code=str(REPO),
        command=(
            "python ml/components/evaluate.py --model-dir ${{inputs.model_dir}} --prepared ${{inputs.prepared}} "
            "--metrics-out ${{outputs.metrics}} --f1-threshold ${{inputs.f1_threshold}}"
        ),
        inputs={
            "model_dir": Input(type=AssetTypes.URI_FOLDER),
            "prepared": Input(type=AssetTypes.URI_FOLDER),
            "f1_threshold": f1_threshold,
        },
        outputs={"metrics": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
    )

    @dsl.pipeline(name="ai4i_anomaly_training", compute="serverless", description="AI4I 2020: prep -> train -> evaluate (F1 gate)")
    def pipeline(raw_csv):
        prepared = prep_step(raw=raw_csv)
        trained = train_step(prepared=prepared.outputs.prepared)
        evaluated = evaluate_step(model_dir=trained.outputs.model_dir, prepared=prepared.outputs.prepared)
        # Step names come from the last variable bound to each node, so no loop variable here.
        prepared.resources = {"instance_type": instance_type, "instance_count": 1}
        trained.resources = {"instance_type": instance_type, "instance_count": 1}
        evaluated.resources = {"instance_type": instance_type, "instance_count": 1}
        return {
            "dataset": prepared.outputs.prepared,
            "model_dir": trained.outputs.model_dir,
            "metrics": evaluated.outputs.metrics,
        }

    return pipeline(raw_csv=Input(type=AssetTypes.URI_FILE, path=str(raw)))


def _read_output(client: MLClient, job_name: str, output: str, filename: str) -> Dict[str, Any]:
    folder = Path(tempfile.mkdtemp())
    client.jobs.download(job_name, output_name=output, download_path=str(folder))
    match = next(folder.rglob(filename))
    return json.loads(match.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subscription", default=os.environ.get("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.environ.get("AZUREML_WORKSPACE"))
    parser.add_argument("--raw", help="local AI4I 2020 CSV; downloaded from UCI when omitted")
    parser.add_argument("--instance-type", default="Standard_DS2_v2")
    parser.add_argument("--f1-threshold", type=float, default=0.5)
    parser.add_argument("--no-register", action="store_true", help="run the pipeline without registering the model")
    args = parser.parse_args()
    if not (args.subscription and args.resource_group and args.workspace):
        parser.error("subscription, resource group and workspace are required (flags or AZURE_* / AZUREML_WORKSPACE)")

    client = MLClient(DefaultAzureCredential(), args.subscription, args.resource_group, args.workspace)
    environment = client.environments.create_or_update(
        Environment(
            name=ENVIRONMENT_NAME,
            image=BASE_IMAGE,
            conda_file=str(REPO / "ml" / "environment" / "conda.yml"),
            description="App's pinned requirements plus azureml-mlflow.",
        )
    )
    commit = _git_commit()
    job = build_pipeline("{}:{}".format(environment.name, environment.version), args.instance_type, args.f1_threshold, _raw_dataset(args.raw))
    job.experiment_name = EXPERIMENT_NAME
    job.tags = {"git_commit": commit, "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual")}
    started = time.time()
    job = client.jobs.create_or_update(job)
    print("submitted", job.name, job.studio_url, flush=True)
    client.jobs.stream(job.name)
    job = client.jobs.get(job.name)
    print("status", job.status, "after", round(time.time() - started), "s", flush=True)
    if job.status != "Completed":
        print("pipeline did not complete (a failed quality gate fails the evaluate step); nothing registered")
        return 1

    metrics = _read_output(client, job.name, "metrics", "metrics.json")
    dataset = _read_output(client, job.name, "dataset", "dataset.json")
    print("metrics", json.dumps(metrics))
    if args.no_register:
        return 0
    model = client.models.create_or_update(
        Model(
            name=MODEL_NAME,
            path="azureml://jobs/{}/outputs/model_dir".format(job.name),
            type=AssetTypes.CUSTOM_MODEL,
            description="AI4I 2020 anomaly detector (personal project). Joblib dict with model, reference and features.",
            tags={
                "dataset": "AI4I 2020 (UCI 601)",
                "dataset_sha256": dataset["dataset_sha256"],
                "f1": str(metrics["f1"]),
                "roc_auc": str(metrics["roc_auc"]),
                "f1_threshold": str(metrics["f1_threshold"]),
                "git_commit": commit,
                "pipeline_job": job.name,
            },
        )
    )
    print("registered", model.name, "version", model.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
