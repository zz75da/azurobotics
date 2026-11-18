"""Azure ML pipeline components, run locally on a small synthetic file in the AI4I column format."""
import csv
import io
import json
import random
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "components"))

import evaluate  # noqa: E402
import prep  # noqa: E402
import train  # noqa: E402

AI4I_HEADER = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]


def synthetic_ai4i(rows: int = 1500, seed: int = 7) -> str:
    """AI4I-shaped rows whose failures follow the dataset's heat (HDF) and power (PWF) rules."""
    rng = random.Random(seed)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(AI4I_HEADER)
    for udi in range(1, rows + 1):
        kind = rng.choice("LLLMMH")
        air = round(rng.gauss(300, 2), 1)
        process = round(air + rng.uniform(7.5, 11.5), 1)
        speed = int(rng.gauss(1500, 120))
        torque = round(max(5.0, rng.gauss(40, 9)), 1)
        wear = rng.randint(0, 240)
        power = torque * speed * 2 * 3.14159265 / 60
        hdf = int(process - air < 8.6 and speed < 1380)
        pwf = int(power < 3500 or power > 9000)
        failure = int(hdf or pwf)
        writer.writerow([udi, kind + str(udi), kind, air, process, speed, torque, wear, failure, 0, hdf, pwf, 0, 0])
    return output.getvalue()


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    folder = tmp_path_factory.mktemp("prepared")
    summary = prep.run_prep(synthetic_ai4i(), folder)
    return folder, summary


def test_prep_splits_and_fingerprints_the_dataset(prepared):
    folder, summary = prepared
    assert (folder / "train.csv").is_file() and (folder / "test.csv").is_file()
    assert summary["rows"] == 1500 and summary["train_rows"] + summary["test_rows"] == 1500
    assert len(summary["dataset_sha256"]) == 64
    assert summary["failure_types"]["HDF"] + summary["failure_types"]["PWF"] >= summary["failures"] > 0
    assert json.loads((folder / "dataset.json").read_text(encoding="utf-8")) == summary
    with (folder / "test.csv").open(encoding="utf-8") as handle:
        test_rows = list(csv.DictReader(handle))
    # Stratified: the failure rate of the test split matches the whole file.
    test_rate = sum(int(row["is_anomaly"]) for row in test_rows) / len(test_rows)
    assert abs(test_rate - summary["failures"] / summary["rows"]) < 0.01


def test_prep_is_deterministic(tmp_path):
    first = prep.run_prep(synthetic_ai4i(), tmp_path / "a")
    second = prep.run_prep(synthetic_ai4i(), tmp_path / "b")
    assert first == second
    assert (tmp_path / "a" / "test.csv").read_bytes() == (tmp_path / "b" / "test.csv").read_bytes()


def test_prep_rejects_a_single_class_file(tmp_path):
    lines = synthetic_ai4i(200).splitlines()
    healthy = [lines[0]] + [line for line in lines[1:] if line.split(",")[8] == "0"]
    with pytest.raises(ValueError, match="both"):
        prep.run_prep("\n".join(healthy) + "\n", tmp_path)


def test_train_writes_the_app_artifact_format(prepared, tmp_path):
    folder, _ = prepared
    info = train.run_train(folder / "train.csv", tmp_path)
    artifact = joblib.load(tmp_path / "model.joblib")
    assert set(artifact) == {"model", "reference", "features"}
    assert "temp_delta_k" in artifact["features"] and "power_w" in artifact["features"]
    assert info["algorithm"] == "random_forest"
    assert json.loads((tmp_path / "train_info.json").read_text(encoding="utf-8"))["train_rows"] == info["train_rows"]


def test_train_refuses_automatic_model_selection(prepared, tmp_path):
    folder, _ = prepared
    with pytest.raises(ValueError, match="test split"):
        train.run_train(folder / "train.csv", tmp_path, {"algorithm": "auto"})


def test_evaluate_passes_the_gate_and_writes_metrics(prepared, tmp_path):
    folder, _ = prepared
    train.run_train(folder / "train.csv", tmp_path / "model")
    metrics = evaluate.run_evaluate(tmp_path / "model" / "model.joblib", folder / "test.csv", tmp_path / "metrics", 0.5)
    assert metrics["gate_passed"] and metrics["f1"] >= 0.5
    assert json.loads((tmp_path / "metrics" / "metrics.json").read_text(encoding="utf-8")) == metrics


def test_evaluate_fails_the_pipeline_below_the_threshold(prepared, tmp_path, monkeypatch, capsys):
    folder, _ = prepared
    train.run_train(folder / "train.csv", tmp_path / "model")
    with pytest.raises(evaluate.QualityGateError):
        evaluate.run_evaluate(tmp_path / "model" / "model.joblib", folder / "test.csv", tmp_path / "m1", 1.01)
    # The command-line entry point turns the gate failure into a non-zero exit code.
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--model-dir", str(tmp_path / "model"), "--prepared", str(folder),
        "--metrics-out", str(tmp_path / "m2"), "--f1-threshold", "1.01",
    ])
    assert evaluate.main() == 1
    assert "QUALITY GATE FAILED" in capsys.readouterr().err
    assert json.loads((tmp_path / "m2" / "metrics.json").read_text(encoding="utf-8"))["gate_passed"] is False
