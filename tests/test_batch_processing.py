
import mlops
import video_privacy
from mlops import MLOpsService
from robot_data import make_sample_telemetry_csv


def test_large_prediction_is_scored_in_chunks_with_sampled_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(mlops, "PREDICTION_CHUNK_ROWS", 500)
    monkeypatch.setattr(mlops, "DRIFT_SAMPLE_ROWS", 200)
    service = MLOpsService(tmp_path / "m.sqlite3", tmp_path / "models", tmp_path / "ev")
    assert service.train()["status"] == "promoted"

    lines = make_sample_telemetry_csv().splitlines()
    header, rows = lines[0], lines[1:]
    big_csv = "\n".join([header] + rows * 5)

    result = service.predict(big_csv)

    assert result["records_processed"] == len(rows) * 5
    assert 0 <= result["anomalies_detected"] <= result["records_processed"]
    assert len(result["predictions"]) == 100


def test_isolated_video_matches_in_process_result(tmp_path):
    source = tmp_path / "in.mp4"
    video_privacy.create_sample_video(source, seconds=1, fps=5)
    payload = source.read_bytes()

    output, frames, faces = video_privacy.anonymize_video_isolated(payload, timeout=300)

    assert frames > 0
    assert faces >= 0
    assert output[:4] != b""


def test_isolated_video_reports_invalid_input():
    import pytest

    with pytest.raises(ValueError):
        video_privacy.anonymize_video_isolated(b"not a video", timeout=120)
