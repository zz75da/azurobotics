import csv
import io
import json
import zipfile

import pytest

from robot_data import analyze_text, make_sample_bundle, make_sample_telemetry_csv


def test_analyze_csv_calculates_robot_and_event_metrics():
    content = (
        "timestamp,robot_id,event,duration_ms,battery_pct\n"
        "2026-01-01T00:00:00Z,AMR-01,move,100,80\n"
        "2026-01-01T00:00:01Z,AMR-02,error,300,60\n"
        "2026-01-01T00:00:02Z,AMR-01,move,200,\n"
    )

    result = analyze_text(content, ".csv")

    assert result["events_total"] == 3
    assert result["robots_total"] == 2
    assert result["errors_total"] == 1
    assert result["error_rate_pct"] == pytest.approx(33.33)
    assert result["battery_avg_pct"] == 70
    assert result["duration_avg_ms"] == 200
    assert result["event_counts"] == {"move": 2, "error": 1}
    assert result["quality_issues"]["invalid_battery_pct"] == 0


def test_quality_checks_find_missing_fields_and_out_of_range_values():
    content = (
        "timestamp,robot_id,event,duration_ms,battery_pct\n"
        ",,move,-5,140\n"
        "2026-01-01T00:00:01Z,R2,,100,50\n"
    )

    result = analyze_text(content, ".csv")

    assert result["quality_issues"] == {
        "missing_timestamp": 1,
        "missing_robot_id": 1,
        "missing_event": 1,
        "invalid_battery_pct": 1,
        "invalid_duration_ms": 1,
    }


def test_analyze_jsonl_and_missing_optional_fields():
    records = [
        {"robot_id": "R1", "type": "failure"},
        {"robot_id": "R2", "event": "pick", "battery_pct": 75},
    ]

    result = analyze_text("\n".join(json.dumps(item) for item in records), ".jsonl")

    assert result["errors_total"] == 1
    assert result["robots_total"] == 2
    assert result["battery_avg_pct"] == 75


def test_analyze_rejects_invalid_jsonl():
    with pytest.raises(ValueError, match="ligne 2"):
        analyze_text('{"event":"move"}\nnot json', ".jsonl")


def test_analyze_stops_at_the_row_limit():
    content = '{"event":"move"}\n' * 200_001

    with pytest.raises(ValueError, match="200000 lignes"):
        analyze_text(content, ".jsonl")


def test_sample_bundle_contains_synthetic_csv_incidents_and_video():
    bundle = make_sample_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {
            "data/telemetry.csv",
            "data/incidents.jsonl",
            "video/warehouse_test.avi",
            "README.txt",
        }
        rows = list(csv.DictReader(io.StringIO(archive.read("data/telemetry.csv").decode("utf-8"))))
        assert len(rows) == 600
        assert sum(int(row["is_anomaly"]) for row in rows) == 30
        assert len(archive.read("video/warehouse_test.avi")) < 10 * 1024 * 1024
        assert b"synth" in archive.read("README.txt").lower()


def test_sample_data_contains_reproducible_anomaly_labels():
    rows = list(csv.DictReader(io.StringIO(make_sample_telemetry_csv())))

    assert len(rows) == 600
    assert sum(int(row["is_anomaly"]) for row in rows) == 30
