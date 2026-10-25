import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import convert_ai4i  # noqa: E402
from mlops import FEATURES, MLOpsService  # noqa: E402

SOURCE = (
    "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
    "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF\n"
    "1,M1,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n"
    "2,L2,L,298.2,308.7,1408,46.3,3,1,0,1,0,0,0\n"
)


def test_convert_maps_columns_and_labels():
    rows = list(csv.DictReader(io.StringIO(convert_ai4i.convert(SOURCE))))
    assert rows[0]["duration_ms"] == "1551" and rows[0]["battery_pct"] == "42.8"
    assert rows[0]["x_m"] == "308.6" and rows[0]["y_m"] == "0"
    assert [r["is_anomaly"] for r in rows] == ["0", "1"]
    assert rows[1]["robot_id"] == "MACH-L"


def test_converted_data_is_accepted_by_training(tmp_path):
    lines = ["UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
             "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF"]
    for i in range(1, 201):
        failure = i % 10 == 0
        speed, torque = (1200, 68.0 + i % 5) if failure else (1500 + i % 40, 40.0 + i % 7)
        lines.append("{},P{},L,298.1,308.6,{},{},{},{},0,0,0,0,0".format(i, i, speed, torque, i % 200, int(failure)))
    text = convert_ai4i.convert("\n".join(lines) + "\n")
    assert all(name in text.splitlines()[0] for name in FEATURES)
    result = MLOpsService(tmp_path / "m.sqlite3", tmp_path / "models", tmp_path / "ev").train(text)
    assert result["status"] in ("promoted", "rejected")
