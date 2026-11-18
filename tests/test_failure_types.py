"""AI4I failure-type rules: boundary unit tests, and agreement with the real labels when available."""
import csv
import os
from pathlib import Path

import pytest

from search.failure_types import Reading, likely_failure_types, reading_from_ai4i_row


def reading(**overrides):
    # A healthy reading: no rule triggers.
    values = dict(product_type="L", air_temperature_k=300.0, process_temperature_k=310.0,
                  rotational_speed_rpm=1500.0, torque_nm=40.0, tool_wear_min=100.0)
    values.update(overrides)
    return Reading(**values)


def test_healthy_reading_has_no_failure_type():
    assert likely_failure_types(reading()) == []


def test_heat_dissipation_needs_both_conditions():
    assert "HDF" in likely_failure_types(reading(process_temperature_k=308.5, rotational_speed_rpm=1370))
    assert "HDF" not in likely_failure_types(reading(process_temperature_k=308.5, rotational_speed_rpm=1380))
    assert "HDF" not in likely_failure_types(reading(process_temperature_k=308.6, rotational_speed_rpm=1370))


def test_power_failure_outside_the_band():
    assert "PWF" in likely_failure_types(reading(torque_nm=20.0, rotational_speed_rpm=1500))  # about 3142 W
    assert "PWF" in likely_failure_types(reading(torque_nm=60.0, rotational_speed_rpm=1500))  # about 9425 W
    assert "PWF" not in likely_failure_types(reading(torque_nm=40.0, rotational_speed_rpm=1500))


@pytest.mark.parametrize("product_type,limit", [("L", 11_000), ("M", 12_000), ("H", 13_000)])
def test_overstrain_limit_depends_on_the_variant(product_type, limit):
    at_limit = reading(product_type=product_type, torque_nm=50.0, tool_wear_min=limit / 50.0)
    above = reading(product_type=product_type, torque_nm=50.0, tool_wear_min=limit / 50.0 + 1)
    assert "OSF" not in likely_failure_types(at_limit)
    assert "OSF" in likely_failure_types(above)


def test_tool_wear_window():
    assert "TWF" in likely_failure_types(reading(tool_wear_min=200))
    assert "TWF" in likely_failure_types(reading(tool_wear_min=240))
    assert "TWF" not in likely_failure_types(reading(tool_wear_min=199))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        likely_failure_types(reading(product_type="X"))


AI4I_CSV = os.environ.get("AI4I_CSV")


@pytest.mark.skipif(not (AI4I_CSV and Path(AI4I_CSV).is_file()), reason="set AI4I_CSV to the UCI ai4i2020.csv")
def test_rules_agree_with_ai4i_labels():
    rows = list(csv.DictReader(open(AI4I_CSV, encoding="utf-8-sig")))
    agreement = {}
    for kind in ("TWF", "HDF", "PWF", "OSF"):
        flagged = [kind in likely_failure_types(reading_from_ai4i_row(row)) for row in rows]
        labelled = [row[kind] == "1" for row in rows]
        hits = sum(f and lab for f, lab in zip(flagged, labelled))
        agreement[kind] = (hits / sum(labelled), hits / max(1, sum(flagged)))
    # Deterministic rules reproduce the labels exactly.
    for kind in ("HDF", "PWF", "OSF"):
        assert agreement[kind] == (1.0, 1.0), kind
    # Tool wear failures are random inside the window: high recall, low precision by design.
    assert agreement["TWF"][0] >= 0.9
