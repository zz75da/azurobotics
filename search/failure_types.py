"""Likely AI4I 2020 failure types from one machine reading, using the dataset's documented rules.

The AI4I 2020 dataset (Matzka, UCI 601) generates its failure labels from these rules:
    TWF  tool wear failure: the tool is replaced or fails at a random wear time between 200 and 240 min
    HDF  heat dissipation: process minus air temperature below 8.6 K and rotational speed below 1380 rpm
    PWF  power failure: power (torque x angular speed) below 3500 W or above 9000 W
    OSF  overstrain: tool wear x torque above 11,000 (variant L), 12,000 (M) or 13,000 (H) minNm
    RNF  random failure with probability 0.1 %, which no rule can predict
TWF is random inside its window, so the rule flags the window ("likely"), not the exact failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

HDF_MAX_TEMP_DELTA_K = 8.6
HDF_MAX_SPEED_RPM = 1380
PWF_MIN_POWER_W = 3500
PWF_MAX_POWER_W = 9000
OSF_LIMITS_MIN_NM = {"L": 11_000, "M": 12_000, "H": 13_000}
TWF_WEAR_WINDOW_MIN = (200, 240)


@dataclass(frozen=True)
class Reading:
    product_type: str  # "L", "M" or "H"
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    @property
    def power_w(self) -> float:
        return self.torque_nm * self.rotational_speed_rpm * 2 * math.pi / 60

    @property
    def temp_delta_k(self) -> float:
        return self.process_temperature_k - self.air_temperature_k


def likely_failure_types(reading: Reading) -> List[str]:
    """Failure types whose documented condition holds for this reading (possibly several, or none)."""
    if reading.product_type not in OSF_LIMITS_MIN_NM:
        raise ValueError("product_type must be L, M or H")
    types = []
    low, high = TWF_WEAR_WINDOW_MIN
    if low <= reading.tool_wear_min <= high:
        types.append("TWF")
    if reading.temp_delta_k < HDF_MAX_TEMP_DELTA_K and reading.rotational_speed_rpm < HDF_MAX_SPEED_RPM:
        types.append("HDF")
    if reading.power_w < PWF_MIN_POWER_W or reading.power_w > PWF_MAX_POWER_W:
        types.append("PWF")
    if reading.tool_wear_min * reading.torque_nm > OSF_LIMITS_MIN_NM[reading.product_type]:
        types.append("OSF")
    return types


def reading_from_ai4i_row(row: dict) -> Reading:
    """Build a Reading from a row of the UCI AI4I 2020 CSV (original column names)."""
    return Reading(
        product_type=row["Type"],
        air_temperature_k=float(row["Air temperature [K]"]),
        process_temperature_k=float(row["Process temperature [K]"]),
        rotational_speed_rpm=float(row["Rotational speed [rpm]"]),
        torque_nm=float(row["Torque [Nm]"]),
        tool_wear_min=float(row["Tool wear [min]"]),
    )
