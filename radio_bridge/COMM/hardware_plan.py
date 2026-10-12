#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict


def auto_hardware_lo_shift_hz(plan: Dict, samp_rate_hz: float, avoid_dc_hz: float) -> float:
    """Choose a signed SDR LO shift that keeps both official channels in band."""
    magnitude = abs(float(avoid_dc_hz or 0.0))
    if magnitude <= 0.0:
        return 0.0

    jam_offset_hz = float(plan.get("jam_offset_hz", 0.0))
    sign = -1.0 if jam_offset_hz > 0.0 else 1.0
    nyquist_hz = 0.5 * float(samp_rate_hz)
    bc_half_hz = 0.5 * float(plan.get("broadcast_bandwidth_hz", 0.0))
    jam_half_hz = 0.5 * float(plan.get("jam_bandwidth_hz", 0.0))
    guard_hz = 100e3
    max_shift_hz = nyquist_hz - max(bc_half_hz, abs(jam_offset_hz) + jam_half_hz) - guard_hz
    if max_shift_hz <= 0.0:
        return 0.0
    return sign * min(magnitude, max_shift_hz)


def resolve_hardware_plan(
    plan: Dict,
    samp_rate_hz: float,
    avoid_dc_hz: float = 400e3,
    explicit_shift_hz=None,
    prefix: str = "hardware",
    center_key: str = "rx_center_hz",
) -> Dict:
    if explicit_shift_hz is None:
        shift_hz = auto_hardware_lo_shift_hz(plan, samp_rate_hz, avoid_dc_hz)
    else:
        shift_hz = float(explicit_shift_hz)

    resolved = dict(plan)
    resolved.update(
        {
            f"{prefix}_lo_shift_hz": shift_hz,
            f"{prefix}_lo_hz": float(plan[center_key]) + shift_hz,
            f"{prefix}_broadcast_offset_hz": float(plan.get("broadcast_offset_hz", 0.0)) - shift_hz,
            f"{prefix}_jam_offset_hz": float(plan.get("jam_offset_hz", 0.0)) - shift_hz,
        }
    )
    return resolved


def resolve_hardware_rx_plan(
    plan: Dict,
    samp_rate_hz: float,
    avoid_dc_hz: float = 400e3,
    explicit_shift_hz=None,
) -> Dict:
    return resolve_hardware_plan(
        plan,
        samp_rate_hz,
        avoid_dc_hz,
        explicit_shift_hz,
        prefix="hardware_rx",
        center_key="rx_center_hz",
    )


def resolve_hardware_tx_plan(
    plan: Dict,
    samp_rate_hz: float,
    avoid_dc_hz: float = 400e3,
    explicit_shift_hz=None,
) -> Dict:
    return resolve_hardware_plan(
        plan,
        samp_rate_hz,
        avoid_dc_hz,
        explicit_shift_hz,
        prefix="hardware_tx",
        center_key="tx_center_hz",
    )
