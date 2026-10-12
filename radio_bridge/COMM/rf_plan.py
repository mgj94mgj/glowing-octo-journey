#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import string
from typing import Dict, Tuple


OFFICIAL_RF_PLAN_HZ = {
    "red": {
        "broadcast_hz": 433.2e6,
        "jam_hz": {
            "1": 432.2e6,
            "2": 432.5e6,
            "3": 432.8e6,
        },
    },
    "blue": {
        "broadcast_hz": 433.92e6,
        "jam_hz": {
            "1": 434.92e6,
            "2": 434.62e6,
            "3": 434.32e6,
        },
    },
}

OFFICIAL_SIGNAL_BW_HZ = {
    "broadcast": 0.54e6,
    "jam": {
        "1": 0.94e6,
        "2": 0.86e6,
        "3": 0.25e6,
    },
}

LEVEL_ALIASES = {
    "1": "1",
    "l": "1",
    "low": "1",
    "level1": "1",
    "level_1": "1",
    "jam_l": "1",
    "2": "2",
    "m": "2",
    "mid": "2",
    "medium": "2",
    "level2": "2",
    "level_2": "2",
    "jam_m": "2",
    "3": "3",
    "h": "3",
    "high": "3",
    "level3": "3",
    "level_3": "3",
    "jam_h": "3",
}

DEFAULT_INFO_PAYLOAD = {
    "hero_x": 32,
    "hero_y": 123,
    "engineer_x": 4,
    "engineer_y": 3,
    "hero_hp": 2000,
    "engineer_hp": 500,
    "hero_ammo": 100,
    "remaining_gold": 1000,
    "total_gold": 2000,
}

DEFAULT_JAM_PAYLOAD = {
    "key": "gvwr4d",
}

_KEY_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def normalize_team(team: str) -> str:
    token = str(team).strip().lower()
    if token in OFFICIAL_RF_PLAN_HZ:
        return token
    return "red"


def normalize_level(level: str) -> str:
    token = str(level).strip().lower().replace("-", "_").replace(" ", "")
    return LEVEL_ALIASES.get(token, "1")


def normalize_mode(mode: str) -> str:
    token = str(mode).strip().lower()
    if token in ("hardware", "sim", "auto", "zmq"):
        return token
    return "auto"


def jam_profile_name(team: str, level: str) -> str:
    suffix = {"1": "l", "2": "m", "3": "h"}[normalize_level(level)]
    return f"{normalize_team(team)}_jam_{suffix}"


def build_rf_plan(team: str, level: str) -> Dict[str, float]:
    team_token = normalize_team(team)
    level_token = normalize_level(level)

    bc_hz = float(OFFICIAL_RF_PLAN_HZ[team_token]["broadcast_hz"])
    jam_hz = float(OFFICIAL_RF_PLAN_HZ[team_token]["jam_hz"][level_token])
    bc_bw_hz = float(OFFICIAL_SIGNAL_BW_HZ["broadcast"])
    jam_bw_hz = float(OFFICIAL_SIGNAL_BW_HZ["jam"][level_token])
    lower_edge_hz = min(bc_hz - 0.5 * bc_bw_hz, jam_hz - 0.5 * jam_bw_hz)
    upper_edge_hz = max(bc_hz + 0.5 * bc_bw_hz, jam_hz + 0.5 * jam_bw_hz)
    span_hz = upper_edge_hz - lower_edge_hz

    return {
        "team": team_token,
        "level": level_token,
        "tx_center_hz": bc_hz,
        "rx_center_hz": bc_hz,
        "broadcast_hz": bc_hz,
        "jam_hz": jam_hz,
        "broadcast_bandwidth_hz": bc_bw_hz,
        "jam_bandwidth_hz": jam_bw_hz,
        "lower_edge_hz": lower_edge_hz,
        "upper_edge_hz": upper_edge_hz,
        "occupied_span_hz": span_hz,
        "recommended_samp_rate_hz": max(
            4e6,
            2.0 * max(abs(lower_edge_hz - bc_hz), abs(upper_edge_hz - bc_hz)) + 0.2e6,
        ),
        "broadcast_offset_hz": 0.0,
        "jam_offset_hz": jam_hz - bc_hz,
    }


def load_payload_json(path: str, defaults: Dict) -> Dict:
    payload = dict(defaults)
    if not path:
        return payload

    with open(path, "r", encoding="utf-8") as handle:
        parsed = json.load(handle)

    if not isinstance(parsed, dict):
        raise ValueError(f"payload file must contain a JSON object: {path}")

    payload.update(parsed)
    return payload


def random_team_level() -> Tuple[str, str]:
    return random.choice(("red", "blue")), random.choice(("1", "2", "3"))


def random_key(length: int = 6) -> str:
    return "".join(random.choice(_KEY_ALPHABET) for _ in range(int(length)))


def random_info_payload() -> Dict[str, int]:
    payload = {}
    robot_prefixes = ("hero", "engineer", "infantry_3", "infantry_4", "aerial", "sentry")

    for prefix in robot_prefixes:
        payload[f"{prefix}_x"] = random.randint(0, 2800)
        payload[f"{prefix}_y"] = random.randint(0, 1500)

    hp_ranges = {
        "hero": (0, 2000),
        "engineer": (0, 1000),
        "infantry_3": (0, 1500),
        "infantry_4": (0, 1500),
        "sentry": (0, 1500),
    }
    for prefix, (lo, hi) in hp_ranges.items():
        payload[f"{prefix}_hp"] = random.randint(lo, hi)
    payload["reserved"] = 0

    for key in ("hero_ammo", "infantry_3_ammo", "infantry_4_ammo", "aerial_ammo", "sentry_ammo"):
        payload[key] = random.randint(0, 500)

    total_gold = random.randint(0, 6000)
    payload["total_gold"] = total_gold
    payload["remaining_gold"] = random.randint(0, total_gold)
    macro_fields = {
        "opponent_supply_zone_occupied": random.randint(0, 1),
        "opponent_center_highland_occupancy": random.randint(0, 2),
        "opponent_trapezoid_highland_occupied": random.randint(0, 1),
        "opponent_fortress_buff_occupancy": random.randint(0, 3),
        "opponent_outpost_buff_occupancy": random.randint(0, 3),
        "opponent_base_buff_occupied": random.randint(0, 1),
        "terrain_near_opponent_ramp_front_card": random.randint(0, 1),
        "terrain_near_opponent_ramp_back_card": random.randint(0, 1),
        "terrain_near_own_ramp_front_card": random.randint(0, 1),
        "terrain_near_own_ramp_back_card": random.randint(0, 1),
        "terrain_opponent_highland_upper_card": random.randint(0, 1),
        "terrain_opponent_ramp_back_card": random.randint(0, 1),
        "terrain_opponent_road_upper_card": random.randint(0, 1),
    }
    payload.update(macro_fields)
    payload["macro_bits"] = (
        macro_fields["opponent_supply_zone_occupied"]
        | (macro_fields["opponent_center_highland_occupancy"] << 1)
        | (macro_fields["opponent_trapezoid_highland_occupied"] << 3)
        | (macro_fields["opponent_fortress_buff_occupancy"] << 4)
        | (macro_fields["opponent_outpost_buff_occupancy"] << 6)
        | (macro_fields["opponent_base_buff_occupied"] << 8)
        | (macro_fields["terrain_near_opponent_ramp_front_card"] << 9)
        | (macro_fields["terrain_near_opponent_ramp_back_card"] << 10)
        | (macro_fields["terrain_near_own_ramp_front_card"] << 11)
        | (macro_fields["terrain_near_own_ramp_back_card"] << 12)
        | (macro_fields["terrain_opponent_highland_upper_card"] << 13)
        | (macro_fields["terrain_opponent_ramp_back_card"] << 14)
        | (macro_fields["terrain_opponent_road_upper_card"] << 15)
    )

    gain_prefixes = ("hero", "engineer", "infantry_3", "infantry_4", "aerial", "sentry")
    for prefix in gain_prefixes:
        payload[f"{prefix}_health_regen"] = random.randint(0, 100)
        payload[f"{prefix}_cooling_boost"] = random.randint(0, 100)
        payload[f"{prefix}_defense_boost"] = random.randint(0, 100)
        payload[f"{prefix}_defense_debuff"] = random.randint(0, 100)
        payload[f"{prefix}_attack_boost"] = random.randint(0, 100)
        if prefix != "sentry":
            payload[f"{prefix}_status"] = random.randint(0, 1)
    payload["sentry_posture"] = random.randint(0, 3)

    return payload


def random_jam_payload() -> Dict[str, str]:
    return {"key": random_key()}
