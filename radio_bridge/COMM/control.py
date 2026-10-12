#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import struct
from typing import Dict, Optional, Tuple

from .crc import crc8, crc16
from .models import ControlUpdate, RadarRefereeStatus
from .radio_packets import parse_radio_ext_mode_packet, parse_radio_mode_packet
from .rf_plan import normalize_level, normalize_team


RADAR_REFEREE_CMD_ID = 0x020E


def parse_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def bytes_from_value(value) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, list):
        return bytes(int(item) & 0xFF for item in value)
    if isinstance(value, int):
        return bytes([value & 0xFF])

    text = str(value).strip()
    if not text:
        return b""
    if text.startswith(("0x", "0X")) and re.fullmatch(r"0[xX][0-9a-fA-F]+", text):
        text = text[2:]
    cleaned = re.sub(r"[^0-9a-fA-F]", "", text)
    if len(cleaned) % 2:
        cleaned = "0" + cleaned
    if cleaned:
        try:
            return bytes.fromhex(cleaned)
        except ValueError:
            pass
    return text.encode("utf-8", errors="ignore")


def parse_referee_payload(cmd_id, payload) -> Optional[RadarRefereeStatus]:
    parsed_cmd = parse_int(cmd_id)
    if parsed_cmd != RADAR_REFEREE_CMD_ID:
        return None

    if not isinstance(payload, (bytes, bytearray, list)):
        direct_value = parse_int(payload)
        if direct_value is not None:
            return RadarRefereeStatus.from_byte(direct_value)

    data = bytes_from_value(payload)
    if not data:
        return None
    return RadarRefereeStatus.from_byte(data[0])


def parse_rm_frame(data: bytes) -> Optional[Tuple[int, bytes]]:
    if len(data) < 9 or data[0] != 0xA5:
        return None

    data_len = data[1] | (data[2] << 8)
    full_len = 9 + data_len
    if len(data) < full_len:
        return None
    frame = data[:full_len]
    if crc8(frame[:4]) != frame[4]:
        return None
    recv_crc = frame[-2] | (frame[-1] << 8)
    if crc16(frame[:-2]) != recv_crc:
        return None

    cmd_id = frame[5] | (frame[6] << 8)
    return cmd_id, frame[7 : 7 + data_len]


def build_rm_frame(cmd_id: int, payload: bytes, seq: int = 0) -> bytes:
    data = bytes(payload or b"")
    header = struct.pack("<BHB", 0xA5, len(data), int(seq) & 0xFF)
    frame = header + struct.pack("<B", crc8(header)) + struct.pack("<H", int(cmd_id) & 0xFFFF) + data
    return frame + struct.pack("<H", crc16(frame))


def parse_json_update(payload: Dict) -> ControlUpdate:
    def first_present(*keys):
        for key in keys:
            if key in payload:
                return payload.get(key)
        return None

    team = first_present("team", "side", "color", "self_team", "own_team")
    level = first_present(
        "level",
        "interference_level",
        "jam_level",
        "jamming_level",
        "enemy_jam_level",
        "encryption_level",
        "self_level",
        "own_level",
    )

    status = None
    for key in ("status_byte", "radar_status", "referee_status", "data"):
        if key in payload:
            status = parse_referee_payload(payload.get("cmd_id", RADAR_REFEREE_CMD_ID), payload.get(key))
            if status is not None:
                break

    if status is None and "cmd_id" in payload and "payload" in payload:
        status = parse_referee_payload(payload.get("cmd_id"), payload.get("payload"))

    return ControlUpdate(
        team=normalize_team(team) if team is not None else None,
        level=normalize_level(level) if level is not None else (status.level if status is not None else None),
        referee_status=status,
        random_payload=bool(payload.get("random_payload", False)),
        random_key=bool(payload.get("random_key", False)),
    )


def parse_control_update(data: bytes) -> ControlUpdate:
    if not isinstance(data, (bytes, bytearray)):
        data = str(data).encode("utf-8", errors="ignore")
    raw = bytes(data)
    if not raw:
        raise ValueError("empty control payload")

    radio_update = parse_radio_ext_mode_packet(raw) or parse_radio_mode_packet(raw)
    if radio_update is not None:
        return radio_update

    frame = parse_rm_frame(raw)
    if frame is not None:
        cmd_id, payload = frame
        status = parse_referee_payload(cmd_id, payload)
        if status is not None:
            return ControlUpdate(level=status.level, referee_status=status)

    if len(raw) == 1:
        status = RadarRefereeStatus.from_byte(raw[0])
        return ControlUpdate(level=status.level, referee_status=status)

    text = raw.strip().decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("empty control payload")
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("control JSON must be an object")
        update = parse_json_update(parsed)
        if update.has_update():
            return update

    parts = text.split()
    if len(parts) >= 2:
        maybe_cmd = parse_int(parts[0])
        if maybe_cmd == RADAR_REFEREE_CMD_ID:
            status = parse_referee_payload(maybe_cmd, parts[1])
            if status is not None:
                return ControlUpdate(level=status.level, referee_status=status)
        return ControlUpdate(team=normalize_team(parts[0]), level=normalize_level(parts[1]))

    status_value = parse_int(text)
    if status_value is not None:
        status = RadarRefereeStatus.from_byte(status_value)
        return ControlUpdate(level=status.level, referee_status=status)

    raise ValueError("control payload must be JSON, RM frame, status byte, or 'team level'")


_parse_int = parse_int
_bytes_from_value = bytes_from_value
_parse_referee_payload = parse_referee_payload
_parse_rm_frame = parse_rm_frame
_parse_json_update = parse_json_update
