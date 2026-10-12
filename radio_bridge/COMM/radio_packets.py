#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
from typing import Optional

from .crc import crc16
from .models import ControlUpdate, RadarRefereeStatus
from .rf_plan import normalize_level, normalize_team


RADIO_MODE_TX_PACKET_ID = 0x5C
RADIO_MODE_TX_PACKET_LEN = 8
RADIO_MODE_DEMOD_CONFIG_TYPE = 0x01
RADAR_ROBOT_IDS = {9, 109}

RADIO_EXT_MODE_TX_PACKET_ID = 0x5D
RADIO_EXT_MODE_TX_PACKET_LEN = 16
RADIO_EXT_MODE_VERSION = 0x01

RADIO_EXT_VALID_ROBOT_ID = 0x0001
RADIO_EXT_VALID_TEAM = 0x0002
RADIO_EXT_VALID_JAM_LEVEL = 0x0004
RADIO_EXT_VALID_RADAR_INFO = 0x0008
RADIO_EXT_VALID_GAME_STATUS = 0x0010

RADIO_JAMMING_KEY_RX_PACKET_ID = 0xC5
RADIO_JAMMING_KEY_RX_PACKET_LEN = 9

RADIO_INFO_WAVE_RX_PACKET_ID = 0xD5
RADIO_INFO_WAVE_RX_PACKET_LEN = 94

RADIO_PACKET_LENGTHS = {
    RADIO_MODE_TX_PACKET_ID: RADIO_MODE_TX_PACKET_LEN,
    RADIO_EXT_MODE_TX_PACKET_ID: RADIO_EXT_MODE_TX_PACKET_LEN,
    RADIO_JAMMING_KEY_RX_PACKET_ID: RADIO_JAMMING_KEY_RX_PACKET_LEN,
    RADIO_INFO_WAVE_RX_PACKET_ID: RADIO_INFO_WAVE_RX_PACKET_LEN,
}

RADIO_INFO_POSITION_VALID = 0x01
RADIO_INFO_HP_VALID = 0x02
RADIO_INFO_BULLET_REMAINING_VALID = 0x04
RADIO_INFO_MACRO_STATUS_VALID = 0x08
RADIO_INFO_BUFF_VALID = 0x10

_INFO_WAVE_FORMAT = "<B" + ("H" * 12) + ("H" * 6) + ("H" * 5) + "HHI" + ("BHBBH" * 5) + "B"
assert struct.calcsize(_INFO_WAVE_FORMAT) == RADIO_INFO_WAVE_RX_PACKET_LEN - 3


def _u16(value: int) -> int:
    return int(value) & 0xFFFF


def _u8(value: int) -> int:
    return int(value) & 0xFF


def with_radio_crc(header: int, payload: bytes, total_len: int) -> bytes:
    frame_no_crc = bytes([int(header) & 0xFF]) + bytes(payload)
    if len(frame_no_crc) != int(total_len) - 2:
        raise ValueError(f"radio packet length mismatch: {len(frame_no_crc) + 2} != {total_len}")
    return frame_no_crc + struct.pack("<H", crc16(frame_no_crc))


def radio_crc_ok(frame: bytes) -> bool:
    raw = bytes(frame)
    if len(raw) < 3:
        return False
    expected_len = RADIO_PACKET_LENGTHS.get(raw[0])
    if expected_len is None or len(raw) != expected_len:
        return False
    recv = raw[-2] | (raw[-1] << 8)
    return crc16(raw[:-2]) == recv


def radio_crc_debug(frame: bytes) -> str:
    raw = bytes(frame or b"")
    if len(raw) < 3:
        return f"too_short len={len(raw)}"
    expected_len = RADIO_PACKET_LENGTHS.get(raw[0])
    if expected_len is None:
        return f"unknown_id=0x{raw[0]:02X} len={len(raw)}"
    recv = raw[-2] | (raw[-1] << 8)
    calc = crc16(raw[:-2])
    return (
        f"id=0x{raw[0]:02X} len={len(raw)}/{expected_len} "
        f"crc_recv=0x{recv:04X} crc_calc=0x{calc:04X}"
    )


def parse_radio_mode_packet(data: bytes) -> Optional[ControlUpdate]:
    raw = bytes(data)
    if len(raw) != RADIO_MODE_TX_PACKET_LEN or raw[0] != RADIO_MODE_TX_PACKET_ID:
        return None
    if not radio_crc_ok(raw):
        return None

    msg_type, team_id, jam_level, robot_id, radar_info = struct.unpack("<BBBBB", raw[1:-2])
    if msg_type != RADIO_MODE_DEMOD_CONFIG_TYPE:
        return None
    if int(robot_id) not in RADAR_ROBOT_IDS:
        return None

    status = RadarRefereeStatus.from_byte(radar_info)
    team = "blue" if int(team_id) == 1 else "red"
    level_value = int(jam_level)
    if level_value not in (1, 2, 3):
        level_value = int(status.encryption_level)
    level = normalize_level(str(level_value))

    return ControlUpdate(
        team=team,
        level=level,
        referee_status=status,
        robot_id=int(robot_id),
        radar_info=int(radar_info),
    )


def parse_radio_ext_mode_packet(data: bytes) -> Optional[ControlUpdate]:
    raw = bytes(data)
    if len(raw) != RADIO_EXT_MODE_TX_PACKET_LEN or raw[0] != RADIO_EXT_MODE_TX_PACKET_ID:
        return None
    if not radio_crc_ok(raw):
        return None

    (
        protocol_version,
        valid_mask,
        seq,
        team_id,
        jam_level,
        robot_id,
        radar_info,
        game_type,
        game_progress,
        stage_remain_time,
    ) = struct.unpack("<BHHBBBBBBH", raw[1:-2])
    if protocol_version != RADIO_EXT_MODE_VERSION:
        return None

    has_robot_id = bool(valid_mask & RADIO_EXT_VALID_ROBOT_ID)
    has_team = bool(valid_mask & RADIO_EXT_VALID_TEAM)
    has_jam_level = bool(valid_mask & RADIO_EXT_VALID_JAM_LEVEL)
    has_radar_info = bool(valid_mask & RADIO_EXT_VALID_RADAR_INFO)
    has_game_status = bool(valid_mask & RADIO_EXT_VALID_GAME_STATUS)

    status = RadarRefereeStatus.from_byte(radar_info) if has_radar_info else None
    team = None
    if has_team:
        if int(team_id) == 1:
            team = "blue"
        elif int(team_id) == 0:
            team = "red"

    level = None
    level_value = int(jam_level)
    if has_jam_level and level_value in (1, 2, 3):
        level = normalize_level(str(level_value))

    return ControlUpdate(
        team=team,
        level=level,
        referee_status=status,
        robot_id=int(robot_id) if has_robot_id else None,
        radar_info=int(radar_info) if has_radar_info else None,
        protocol_version=int(protocol_version),
        valid_mask=int(valid_mask),
        seq=int(seq),
        game_type=int(game_type) if has_game_status else None,
        game_progress=int(game_progress) if has_game_status else None,
        stage_remain_time=int(stage_remain_time) if has_game_status else None,
    )


def build_radio_mode_packet(
    team: str = "red",
    level: str = None,
    robot_id: int = 0,
    radar_info: int = None,
    msg_type: int = RADIO_MODE_DEMOD_CONFIG_TYPE,
) -> bytes:
    if radar_info is None:
        level_value = int(normalize_level(level or "1"))
        radar_info = (level_value & 0x03) << 3
    else:
        radar_info = _u8(radar_info)
        if level is None:
            level = RadarRefereeStatus.from_byte(radar_info).level
        level_value = int(normalize_level(level))

    team_id = 1 if normalize_team(team) == "blue" else 0
    payload = struct.pack(
        "<BBBBB",
        _u8(msg_type),
        _u8(team_id),
        _u8(level_value),
        _u8(robot_id),
        _u8(radar_info),
    )
    return with_radio_crc(
        RADIO_MODE_TX_PACKET_ID,
        payload,
        RADIO_MODE_TX_PACKET_LEN,
    )


def build_radio_ext_mode_packet(
    team: str = "red",
    level: str = None,
    robot_id: int = 0,
    radar_info: int = None,
    game_type: int = 0,
    game_progress: int = 0,
    stage_remain_time: int = 0,
    seq: int = 0,
    valid_mask: int = (
        RADIO_EXT_VALID_ROBOT_ID
        | RADIO_EXT_VALID_TEAM
        | RADIO_EXT_VALID_JAM_LEVEL
        | RADIO_EXT_VALID_RADAR_INFO
    ),
) -> bytes:
    if radar_info is None:
        level_value = int(normalize_level(level or "1"))
        radar_info = (level_value & 0x03) << 3
    else:
        radar_info = _u8(radar_info)
        if level is None:
            level = RadarRefereeStatus.from_byte(radar_info).level
        level_value = int(normalize_level(level))

    team_id = 1 if normalize_team(team) == "blue" else 0
    payload = struct.pack(
        "<BHHBBBBBBH",
        _u8(RADIO_EXT_MODE_VERSION),
        _u16(valid_mask),
        _u16(seq),
        _u8(team_id),
        _u8(level_value),
        _u8(robot_id),
        _u8(radar_info),
        _u8(game_type),
        _u8(game_progress),
        _u16(stage_remain_time),
    )
    return with_radio_crc(
        RADIO_EXT_MODE_TX_PACKET_ID,
        payload,
        RADIO_EXT_MODE_TX_PACKET_LEN,
    )


def build_radio_jamming_key_packet(key_data: bytes) -> bytes:
    key = bytes(key_data or b"")[:6].ljust(6, b"\x00")
    payload = struct.pack("<6s", key)
    return with_radio_crc(
        RADIO_JAMMING_KEY_RX_PACKET_ID,
        payload,
        RADIO_JAMMING_KEY_RX_PACKET_LEN,
    )


class RadioInfoWaveState:
    def __init__(self):
        self.valid_mask = 0
        self.positions = [0] * 12
        self.hp = [0] * 6
        self.bullets = [0] * 5
        self.coin_remain = 0
        self.cumulative_coin = 0
        self.macro_status = 0
        self.buffs = [[0, 0, 0, 0, 0] for _ in range(5)]
        self.sentry_posture = 0

    def update_from_frame(self, cmd_id: int, data: bytes) -> bool:
        raw = bytes(data or b"")
        cmd = int(cmd_id) & 0xFFFF

        if cmd == 0x0A01 and len(raw) >= 24:
            self.positions = [_u16(v) for v in struct.unpack("<hhhhhhhhhhhh", raw[:24])]
            self.valid_mask |= RADIO_INFO_POSITION_VALID
            return True

        if cmd == 0x0A02 and len(raw) >= 12:
            self.hp = [_u16(v) for v in struct.unpack("<HHHHHH", raw[:12])]
            self.valid_mask |= RADIO_INFO_HP_VALID
            return True

        if cmd == 0x0A03 and len(raw) >= 10:
            self.bullets = [_u16(v) for v in struct.unpack("<HHHHH", raw[:10])]
            self.valid_mask |= RADIO_INFO_BULLET_REMAINING_VALID
            return True

        if cmd == 0x0A04 and len(raw) >= 8:
            self.coin_remain, self.cumulative_coin, self.macro_status = struct.unpack("<HHI", raw[:8])
            self.coin_remain = _u16(self.coin_remain)
            self.cumulative_coin = _u16(self.cumulative_coin)
            self.macro_status = int(self.macro_status) & 0xFFFFFFFF
            self.valid_mask |= RADIO_INFO_MACRO_STATUS_VALID
            return True

        if cmd == 0x0A05 and len(raw) >= 36:
            self.buffs = [
                [_u8(regen), _u16(cooling), _u8(defense), _u8(debuff), _u16(attack)]
                for regen, cooling, defense, debuff, attack in (
                    struct.unpack_from("<BHBBH", raw, offset)
                    for offset in (0, 7, 14, 21, 28)
                )
            ]
            self.sentry_posture = _u8(raw[35])
            self.valid_mask |= RADIO_INFO_BUFF_VALID
            return True

        return False

    def to_payload(self) -> bytes:
        buff_values = []
        for item in self.buffs:
            buff_values.extend([_u8(item[0]), _u16(item[1]), _u8(item[2]), _u8(item[3]), _u16(item[4])])

        values = (
            [_u8(self.valid_mask)]
            + [_u16(v) for v in self.positions]
            + [_u16(v) for v in self.hp]
            + [_u16(v) for v in self.bullets]
            + [_u16(self.coin_remain), _u16(self.cumulative_coin), int(self.macro_status) & 0xFFFFFFFF]
            + buff_values
            + [_u8(self.sentry_posture)]
        )
        return struct.pack(_INFO_WAVE_FORMAT, *values)

    def build_packet(self) -> bytes:
        return with_radio_crc(
            RADIO_INFO_WAVE_RX_PACKET_ID,
            self.to_payload(),
            RADIO_INFO_WAVE_RX_PACKET_LEN,
        )


_with_radio_crc = with_radio_crc
_radio_crc_ok = radio_crc_ok
