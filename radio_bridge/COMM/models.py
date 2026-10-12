#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RadarRefereeStatus:
    raw: int

    @classmethod
    def from_byte(cls, value: int) -> "RadarRefereeStatus":
        return cls(int(value) & 0xFF)

    @property
    def double_vulnerability_chances(self) -> int:
        return self.raw & 0x03

    @property
    def enemy_double_vulnerability_active(self) -> bool:
        return bool((self.raw >> 2) & 0x01)

    @property
    def encryption_level(self) -> int:
        value = (self.raw >> 3) & 0x03
        return min(3, max(1, value))

    @property
    def password_change_allowed(self) -> bool:
        return bool((self.raw >> 5) & 0x01)

    @property
    def reserved(self) -> int:
        return (self.raw >> 6) & 0x03

    @property
    def level(self) -> str:
        return str(self.encryption_level)

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "raw_hex": f"0x{self.raw:02X}",
            "double_vulnerability_chances": self.double_vulnerability_chances,
            "enemy_double_vulnerability_active": self.enemy_double_vulnerability_active,
            "encryption_level": self.encryption_level,
            "password_change_allowed": self.password_change_allowed,
            "reserved": self.reserved,
        }


@dataclass
class ControlUpdate:
    team: Optional[str] = None
    level: Optional[str] = None
    referee_status: Optional[RadarRefereeStatus] = None
    robot_id: Optional[int] = None
    radar_info: Optional[int] = None
    protocol_version: Optional[int] = None
    valid_mask: Optional[int] = None
    seq: Optional[int] = None
    game_type: Optional[int] = None
    game_progress: Optional[int] = None
    stage_remain_time: Optional[int] = None
    random_payload: bool = False
    random_key: bool = False

    def has_update(self) -> bool:
        return (
            self.team is not None
            or self.level is not None
            or self.referee_status is not None
            or self.robot_id is not None
            or self.radar_info is not None
            or self.protocol_version is not None
            or self.valid_mask is not None
            or self.seq is not None
            or self.game_type is not None
            or self.game_progress is not None
            or self.stage_remain_time is not None
            or self.random_payload
            or self.random_key
        )
