#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in bytes(data):
        crc ^= byte & 0xFF
        for _ in range(8):
            if crc & 0x01:
                crc = ((crc >> 1) ^ 0x8C) & 0xFF
            else:
                crc = (crc >> 1) & 0xFF
    return crc & 0xFF


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in bytes(data):
        crc ^= byte & 0xFF
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc & 0xFFFF


_crc8 = crc8
_crc16 = crc16
