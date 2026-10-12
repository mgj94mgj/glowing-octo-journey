#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .crc import _crc8, _crc16, crc8, crc16
from .models import ControlUpdate, RadarRefereeStatus
from .rf_plan import (
    DEFAULT_INFO_PAYLOAD,
    DEFAULT_JAM_PAYLOAD,
    LEVEL_ALIASES,
    OFFICIAL_RF_PLAN_HZ,
    OFFICIAL_SIGNAL_BW_HZ,
    build_rf_plan,
    jam_profile_name,
    load_payload_json,
    normalize_level,
    normalize_mode,
    normalize_team,
    random_info_payload,
    random_jam_payload,
    random_key,
    random_team_level,
)
from .hardware_plan import (
    auto_hardware_lo_shift_hz,
    resolve_hardware_plan,
    resolve_hardware_rx_plan,
    resolve_hardware_tx_plan,
)
from .radio_packets import (
    RADIO_INFO_BUFF_VALID,
    RADIO_INFO_BULLET_REMAINING_VALID,
    RADIO_INFO_HP_VALID,
    RADIO_INFO_MACRO_STATUS_VALID,
    RADIO_INFO_POSITION_VALID,
    RADIO_INFO_WAVE_RX_PACKET_ID,
    RADIO_INFO_WAVE_RX_PACKET_LEN,
    RADIO_JAMMING_KEY_RX_PACKET_ID,
    RADIO_JAMMING_KEY_RX_PACKET_LEN,
    RADIO_EXT_MODE_TX_PACKET_ID,
    RADIO_EXT_MODE_TX_PACKET_LEN,
    RADIO_EXT_MODE_VERSION,
    RADIO_EXT_VALID_GAME_STATUS,
    RADIO_EXT_VALID_JAM_LEVEL,
    RADIO_EXT_VALID_RADAR_INFO,
    RADIO_EXT_VALID_ROBOT_ID,
    RADIO_EXT_VALID_TEAM,
    RADIO_MODE_DEMOD_CONFIG_TYPE,
    RADIO_MODE_TX_PACKET_ID,
    RADIO_MODE_TX_PACKET_LEN,
    RADIO_PACKET_LENGTHS,
    RadioInfoWaveState,
    _radio_crc_ok,
    _with_radio_crc,
    build_radio_ext_mode_packet,
    build_radio_mode_packet,
    build_radio_jamming_key_packet,
    parse_radio_ext_mode_packet,
    parse_radio_mode_packet,
    radio_crc_debug,
    radio_crc_ok,
    with_radio_crc,
)
from .control import (
    RADAR_REFEREE_CMD_ID,
    _bytes_from_value,
    _parse_int,
    _parse_json_update,
    _parse_referee_payload,
    _parse_rm_frame,
    build_rm_frame,
    bytes_from_value,
    parse_control_update,
    parse_int,
    parse_json_update,
    parse_referee_payload,
    parse_rm_frame,
)


__all__ = [name for name in globals() if not name.startswith("__")]
