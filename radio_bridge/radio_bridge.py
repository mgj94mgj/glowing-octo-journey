import queue
import serial
import struct
import collections
import time
import os
import threading
import json

from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
from .COMM.radio_packets import (
    RADIO_JAMMING_KEY_RX_PACKET_ID,
    RADIO_INFO_POSITION_VALID,
    RADIO_INFO_WAVE_RX_PACKET_ID,
    _INFO_WAVE_FORMAT,
)

# Constants
RX_BUF_SIZE = 1024
MAX_PACKAGE_SIZE = 128
START_BYTE = 0xA5
CRC16_SIZE = 2
FRAME_HEADER_SIZE = 5

class SerialState(Enum):
    CLOSED = 0
    SCANNING = 3
    OPENNED = 1
    RECOVERING = 2

class PackageStatus:
    HEADER_INCOMPLETE = 1
    HEADER_CRC_ERROR = 2
    PAYLOAD_INCOMPLETE = 3
    CRC_ERROR = 4
    PACKAGE_COMPLETE = 5

class RadioListener(Node):

    def __init__(self,data_queue:queue.Queue,key_queue:queue.Queue):
        super().__init__("radio_listener")

        self.subscription = self.create_subscription(
            UInt8MultiArray,
            "/Radio",
            self.callback,
            10
        )
        self.radio_data = {}
        self.data_queue = data_queue
        self.key_queue = key_queue
        self.get_logger().info("Listening /Radio ...")

    def _put_latest_info_wave(self, data):
        """队列满时淘汰旧帧，确保主程序只处理最新信息波。"""
        while True:
            try:
                self.data_queue.put_nowait(data)
                return
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    continue

    def callback(self, msg):

        raw = bytes(msg.data)

        if len(raw) < 3:
            return

        # 判断是否为信息波
        if raw[0] == RADIO_INFO_WAVE_RX_PACKET_ID:
            # self.get_logger().warning(f"Unknown Packet ID: 0x{raw[0]:02X}")
            # return
            # 去掉 PacketID 和 CRC16
            payload = raw[1:-2]

            try:
                values = struct.unpack(_INFO_WAVE_FORMAT, payload)
            except struct.error as e:
                self.get_logger().error(f"Unpack failed: {e}")
                return

            idx = 0

            # ---------- Valid Mask ----------
            self.radio_data["valid_mask"] = values[idx]
            idx += 1

            # ---------- Position ----------
            # position_keys = (
            #     "hero_position", "hero_y",
            #     "engineer_x", "engineer_y",
            #     "infantry_3_x", "infantry_3_y",
            #     "infantry_4_x", "infantry_4_y",
            #     "aerial_x", "aerial_y",
            #     "sentry_x", "sentry_y",
            # )
            position_keys = (
                "hero",
                "engineer",
                "infantry_3",
                "infantry_4",
                "aerial",
                "sentry",
            )
            for key in position_keys:
                self.radio_data[key] = [values[idx], values[idx + 1]]

                idx += 2

            # ---------- HP ----------
            hp_keys = (
                "hero_hp",
                "engineer_hp",
                "infantry_3_hp",
                "infantry_4_hp",
                "reserved",
                "sentry_hp",
            )

            for key in hp_keys:
                self.radio_data[key] = values[idx]
                idx += 1

            # ---------- Bullet ----------
            bullet_keys = (
                "hero_ammo",
                "infantry_3_ammo",
                "infantry_4_ammo",
                "aerial_ammo",
                "sentry_ammo",
            )

            for key in bullet_keys:
                self.radio_data[key] = values[idx]
                idx += 1

            # ---------- Gold ----------
            self.radio_data["remaining_gold"] = values[idx]
            idx += 1

            self.radio_data["total_gold"] = values[idx]
            idx += 1

            # ---------- Macro ----------
            macro = values[idx]
            idx += 1

            self.radio_data["macro_bits"] = macro

            self.radio_data["opponent_supply_zone_occupied"] = macro & 0x01
            self.radio_data["opponent_center_highland_occupancy"] = (macro >> 1) & 0x03
            self.radio_data["opponent_trapezoid_highland_occupied"] = (macro >> 3) & 0x01
            self.radio_data["opponent_fortress_buff_occupancy"] = (macro >> 4) & 0x03
            self.radio_data["opponent_outpost_buff_occupancy"] = (macro >> 6) & 0x03
            self.radio_data["opponent_base_buff_occupied"] = (macro >> 8) & 0x01
            self.radio_data["terrain_near_opponent_ramp_front_card"] = (macro >> 9) & 0x01
            self.radio_data["terrain_near_opponent_ramp_back_card"] = (macro >> 10) & 0x01
            self.radio_data["terrain_near_own_ramp_front_card"] = (macro >> 11) & 0x01
            self.radio_data["terrain_near_own_ramp_back_card"] = (macro >> 12) & 0x01
            self.radio_data["terrain_opponent_highland_upper_card"] = (macro >> 13) & 0x01
            self.radio_data["terrain_opponent_ramp_back_card"] = (macro >> 14) & 0x01
            self.radio_data["terrain_opponent_road_upper_card"] = (macro >> 15) & 0x01

            # ---------- Buff ----------
            prefixes = (
                "hero",
                "engineer",
                "infantry_3",
                "infantry_4",
                "sentry",
            )

            for prefix in prefixes:
                self.radio_data[f"{prefix}_health_regen"] = values[idx]
                idx += 1

                self.radio_data[f"{prefix}_cooling_boost"] = values[idx]
                idx += 1

                self.radio_data[f"{prefix}_defense_boost"] = values[idx]
                idx += 1

                self.radio_data[f"{prefix}_defense_debuff"] = values[idx]
                idx += 1

                self.radio_data[f"{prefix}_attack_boost"] = values[idx]
                idx += 1

            # ---------- Sentry ----------
            self.radio_data["sentry_posture"] = values[idx]

            # 坐标字段无效时不覆盖相机识别位置。
            if not (self.radio_data["valid_mask"] & RADIO_INFO_POSITION_VALID):
                self.get_logger().warning("信息波坐标字段无效，已忽略该帧")
                return

            # 只保留最新信息波，避免无线输入快于主循环时积压旧坐标。
            self._put_latest_info_wave(self.radio_data.copy())
        elif raw[0] == RADIO_JAMMING_KEY_RX_PACKET_ID:
            payload = raw[1:-2]

            if len(payload) < 6:
                self.get_logger().warning(
                    f"Jamming key payload too short: {len(payload)} bytes"
                )
                return

            key_bytes = payload[:6]

            key = "".join(
                chr(byte) if 32 <= byte <= 126 else "."
                for byte in key_bytes
            )

            #self.radio_data["key"] = key

            self.key_queue.put(key)
class FeedbackPublisher(Node):
    def __init__(self,Feedback_data:queue.Queue):
        super().__init__("lower_feedback_publisher")

        self.publisher = self.create_publisher(
            UInt8MultiArray,
            "/RadioFeedback",
            1
        )
        self.Feedback_data = Feedback_data
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.stage_remain_time = 420
        self.feedback_data=None
        # self.freeback_data = {
        #     "team": "red",
        #     "level": "1",
        #
        #     # 比赛状态，4 一般表示比赛进行中
        #     "game_progress": 4,
        #
        #     # 比赛剩余时间，单位秒
        #     "stage_remain_time": self.stage_remain_time,
        # }
        self.get_logger().info("Publishing lower feedback to /RadioFeedback")

    def timer_callback(self):
        # 模拟下位机反馈给上位机的数据
        if not self.Feedback_data.empty():
            self.feedback_data = self.Feedback_data.get_nowait()
        if self.feedback_data is not None:
            raw = json.dumps(self.feedback_data).encode("utf-8")

            msg = UInt8MultiArray()
            msg.data = [int(byte) & 0xFF for byte in raw]

            self.publisher.publish(msg)

            self.get_logger().info(
                f"send /RadioFeedback: {self.feedback_data}"
            )

            # if self.stage_remain_time > 0:
            #     self.stage_remain_time -= 1
