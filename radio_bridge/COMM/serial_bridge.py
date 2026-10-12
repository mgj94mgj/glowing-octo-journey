#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import threading
import time
from typing import Dict, Optional

from .control import parse_control_update
from .radio_packets import (
    build_radio_jamming_key_packet,
    RadioInfoWaveState,
)


def _status_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _status_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_status_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _control_update_status(update) -> Dict:
    if update is None:
        return {}
    status = {
        "team": getattr(update, "team", None),
        "level": getattr(update, "level", None),
        "robot_id": getattr(update, "robot_id", None),
        "radar_info": getattr(update, "radar_info", None),
        "protocol_version": getattr(update, "protocol_version", None),
        "valid_mask": getattr(update, "valid_mask", None),
        "seq": getattr(update, "seq", None),
        "game_type": getattr(update, "game_type", None),
        "game_progress": getattr(update, "game_progress", None),
        "stage_remain_time": getattr(update, "stage_remain_time", None),
        "random_payload": bool(getattr(update, "random_payload", False)),
        "random_key": bool(getattr(update, "random_key", False)),
    }
    referee_status = getattr(update, "referee_status", None)
    if referee_status is not None and hasattr(referee_status, "as_dict"):
        status["referee_status"] = referee_status.as_dict()
    return status


class FeedbackStatusUdpPublisher:
    """Best-effort UDP JSON mirror for lower-computer feedback status."""

    def __init__(self, host: str, port: int, label: str = "feedback-status"):
        self._host = str(host or "").strip()
        self._port = int(port or 0)
        self._label = str(label or "feedback-status")
        self._sock = None
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._host) and self._port > 0

    def send(self, status: Dict) -> bool:
        if not self.enabled:
            return False

        record = dict(status or {})
        with self._lock:
            self._seq += 1
            record.setdefault("seq", self._seq)
            record.setdefault("label", self._label)
            record.setdefault("time_unix_s", time.time())
            record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            payload = json.dumps(
                _status_json_safe(record),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            try:
                if self._sock is None:
                    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.sendto(payload, (self._host, self._port))
            except Exception as exc:
                print(f"[feedback-status] publish failed udp://{self._host}:{self._port}: {exc}")
                return False
        return True

    def close(self):
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


class Ros2LowerComputerPublisher:
    """Publish lower-computer radio packets as raw bytes on a ROS2 topic."""

    def __init__(
        self,
        topic: str,
        node_name: str = "roboframe_lower_computer_tx",
        qos_depth: int = 10,
        label: str = "ROS2 lower-computer publisher",
        node=None,
        msg_type=None,
        rclpy_module=None,
    ):
        self._topic = str(topic or "").strip()
        self._node_name = str(node_name or "roboframe_lower_computer_tx").strip()
        self._qos_depth = int(qos_depth)
        self._label = str(label or "ROS2 lower-computer publisher")
        self._node = node
        self._msg_type = msg_type
        self._rclpy = rclpy_module
        self._publisher = None
        self._owns_node = False
        self._owns_context = False
        self._info_wave_state = RadioInfoWaveState()
        self._tx_packets = 0
        self._tx_errors = 0

    @property
    def enabled(self) -> bool:
        return bool(self._topic)

    def start(self) -> bool:
        return self._ensure_publisher() is not None

    def _ensure_publisher(self):
        if not self.enabled:
            return None
        if self._publisher is not None:
            return self._publisher

        try:
            if self._msg_type is None:
                from std_msgs.msg import UInt8MultiArray

                self._msg_type = UInt8MultiArray

            if self._node is None:
                if self._rclpy is None:
                    import rclpy

                    self._rclpy = rclpy
                already_initialized = bool(self._rclpy.ok()) if hasattr(self._rclpy, "ok") else False
                if not already_initialized:
                    self._rclpy.init(args=None)
                    self._owns_context = True
                self._node = self._rclpy.create_node(self._node_name)
                self._owns_node = True

            self._publisher = self._node.create_publisher(self._msg_type, self._topic, self._qos_depth)
        except Exception as exc:
            self._tx_errors += 1
            print(f"[ros2] {self._label} unavailable topic={self._topic}: {exc}")
            self._publisher = None
            return None

        print(f"[ros2] {self._label} publishing raw bytes -> {self._topic}")
        return self._publisher

    def send_raw(self, payload: bytes) -> bool:
        if not self.enabled:
            return False

        publisher = self._ensure_publisher()
        if publisher is None:
            return False

        raw = bytes(payload or b"")
        try:
            message = self._msg_type()
            message.data = [int(byte) & 0xFF for byte in raw]
            publisher.publish(message)
        except Exception as exc:
            self._tx_errors += 1
            print(f"[ros2] {self._label} publish failed: {exc}")
            return False

        self._tx_packets += 1
        return True

    def send_wave_packet(self, wave: str, cmd_id: int, data: bytes, extra: Optional[Dict] = None) -> bool:
        data_bytes = bytes(data or b"")
        cmd = int(cmd_id) & 0xFFFF
        if cmd == 0x0A06:
            return self.send_raw(build_radio_jamming_key_packet(data_bytes))

        if self._info_wave_state.update_from_frame(cmd, data_bytes):
            return self.send_raw(self._info_wave_state.build_packet())

        print(f"[ros2] {self._label} skip unsupported decoded cmd=0x{cmd:04X} len={len(data_bytes)}")
        return False

    def close(self):
        if self._node is not None and self._owns_node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        self._publisher = None

        if self._rclpy is not None and self._owns_context:
            try:
                if not hasattr(self._rclpy, "ok") or self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass
        self._owns_node = False
        self._owns_context = False


class Ros2LowerComputerSubscriber:
    """Subscribe lower-computer feedback raw bytes from a ROS2 topic."""

    def __init__(
        self,
        topic: str,
        on_update,
        node_name: str = "roboframe_lower_computer_rx",
        qos_depth: int = 10,
        label: str = "ROS2 lower-computer subscriber",
        node=None,
        msg_type=None,
        rclpy_module=None,
        on_status=None,
    ):
        self._topic = str(topic or "").strip()
        self._on_update = on_update
        self._node_name = str(node_name or "roboframe_lower_computer_rx").strip()
        self._qos_depth = int(qos_depth)
        self._label = str(label or "ROS2 lower-computer subscriber")
        self._node = node
        self._msg_type = msg_type
        self._rclpy = rclpy_module
        self._on_status = on_status
        self._subscription = None
        self._owns_node = False
        self._owns_context = False
        self._stop_event = threading.Event()
        self._spin_thread = None
        self._rx_packets = 0
        self._rx_errors = 0

    @property
    def enabled(self) -> bool:
        return bool(self._topic)

    def _status_base(self) -> Dict:
        return {
            "link": "ros2",
            "label": self._label,
            "topic": self._topic,
            "node_name": self._node_name,
            "rx_packets": int(self._rx_packets),
            "rx_errors": int(self._rx_errors),
        }

    def _emit_status(self, event: str, **fields):
        if self._on_status is None:
            return
        status = self._status_base()
        status["event"] = str(event)
        status.update(fields)
        try:
            self._on_status(status)
        except Exception as exc:
            print(f"[ros2] {self._label} status callback failed: {exc}")

    def start(self) -> bool:
        subscription = self._ensure_subscription()
        if subscription is None:
            return False

        if self._rclpy is not None and self._node is not None and self._spin_thread is None:
            self._stop_event.clear()
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()
        return True

    def _ensure_subscription(self):
        if not self.enabled:
            return None
        if self._subscription is not None:
            return self._subscription

        try:
            if self._msg_type is None:
                from std_msgs.msg import UInt8MultiArray

                self._msg_type = UInt8MultiArray

            if self._node is None:
                if self._rclpy is None:
                    import rclpy

                    self._rclpy = rclpy
                already_initialized = bool(self._rclpy.ok()) if hasattr(self._rclpy, "ok") else False
                if not already_initialized:
                    self._rclpy.init(args=None)
                    self._owns_context = True
                self._node = self._rclpy.create_node(self._node_name)
                self._owns_node = True

            self._subscription = self._node.create_subscription(
                self._msg_type,
                self._topic,
                self._on_message,
                self._qos_depth,
            )
        except Exception as exc:
            self._rx_errors += 1
            print(f"[ros2] {self._label} unavailable topic={self._topic}: {exc}")
            self._emit_status("unavailable", error=str(exc))
            self._subscription = None
            return None

        print(f"[ros2] {self._label} listening raw bytes <- {self._topic}")
        self._emit_status("listening")
        return self._subscription

    def _spin_loop(self):
        while not self._stop_event.is_set():
            try:
                if hasattr(self._rclpy, "spin_once"):
                    self._rclpy.spin_once(self._node, timeout_sec=0.1)
                else:
                    time.sleep(0.1)
            except Exception as exc:
                self._rx_errors += 1
                print(f"[ros2] {self._label} spin failed: {exc}")
                self._emit_status("spin_failed", error=str(exc))
                self._stop_event.wait(0.25)

    def _on_message(self, message):
        data = getattr(message, "data", None)
        try:
            raw = bytes(int(byte) & 0xFF for byte in data)
        except Exception as exc:
            self._rx_errors += 1
            print(f"[ros2] {self._label} ignored malformed message: {exc}")
            self._emit_status("malformed_message", error=str(exc))
            return

        self.receive_raw(raw, source="ros2")

    def receive_raw(self, raw: bytes, source: str = "ros2") -> bool:
        raw = bytes(raw or b"")
        if not raw:
            return False

        try:
            update = parse_control_update(raw)
        except Exception as exc:
            self._rx_errors += 1
            print(f"[ros2] {self._label} ignored {source} len={len(raw)} raw={raw.hex()}: {exc}")
            self._emit_status("ignored_packet", source=source, raw_len=len(raw), error=str(exc))
            return False

        self._rx_packets += 1
        print(
            f"[ros2] {self._label} rx update source={source} "
            f"team={update.team or '-'} level={update.level or '-'} "
            f"robot_id={update.robot_id if update.robot_id is not None else '-'} "
            f"radar_info={f'0x{update.radar_info:02X}' if update.radar_info is not None else '-'}"
        )
        self._emit_status("rx_update", source=source, raw_len=len(raw), **_control_update_status(update))

        try:
            self._on_update(update)
        except Exception as exc:
            self._rx_errors += 1
            print(f"[ros2] {self._label} update failed: {exc}")
            self._emit_status("update_failed", source=source, error=str(exc))
            return False
        return True

    def close(self):
        self._stop_event.set()
        thread = self._spin_thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._spin_thread = None

        if self._node is not None and self._owns_node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        self._subscription = None

        if self._rclpy is not None and self._owns_context:
            try:
                if not hasattr(self._rclpy, "ok") or self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass
        self._owns_node = False
        self._owns_context = False


__all__ = [
    "FeedbackStatusUdpPublisher",
    "Ros2LowerComputerPublisher",
    "Ros2LowerComputerSubscriber",
]
