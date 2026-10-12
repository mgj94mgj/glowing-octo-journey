#include "vision/protocol.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace vision::protocol {
namespace {

void append_u16_le(std::vector<std::uint8_t>& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        output.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU));
    }
}

std::uint16_t read_u16_le(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8U);
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

}  // namespace

std::uint8_t crc8(const std::uint8_t* data, std::size_t size) {
    std::uint8_t crc = 0xFFU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x01U) != 0U
                      ? static_cast<std::uint8_t>((crc >> 1U) ^ 0x8CU)
                      : static_cast<std::uint8_t>(crc >> 1U);
        }
    }
    return crc;
}

std::uint8_t crc8(const std::vector<std::uint8_t>& data) {
    return crc8(data.data(), data.size());
}

std::uint16_t crc16(const std::uint8_t* data, std::size_t size) {
    std::uint16_t crc = 0xFFFFU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x0001U) != 0U
                      ? static_cast<std::uint16_t>((crc >> 1U) ^ 0x8408U)
                      : static_cast<std::uint16_t>(crc >> 1U);
        }
    }
    return crc;
}

std::uint16_t crc16(const std::vector<std::uint8_t>& data) {
    return crc16(data.data(), data.size());
}

std::vector<std::uint8_t> build_data_radar_all(
    const std::map<std::string, SerialPoint>& send_map,
    char state
) {
    static constexpr std::array<const char*, 6> roles = {"1", "2", "3", "4", "6", "7"};
    const std::string opponent_prefix = state == 'R' ? "B" : "R";
    const std::string ally_prefix = state == 'R' ? "R" : "B";

    std::vector<std::uint8_t> data;
    data.reserve(48);
    for (const std::string& prefix : {opponent_prefix, ally_prefix}) {
        for (const char* role : roles) {
            const auto iterator = send_map.find(prefix + role);
            const SerialPoint point =
                iterator == send_map.end() ? SerialPoint{} : iterator->second;
            append_u16_le(data, static_cast<std::uint16_t>(
                                    std::clamp(
                                        point.x,
                                        0,
                                        static_cast<int>(std::numeric_limits<std::uint16_t>::max())
                                    )
                                ));
            append_u16_le(data, static_cast<std::uint16_t>(
                                    std::clamp(
                                        point.y,
                                        0,
                                        static_cast<int>(std::numeric_limits<std::uint16_t>::max())
                                    )
                                ));
        }
    }
    return data;
}

std::vector<std::uint8_t> build_data_decision(
    int radar_cmd,
    char state,
    int password_cmd,
    const std::array<int, 6>* password
) {
    std::vector<std::uint8_t> data;
    data.reserve(14);
    data.push_back(0x21);
    data.push_back(0x01);
    append_u16_le(data, state == 'R' ? 9U : 109U);
    data.push_back(0x80);
    data.push_back(0x80);
    data.push_back(static_cast<std::uint8_t>(radar_cmd & 0xFF));
    data.push_back(static_cast<std::uint8_t>(password_cmd & 0xFF));
    for (int i = 0; i < 6; ++i) {
        data.push_back(static_cast<std::uint8_t>(password == nullptr ? 0 : (*password)[i] & 0xFF));
    }
    return data;
}

BuiltPacket build_send_packet(
    const std::vector<std::uint8_t>& data,
    int sequence,
    std::uint16_t command_id
) {
    const auto data_length = static_cast<std::uint16_t>(data.size());
    std::vector<std::uint8_t> packet;
    packet.reserve(5 + 2 + data.size() + 2);

    packet.push_back(kFrameStart);
    append_u16_le(packet, data_length);
    packet.push_back(static_cast<std::uint8_t>(sequence & 0xFF));
    packet.push_back(crc8(packet.data(), packet.size()));

    packet.push_back(static_cast<std::uint8_t>((command_id >> 8U) & 0xFFU));
    packet.push_back(static_cast<std::uint8_t>(command_id & 0xFFU));
    packet.insert(packet.end(), data.begin(), data.end());
    append_u16_le(packet, crc16(packet));

    return BuiltPacket{std::move(packet), (sequence + 1) & 0xFF};
}

std::optional<ParsedFrame> parse_frame(const std::vector<std::uint8_t>& frame) {
    if (frame.size() < 9 || frame[0] != kFrameStart) {
        return std::nullopt;
    }
    const std::uint16_t data_length = read_u16_le(frame.data() + 1);
    if (frame.size() < static_cast<std::size_t>(9) + data_length) {
        return std::nullopt;
    }
    if (crc8(frame.data(), 4) != frame[4]) {
        return std::nullopt;
    }
    const std::uint16_t expected_crc = read_u16_le(frame.data() + 7 + data_length);
    if (crc16(frame.data(), 7 + data_length) != expected_crc) {
        return std::nullopt;
    }

    ParsedFrame parsed;
    parsed.sequence = frame[3];
    parsed.command_id =
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(frame[5]) << 8U) |
        static_cast<std::uint16_t>(frame[6]);
    parsed.data.assign(frame.begin() + 7, frame.begin() + 7 + data_length);
    return parsed;
}

RefereeStatus parse_radar_decision(std::uint8_t value) {
    RefereeStatus status;
    status.double_vulnerability_chance = value & 0x03U;
    status.opponent_double_vulnerability = (value >> 2U) & 0x01U;
    status.encryption_level = (value >> 3U) & 0x03U;
    status.key_modifiable = (value >> 5U) & 0x01U;
    return status;
}

std::optional<GameStatus> parse_game_status(const std::vector<std::uint8_t>& payload) {
    if (payload.size() < 11) {
        return std::nullopt;
    }
    GameStatus status;
    const std::uint8_t status_byte = payload[0];
    status.game_type = status_byte & 0x0FU;
    status.game_progress = (status_byte >> 4U) & 0x0FU;
    status.stage_remain_time = read_u16_le(payload.data() + 1);
    status.sync_timestamp = read_u64_le(payload.data() + 3);
    return status;
}

std::vector<int> generate_random_password() {
    static constexpr char alphabet[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<std::size_t> distribution(0, sizeof(alphabet) - 2);
    std::vector<int> password(6);
    for (int& value : password) {
        value = static_cast<unsigned char>(alphabet[distribution(generator)]);
    }
    return password;
}

std::string password_to_string(const std::vector<int>& password) {
    std::string output;
    output.reserve(password.size());
    for (int value : password) {
        output.push_back(static_cast<char>(value & 0xFF));
    }
    return output;
}

std::vector<ParsedFrame> RefereeFrameBuffer::consume(const std::vector<std::uint8_t>& bytes) {
    std::vector<ParsedFrame> frames;
    buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
    stats_.bytes += bytes.size();

    while (true) {
        const auto start = std::find(buffer_.begin(), buffer_.end(), kFrameStart);
        if (start == buffer_.end()) {
            buffer_.clear();
            break;
        }
        if (start != buffer_.begin()) {
            buffer_.erase(buffer_.begin(), start);
        }
        if (buffer_.size() < 5) {
            break;
        }

        const std::uint8_t actual_crc8 = crc8(buffer_.data(), 4);
        if (actual_crc8 != buffer_[4]) {
            if (synced_) {
                ++stats_.header_crc8_errors;
            } else {
                ++stats_.startup_resync_discarded;
            }
            buffer_.erase(buffer_.begin());
            continue;
        }

        const std::uint16_t data_length = read_u16_le(buffer_.data() + 1);
        const std::size_t packet_length = 5 + 2 + data_length + 2;
        if (buffer_.size() < packet_length) {
            break;
        }

        ++stats_.complete_frames;
        const std::uint16_t received_crc16 = read_u16_le(buffer_.data() + packet_length - 2);
        const std::uint16_t calculated_crc16 = crc16(buffer_.data(), packet_length - 2);
        if (received_crc16 != calculated_crc16) {
            if (synced_) {
                ++stats_.crc16_errors;
            } else {
                ++stats_.startup_resync_discarded;
            }
            buffer_.erase(buffer_.begin());
            continue;
        }

        synced_ = true;
        ++stats_.crc16_ok;
        const std::uint16_t command_id =
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(buffer_[5]) << 8U) |
            static_cast<std::uint16_t>(buffer_[6]);
        ++stats_.commands[command_id];

        ParsedFrame parsed;
        parsed.command_id = command_id;
        parsed.sequence = buffer_[3];
        parsed.data.assign(buffer_.begin() + 7, buffer_.begin() + 7 + data_length);
        frames.push_back(std::move(parsed));
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(packet_length));
    }
    return frames;
}

void RefereeFrameBuffer::reset() {
    buffer_.clear();
    synced_ = false;
    stats_ = {};
}

const RefereeReceiverStats& RefereeFrameBuffer::stats() const {
    return stats_;
}

RefereeReceiverStats RefereeFrameBuffer::take_stats() {
    RefereeReceiverStats output = stats_;
    stats_ = {};
    return output;
}

}  // namespace vision::protocol
