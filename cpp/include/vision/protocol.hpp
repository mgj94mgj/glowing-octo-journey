#pragma once

#include "vision/types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vision::protocol {

inline constexpr std::uint8_t kFrameStart = 0xA5;
inline constexpr std::uint16_t kRadarAllDataCmd = 0x0305;
inline constexpr std::uint16_t kRadarDecisionCmd = 0x0121;
inline constexpr std::uint16_t kGameStatusCmd = 0x0001;
inline constexpr std::uint16_t kMarkProgressCmd = 0x020C;
inline constexpr std::uint16_t kVulnerabilityCmd = 0x020E;
inline constexpr std::uint16_t kDartTargetCmd = 0x0105;

std::uint8_t crc8(const std::uint8_t* data, std::size_t size);
std::uint8_t crc8(const std::vector<std::uint8_t>& data);
std::uint16_t crc16(const std::uint8_t* data, std::size_t size);
std::uint16_t crc16(const std::vector<std::uint8_t>& data);

std::vector<std::uint8_t> build_data_radar_all(
    const std::map<std::string, SerialPoint>& send_map,
    char state
);

std::vector<std::uint8_t> build_data_decision(
    int radar_cmd,
    char state,
    int password_cmd = 0,
    const std::array<int, 6>* password = nullptr
);

struct BuiltPacket {
    std::vector<std::uint8_t> bytes;
    int next_sequence = 0;
};

BuiltPacket build_send_packet(
    const std::vector<std::uint8_t>& data,
    int sequence,
    std::uint16_t command_id
);

struct ParsedFrame {
    std::uint16_t command_id = 0;
    std::vector<std::uint8_t> data;
    std::uint8_t sequence = 0;
};

std::optional<ParsedFrame> parse_frame(const std::vector<std::uint8_t>& frame);

RefereeStatus parse_radar_decision(std::uint8_t value);

struct GameStatus {
    int game_type = 0;
    int game_progress = 0;
    int stage_remain_time = 0;
    std::uint64_t sync_timestamp = 0;
};

std::optional<GameStatus> parse_game_status(const std::vector<std::uint8_t>& payload);

std::vector<int> generate_random_password();
std::string password_to_string(const std::vector<int>& password);

struct RefereeReceiverStats {
    std::uint64_t bytes = 0;
    std::uint64_t header_crc8_errors = 0;
    std::uint64_t complete_frames = 0;
    std::uint64_t crc16_ok = 0;
    std::uint64_t crc16_errors = 0;
    std::map<std::uint16_t, std::uint64_t> commands;
    std::uint64_t startup_resync_discarded = 0;
};

class RefereeFrameBuffer {
public:
    std::vector<ParsedFrame> consume(const std::vector<std::uint8_t>& bytes);
    void reset();
    const RefereeReceiverStats& stats() const;
    RefereeReceiverStats take_stats();

private:
    std::vector<std::uint8_t> buffer_;
    bool synced_ = false;
    RefereeReceiverStats stats_;
};

}  // namespace vision::protocol
