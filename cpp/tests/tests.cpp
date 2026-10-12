#include "vision/detector.hpp"
#include "vision/config.hpp"
#include "vision/filters.hpp"
#include "vision/geometry.hpp"
#include "vision/protocol.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void test_protocol() {
    const std::vector<std::uint8_t> header = {0xA5, 0x00, 0x00, 0x00};
    expect(vision::protocol::crc8(header) == 0xC3, "CRC8 reference mismatch");
    expect(vision::protocol::crc16(header) == 0xCF4B, "CRC16 reference mismatch");

    const auto packet = vision::protocol::build_send_packet(
        {0x01, 0x02, 0x03},
        7,
        vision::protocol::kRadarAllDataCmd
    );
    expect(packet.next_sequence == 8, "packet sequence did not advance");
    const auto parsed = vision::protocol::parse_frame(packet.bytes);
    expect(parsed.has_value(), "valid packet did not parse");
    expect(parsed->command_id == vision::protocol::kRadarAllDataCmd, "command id mismatch");
    expect(parsed->sequence == 7, "sequence mismatch");
    expect(parsed->data == std::vector<std::uint8_t>({0x01, 0x02, 0x03}), "payload mismatch");

    const auto decision = vision::protocol::parse_radar_decision(0b00111011);
    expect(decision.double_vulnerability_chance == 3, "double vulnerability bits mismatch");
    expect(decision.opponent_double_vulnerability == 0, "opponent status bits mismatch");
    expect(decision.encryption_level == 3, "encryption level bits mismatch");
    expect(decision.key_modifiable == 1, "key modifiable bit mismatch");

    const std::vector<std::uint8_t> game_payload = {
        0x43, 0xB4, 0x01, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    const auto game = vision::protocol::parse_game_status(game_payload);
    expect(game.has_value(), "game status did not parse");
    expect(game->game_type == 3, "game type mismatch");
    expect(game->game_progress == 4, "game progress mismatch");
    expect(game->stage_remain_time == 436, "stage remain mismatch");
}

void test_filters() {
    vision::SlidingWindowFilter sliding(3, 2.0, 100000.0);
    sliding.add_data("R1", 10.0, 20.0);
    sliding.add_data("R1", 12.0, 22.0);
    sliding.add_data("R1", 14.0, 24.0);
    const auto data = sliding.get_all_data();
    expect(data.count("R1") == 1, "sliding filter did not emit target");
    expect_near(data.at("R1").x, 12.0, 1e-6, "sliding filter x average mismatch");
    expect_near(data.at("R1").y, 22.0, 1e-6, "sliding filter y average mismatch");

    vision::HybridGatedKalmanFilter hybrid(
        3,
        "median",
        2.0,
        120.0,
        100.0,
        1e-5,
        1e-1
    );
    hybrid.add_data("B1", 100.0, 100.0);
    hybrid.add_data("B1", 101.0, 101.0);
    hybrid.add_data("B1", 102.0, 102.0);
    const auto hybrid_data = hybrid.get_all_data();
    expect(hybrid_data.count("B1") == 1, "hybrid filter did not emit target");
    expect_near(hybrid_data.at("B1").x, 101.0, 8.0, "hybrid filter x drifted");
    expect_near(hybrid_data.at("B1").y, 101.0, 8.0, "hybrid filter y drifted");
}

void test_nms() {
    const std::vector<vision::Detection> detections = {
        {"car", cv::Rect2f(10, 10, 100, 100), 0.9F},
        {"car", cv::Rect2f(12, 12, 98, 98), 0.8F},
        {"armor", cv::Rect2f(10, 10, 100, 100), 0.7F},
    };
    const auto selected = vision::non_max_suppression(detections, 0.5F, 10);
    expect(selected.size() == 2, "class-aware NMS suppressed the wrong boxes");
}

void test_calibration_loader() {
    const std::filesystem::path source_root(VISION_SOURCE_ROOT);
    const auto matrices = vision::load_calibration_npy(source_root / "arrays_test_blue.npy");
    expect(matrices.size() == 2, "calibration loader did not return two matrices");
    expect(matrices[0].rows == 3 && matrices[0].cols == 3, "calibration matrix shape mismatch");
    expect_near(
        matrices[0].at<double>(2, 2),
        1.0,
        1e-12,
        "homography normalization mismatch"
    );
}

void test_config_loader() {
    const std::filesystem::path source_root(VISION_SOURCE_ROOT);
    const vision::AppConfig config =
        vision::ConfigLoader::load(source_root / "config.yaml");
    expect(config.camera_mode == "hik", "config camera mode mismatch");
    expect(config.filter_type == "sliding_window", "config filter type mismatch");
    expect(config.blind_zone.points.count("R1") == 1, "config blind-zone points missing");
    expect(!config.car_model.empty(), "config car model path missing");
}

}  // namespace

int main() {
    try {
        test_protocol();
        test_filters();
        test_nms();
        test_calibration_loader();
        test_config_loader();
        std::cout << "all vision tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "test failed: " << exception.what() << '\n';
        return 1;
    }
}
