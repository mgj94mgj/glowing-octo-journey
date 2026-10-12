#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vision {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Detection {
    std::string cls;
    cv::Rect2f box;
    float confidence = 0.0F;
};

struct Point2f {
    float x = 0.0F;
    float y = 0.0F;
};

struct SerialPoint {
    int x = 0;
    int y = 0;
};

inline bool operator==(const SerialPoint& lhs, const SerialPoint& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(const SerialPoint& lhs, const SerialPoint& rhs) {
    return !(lhs == rhs);
}

struct RefereeStatus {
    int double_vulnerability_chance = -1;
    int opponent_double_vulnerability = -1;
    int encryption_level = 1;
    int key_modifiable = -1;
};

struct RadioPositionFrame {
    std::array<SerialPoint, 6> robots{};
    std::uint8_t valid_mask = 0;
};

struct FilterComparisonConfig {
    bool enabled = false;
    bool draw_on_map = true;
    bool show_delta_text = true;
    std::string compare_type = "hybrid_gated_kalman";
};

struct BlindZoneConfig {
    bool enabled = true;
    double base_time = 3.0;
    double offset_time = 1.7;
    double direction_weight = 0.003;
    double distance_factor = 0.01;
    double min_speed_mps = 0.15;
    std::size_t motion_history_size = 5;
    double motion_reset_gap_s = 1.0;
    double lost_x_offset_cm = -30.0;
    std::unordered_map<std::string, std::vector<SerialPoint>> points;
};

struct DoubleVulnerabilityConfig {
    std::string trigger_mode = "both";
    double stable_duration_s = 2.0;
    double startup_limit_window_s = 120.0;
    int startup_limit_max_uses = 1;
    double distance_threshold = 2.5;
    double close_distance = 3.0;
    double far_distance = 6.0;
    double approach_speed = 0.5;
    double active_speed = 0.5;
    double min_speed = 0.3;
};

struct AppConfig {
    char state = 'B';
    std::string camera_mode = "test";
    bool multi_car_recognition = true;
    bool use_serial = true;
    bool save_img = false;
    cv::Size map_size{2800, 1500};

    std::string test_video_path;
    bool test_loop = true;
    int test_start_frame = 0;
    double test_playback_speed = 1.0;
    std::string test_image_path = "images/test.jpg";

    std::string serial_port = "/dev/referee";
    int serial_baudrate = 115200;
    double serial_timeout = 1.0;

    bool gimbal_enabled = true;
    std::string gimbal_port = "/dev/gimbal";
    int gimbal_baudrate = 115200;
    double gimbal_timeout = 0.01;
    double gimbal_min_interval_s = 0.02;
    bool gimbal_debug = false;

    double info_wave_timeout_s = 1.0;
    std::string filter_type = "sliding_window";

    int sliding_window_size = 5;
    double sliding_max_inactive_time = 2.0;
    double sliding_threshold = 100000.0;

    int hybrid_prefilter_window = 3;
    std::string hybrid_prefilter_mode = "median";
    double hybrid_max_inactive_time = 2.0;
    double hybrid_raw_jump_threshold = 120.0;
    double hybrid_pred_jump_threshold = 100.0;
    double hybrid_process_noise = 1e-5;
    double hybrid_measurement_noise = 1e-1;

    FilterComparisonConfig filter_compare;
    std::string calibration_path;
    std::string map_path;
    std::string mask_path;
    std::string map_backup_path;

    std::string car_model;
    std::string armor_model;
    std::string uav_model;
    double car_confidence = 0.6;
    double armor_confidence = 0.6;
    double uav_confidence = 0.5;

    bool uav_enabled = false;
    double uav_max_y = 750.0;
    std::vector<std::string> uav_target_classes;
    int uav_max_det = 3;
    int uav_detect_interval_frames = 1;

    double camera_exposure_time = 17000.0;
    double camera_gain = 17.9;
    std::string camera_device_serial;

    BlindZoneConfig blind_zone;
    DoubleVulnerabilityConfig double_vulnerability;

    cv::Size info_panel_size{420, 500};
    cv::Size map_display_size{900, 480};
    cv::Size image_display_size{980, 700};

    bool video_recording_enabled = false;
    std::string video_recording_output_dir = "/media/ajian/AJian/radar_recordings";
    double video_recording_fps = 30.0;
    int video_recording_crf = 23;
    std::string video_recording_preset = "ultrafast";
    int video_recording_threads = 2;
    int video_recording_queue_size = 20;

    std::string source_root;
};

}  // namespace vision
