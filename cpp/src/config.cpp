#include "vision/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace vision {
namespace {

template <typename T>
T scalar(const YAML::Node& node, const T& fallback) {
    if (!node || !node.IsScalar()) {
        return fallback;
    }
    try {
        return node.as<T>();
    } catch (const YAML::Exception&) {
        return fallback;
    }
}

std::string text(const YAML::Node& node, const std::string& fallback) {
    return scalar<std::string>(node, fallback);
}

cv::Size size(const YAML::Node& node, const cv::Size& fallback) {
    if (!node || !node.IsSequence() || node.size() < 2) {
        return fallback;
    }
    return cv::Size(node[0].as<int>(), node[1].as<int>());
}

std::string resolve_path(const fs::path& root, const std::string& value) {
    if (value.empty()) {
        return value;
    }
    fs::path path(value);
    return path.is_absolute() ? path.string() : (root / path).lexically_normal().string();
}

}  // namespace

fs::path default_project_root() {
    fs::path current = fs::current_path();
    if (fs::exists(current / "config.yaml")) {
        return current;
    }
    return current.parent_path();
}

AppConfig ConfigLoader::load(const fs::path& path) {
    const YAML::Node root = YAML::LoadFile(path.string());
    if (!root.IsMap()) {
        throw std::runtime_error("config.yaml root must be a mapping");
    }

    AppConfig config;
    config.source_root = path.parent_path().string();

    const YAML::Node global = root["global"];
    config.state = static_cast<char>(std::toupper(scalar<std::string>(global["state"], "B").empty()
                                                     ? 'B'
                                                     : scalar<std::string>(global["state"], "B")[0]));
    config.camera_mode = text(global["camera_mode"], "test");
    config.multi_car_recognition = scalar<bool>(global["multi_car_recognition"], true);
    config.use_serial = scalar<bool>(global["use_serial"], true);
    config.save_img = scalar<bool>(global["save_img"], false);
    config.map_size = size(global["map_size"], cv::Size(2800, 1500));

    const YAML::Node test = global["test"];
    config.test_video_path = text(test["video_path"], "");
    config.test_loop = scalar<bool>(test["loop"], true);
    config.test_start_frame = scalar<int>(test["start_frame"], 0);
    config.test_playback_speed = scalar<double>(test["playback_speed"], 1.0);

    const YAML::Node serial = root["serial"];
    config.serial_port = text(serial["port"], config.serial_port);
    config.serial_baudrate = scalar<int>(serial["baudrate"], config.serial_baudrate);
    config.serial_timeout = scalar<double>(serial["timeout"], config.serial_timeout);

    const YAML::Node gimbal = root["gimbal"];
    config.gimbal_enabled = scalar<bool>(gimbal["enabled"], config.gimbal_enabled);
    config.gimbal_port = text(gimbal["port"], config.gimbal_port);
    config.gimbal_baudrate = scalar<int>(gimbal["baudrate"], config.gimbal_baudrate);
    config.gimbal_timeout = scalar<double>(gimbal["timeout"], config.gimbal_timeout);
    config.gimbal_min_interval_s = scalar<double>(gimbal["min_interval_s"], config.gimbal_min_interval_s);
    config.gimbal_debug = scalar<bool>(gimbal["debug"], config.gimbal_debug);

    const YAML::Node radio = root["radio"];
    config.info_wave_timeout_s = scalar<double>(radio["info_wave_timeout_s"], config.info_wave_timeout_s);

    const YAML::Node filter = root["filter"];
    config.filter_type = text(filter["type"], config.filter_type);
    const YAML::Node sliding = filter["sliding_window"];
    config.sliding_window_size = scalar<int>(sliding["window_size"], config.sliding_window_size);
    config.sliding_max_inactive_time =
        scalar<double>(sliding["max_inactive_time"], config.sliding_max_inactive_time);
    config.sliding_threshold = scalar<double>(sliding["threshold"], config.sliding_threshold);

    const YAML::Node hybrid = filter["hybrid_gated_kalman"];
    config.hybrid_prefilter_window = scalar<int>(hybrid["prefilter_window"], config.hybrid_prefilter_window);
    config.hybrid_prefilter_mode = text(hybrid["prefilter_mode"], config.hybrid_prefilter_mode);
    config.hybrid_max_inactive_time =
        scalar<double>(hybrid["max_inactive_time"], config.hybrid_max_inactive_time);
    config.hybrid_raw_jump_threshold =
        scalar<double>(hybrid["raw_jump_threshold"], config.hybrid_raw_jump_threshold);
    config.hybrid_pred_jump_threshold =
        scalar<double>(hybrid["pred_jump_threshold"], config.hybrid_pred_jump_threshold);
    config.hybrid_process_noise = scalar<double>(hybrid["process_noise"], config.hybrid_process_noise);
    config.hybrid_measurement_noise =
        scalar<double>(hybrid["measurement_noise"], config.hybrid_measurement_noise);

    const YAML::Node compare = root["filter_compare"];
    config.filter_compare.enabled = scalar<bool>(compare["enabled"], false);
    config.filter_compare.compare_type = text(compare["compare_type"], config.filter_compare.compare_type);
    config.filter_compare.draw_on_map = scalar<bool>(compare["draw_on_map"], true);
    config.filter_compare.show_delta_text = scalar<bool>(compare["show_delta_text"], true);

    const YAML::Node paths = root["paths"];
    config.test_image_path = resolve_path(config.source_root, text(paths["test_img"], config.test_image_path));
    if (config.state == 'R') {
        config.calibration_path = resolve_path(config.source_root, text(paths["calibration"]["red"], ""));
        config.map_path = resolve_path(config.source_root, text(paths["map_images"]["red"], ""));
    } else {
        config.calibration_path = resolve_path(config.source_root, text(paths["calibration"]["blue"], ""));
        config.map_path = resolve_path(config.source_root, text(paths["map_images"]["blue"], ""));
    }
    config.mask_path = resolve_path(config.source_root, text(paths["map_images"]["mask"], ""));
    config.map_backup_path = resolve_path(config.source_root, text(paths["map_images"]["backup"], ""));

    const YAML::Node models = paths["models"];
    config.car_model = resolve_path(config.source_root, text(models["car"], ""));
    config.armor_model = resolve_path(config.source_root, text(models["armor"], ""));
    config.uav_model = resolve_path(config.source_root, text(models["uav"], ""));
    config.car_confidence = scalar<double>(models["car_confidence"], config.car_confidence);
    config.armor_confidence = scalar<double>(models["armor_confidence"], config.armor_confidence);
    config.uav_confidence = scalar<double>(models["uav_confidence"], config.uav_confidence);

    const YAML::Node uav = root["uav_detection"];
    config.uav_enabled = scalar<bool>(uav["enabled"], config.uav_enabled);
    config.uav_max_y = scalar<double>(uav["max_y"], config.uav_max_y);
    config.uav_max_det = scalar<int>(uav["max_det"], config.uav_max_det);
    config.uav_detect_interval_frames =
        std::max(1, scalar<int>(uav["detect_interval_frames"], config.uav_detect_interval_frames));
    if (uav["target_classes"] && uav["target_classes"].IsSequence()) {
        for (const auto& item : uav["target_classes"]) {
            std::string name = item.as<std::string>();
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            config.uav_target_classes.push_back(name);
        }
    }
    if (config.uav_target_classes.empty()) {
        config.uav_target_classes = {"uav", "drone"};
    }

    const YAML::Node camera = root["camera_params"];
    config.camera_exposure_time = scalar<double>(camera["exposure_time"], config.camera_exposure_time);
    config.camera_gain = scalar<double>(camera["gain"], config.camera_gain);
    config.camera_device_serial = text(camera["device_serial"], "");

    const YAML::Node blind = root["blind_zone"];
    config.blind_zone.enabled = scalar<bool>(blind["enabled"], config.blind_zone.enabled);
    config.blind_zone.base_time = scalar<double>(blind["base_time"], config.blind_zone.base_time);
    config.blind_zone.offset_time = scalar<double>(blind["offset_time"], config.blind_zone.offset_time);
    config.blind_zone.direction_weight =
        scalar<double>(blind["direction_weight"], config.blind_zone.direction_weight);
    config.blind_zone.distance_factor =
        scalar<double>(blind["distance_factor"], config.blind_zone.distance_factor);
    config.blind_zone.min_speed_mps = scalar<double>(blind["min_speed_mps"], config.blind_zone.min_speed_mps);
    config.blind_zone.motion_history_size =
        std::max<std::size_t>(2, scalar<int>(blind["motion_history_size"], config.blind_zone.motion_history_size));
    config.blind_zone.motion_reset_gap_s =
        scalar<double>(blind["motion_reset_gap_s"], config.blind_zone.motion_reset_gap_s);
    config.blind_zone.lost_x_offset_cm =
        scalar<double>(blind["lost_x_offset_cm"], config.blind_zone.lost_x_offset_cm);
    for (const auto& item : blind["points"]) {
        std::vector<SerialPoint> points;
        for (const auto& point : item.second) {
            if (point.IsSequence() && point.size() >= 2) {
                points.push_back(SerialPoint{point[0].as<int>(), point[1].as<int>()});
            }
        }
        config.blind_zone.points[item.first.as<std::string>()] = std::move(points);
    }

    const YAML::Node dv = root["double_vulnerability"];
    config.double_vulnerability.trigger_mode =
        text(dv["trigger_mode"], config.double_vulnerability.trigger_mode);
    config.double_vulnerability.stable_duration_s =
        scalar<double>(dv["stable_duration_s"], config.double_vulnerability.stable_duration_s);
    config.double_vulnerability.startup_limit_window_s =
        scalar<double>(dv["startup_limit_window_s"], config.double_vulnerability.startup_limit_window_s);
    config.double_vulnerability.startup_limit_max_uses =
        scalar<int>(dv["startup_limit_max_uses"], config.double_vulnerability.startup_limit_max_uses);
    config.double_vulnerability.distance_threshold =
        scalar<double>(dv["distance"]["threshold"], config.double_vulnerability.distance_threshold);
    config.double_vulnerability.close_distance =
        scalar<double>(dv["motion"]["close_distance"], config.double_vulnerability.close_distance);
    config.double_vulnerability.far_distance =
        scalar<double>(dv["motion"]["far_distance"], config.double_vulnerability.far_distance);
    config.double_vulnerability.approach_speed =
        scalar<double>(dv["motion"]["approach_speed"], config.double_vulnerability.approach_speed);
    config.double_vulnerability.active_speed =
        scalar<double>(dv["motion"]["active_speed"], config.double_vulnerability.active_speed);
    config.double_vulnerability.min_speed =
        scalar<double>(dv["motion"]["min_speed"], config.double_vulnerability.min_speed);

    const YAML::Node ui = root["ui"];
    config.info_panel_size = size(ui["info_panel_size"], config.info_panel_size);
    config.map_display_size = size(ui["map_display_size"], config.map_display_size);
    config.image_display_size = size(ui["image_display_size"], config.image_display_size);

    const YAML::Node recording = root["video_recording"];
    config.video_recording_enabled = scalar<bool>(recording["enabled"], false);
    config.video_recording_output_dir =
        text(recording["output_dir"], config.video_recording_output_dir);
    config.video_recording_fps = scalar<double>(recording["fps"], config.video_recording_fps);
    config.video_recording_crf = scalar<int>(recording["crf"], config.video_recording_crf);
    config.video_recording_preset = text(recording["preset"], config.video_recording_preset);
    config.video_recording_threads = scalar<int>(recording["threads"], config.video_recording_threads);
    config.video_recording_queue_size =
        std::max(1, scalar<int>(recording["queue_size"], config.video_recording_queue_size));

    if (config.filter_type != "sliding_window" &&
        config.filter_type != "hybrid_gated_kalman" &&
        config.filter_type != "hybrid") {
        throw std::runtime_error("filter.type must be sliding_window, hybrid_gated_kalman, or hybrid");
    }
    return config;
}

}  // namespace vision
