#include "vision/camera.hpp"
#include "vision/config.hpp"
#include "vision/detector.hpp"
#include "vision/filters.hpp"
#include "vision/geometry.hpp"
#include "vision/protocol.hpp"
#include "vision/radio.hpp"
#include "vision/recorder.hpp"
#include "vision/serial.hpp"
#include "vision/ui.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using vision::Clock;
using vision::SerialPoint;

std::atomic<bool> shutdown_requested{false};

double now_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

void handle_sigint(int) {
    if (!shutdown_requested.exchange(true)) {
        std::cout << "received Ctrl+C, shutting down\n";
    }
}

struct SharedState {
    std::mutex mutex;
    std::array<int, 6> vulnerability{-1, -1, -1, -1, -1, -1};
    vision::RefereeStatus referee;
    int target = -1;
    int gimbal_countermeasure_count = 0;
    bool gimbal_countermeasure_active = false;
};

struct MotionSnapshot {
    SerialPoint position;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double updated_at = 0.0;
};

struct FilterRuntime {
    std::mutex mutex;
    std::unique_ptr<vision::PositionFilter> main_filter;
    std::unique_ptr<vision::PositionFilter> compare_filter;

    std::mutex motion_mutex;
    std::unordered_map<std::string, std::deque<std::pair<double, SerialPoint>>> motion_history;
    std::unordered_map<std::string, MotionSnapshot> motion_snapshot;
    std::unordered_map<std::string, bool> guess_active;
    std::unordered_map<std::string, int> guess_progress;
};

struct RadioCache {
    std::mutex mutex;
    std::unordered_map<std::string, SerialPoint> enemy_points;
    double updated_at = 0.0;
};

std::string robot_name(const std::string& prefix, int role) {
    return prefix + std::to_string(role);
}

std::vector<std::string> all_robot_names() {
    std::vector<std::string> names;
    for (char prefix : {'R', 'B'}) {
        for (int role : {1, 2, 3, 4, 5, 6, 7}) {
            names.push_back(robot_name(std::string(1, prefix), role));
        }
    }
    return names;
}

double point_distance_meters(const SerialPoint& lhs, const SerialPoint& rhs) {
    const double dx = (static_cast<double>(lhs.x) - rhs.x) / 100.0;
    const double dy = (static_cast<double>(lhs.y) - rhs.y) / 100.0;
    return std::hypot(dx, dy);
}

std::string prefix_for_state(char state) {
    return state == 'R' ? "R" : "B";
}

std::string opponent_prefix(char state) {
    return state == 'R' ? "B" : "R";
}

SerialPoint to_serial(const cv::Point2d& map_point, char state) {
    if (state == 'R') {
        return SerialPoint{
            static_cast<int>(2800.0 - map_point.y),
            static_cast<int>(1500.0 - map_point.x),
        };
    }
    return SerialPoint{
        static_cast<int>(map_point.y),
        static_cast<int>(map_point.x),
    };
}

std::pair<cv::Point2d, SerialPoint> project_to_map_and_serial(
    const cv::Point2d& map_point,
    char state
) {
    cv::Point2d projected;
    if (state == 'R') {
        projected = cv::Point2d(2800.0 - map_point.y, map_point.x);
    } else {
        projected = cv::Point2d(map_point.y, 1500.0 - map_point.x);
    }
    return {
        projected,
        SerialPoint{
            static_cast<int>(projected.x),
            static_cast<int>(1500.0 - projected.y),
        },
    };
}

void record_guess_motion(
    FilterRuntime& runtime,
    const vision::AppConfig& config,
    char state,
    const std::string& name,
    double x,
    double y,
    double timestamp
) {
    const SerialPoint position = vision::filter_xy_to_serial(x, y, state);
    std::lock_guard<std::mutex> lock(runtime.motion_mutex);
    auto& history = runtime.motion_history[name];
    if (!history.empty() &&
        timestamp - history.back().first > config.blind_zone.motion_reset_gap_s) {
        history.clear();
    }
    history.emplace_back(timestamp, position);
    while (history.size() > config.blind_zone.motion_history_size) {
        history.pop_front();
    }

    MotionSnapshot snapshot;
    snapshot.position = position;
    snapshot.updated_at = timestamp;
    if (history.size() >= 2) {
        const auto& first = history.front();
        const auto& last = history.back();
        const double dt = last.first - first.first;
        if (dt > 1e-3) {
            snapshot.velocity_x = (last.second.x - first.second.x) / dt;
            snapshot.velocity_y = (last.second.y - first.second.y) / dt;
        }
    }
    runtime.motion_snapshot[name] = snapshot;
}

std::vector<SerialPoint> rank_guess_points(
    FilterRuntime& runtime,
    const vision::AppConfig& config,
    char state,
    const std::string& name
) {
    if (name.rfind(opponent_prefix(state), 0) != 0) {
        return {};
    }
    const auto table_iterator = config.blind_zone.points.find(name);
    std::vector<SerialPoint> points =
        table_iterator == config.blind_zone.points.end()
            ? std::vector<SerialPoint>{}
            : table_iterator->second;

    std::optional<MotionSnapshot> snapshot;
    {
        std::lock_guard<std::mutex> lock(runtime.motion_mutex);
        const auto iterator = runtime.motion_snapshot.find(name);
        if (iterator != runtime.motion_snapshot.end()) {
            snapshot = iterator->second;
        }
    }
    if (!snapshot.has_value()) {
        return points;
    }

    const SerialPoint lost_offset_point{
        std::clamp(
            static_cast<int>(std::round(snapshot->position.x + config.blind_zone.lost_x_offset_cm)),
            0,
            config.map_size.width
        ),
        std::clamp(snapshot->position.y, 0, config.map_size.height),
    };
    if (points.empty()) {
        return {lost_offset_point};
    }

    const double speed = std::hypot(snapshot->velocity_x, snapshot->velocity_y);
    const bool use_direction =
        speed >= config.blind_zone.min_speed_mps * 100.0;
    struct ScoredPoint {
        double score = 0.0;
        std::size_t index = 0;
        SerialPoint point;
    };
    std::vector<ScoredPoint> scored;
    scored.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const double dx = points[index].x - snapshot->position.x;
        const double dy = points[index].y - snapshot->position.y;
        const double distance = std::hypot(dx, dy);
        const double distance_score =
            std::exp(-(distance / 100.0) * config.blind_zone.distance_factor);
        double direction_score = 0.0;
        if (use_direction && distance > 1e-6) {
            direction_score =
                (snapshot->velocity_x * dx + snapshot->velocity_y * dy) /
                (speed * distance);
        }
        const double score =
            config.blind_zone.direction_weight * direction_score +
            (1.0 - config.blind_zone.direction_weight) * distance_score;
        scored.push_back(ScoredPoint{score, index, points[index]});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const ScoredPoint& lhs, const ScoredPoint& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        return lhs.index < rhs.index;
    });

    std::vector<SerialPoint> ordered{lost_offset_point};
    for (const ScoredPoint& item : scored) {
        if (item.point != lost_offset_point) {
            ordered.push_back(item.point);
        }
    }
    return ordered;
}

std::map<std::string, SerialPoint> send_map_snapshot(
    FilterRuntime& runtime,
    const vision::AppConfig& config,
    char state,
    RadioCache& radio,
    double guess_time_limit
) {
    std::map<std::string, cv::Point2d> filtered;
    {
        std::lock_guard<std::mutex> lock(runtime.mutex);
        filtered = runtime.main_filter->get_all_data();
    }

    {
        std::lock_guard<std::mutex> lock(runtime.motion_mutex);
        for (const std::string& name : all_robot_names()) {
            runtime.guess_active[name] = filtered.find(name) == filtered.end();
        }
    }

    static std::map<std::string, std::pair<std::vector<SerialPoint>, std::size_t>> guess_runtime;
    static std::map<std::string, double> last_switch;
    static std::map<std::string, int> last_progress;
    static std::map<std::string, double> snapshot_updated;

    std::map<std::string, SerialPoint> send_map;
    for (const std::string& name : all_robot_names()) {
        send_map[name] = SerialPoint{};
    }

    const std::string ally = prefix_for_state(state);
    const std::string enemy = opponent_prefix(state);
    for (const std::string& role : {"1", "2", "3", "4", "6", "7"}) {
        for (const std::string& prefix : {enemy, ally}) {
            const std::string name = prefix + role;
            const auto filtered_iterator = filtered.find(name);
            if (filtered_iterator != filtered.end()) {
                send_map[name] = to_serial(filtered_iterator->second, state);
                continue;
            }

            bool active_guess = true;
            {
                std::lock_guard<std::mutex> lock(runtime.motion_mutex);
                active_guess = runtime.guess_active[name];
            }
            if (!config.blind_zone.enabled || !active_guess ||
                prefix != enemy || config.blind_zone.points.find(name) == config.blind_zone.points.end()) {
                continue;
            }

            auto points = rank_guess_points(runtime, config, state, name);
            if (points.empty()) {
                continue;
            }
            double updated_at = 0.0;
            int progress = 0;
            {
                std::lock_guard<std::mutex> lock(runtime.motion_mutex);
                const auto snapshot = runtime.motion_snapshot.find(name);
                if (snapshot != runtime.motion_snapshot.end()) {
                    updated_at = snapshot->second.updated_at;
                }
                progress = runtime.guess_progress[name];
            }

            auto& state_for_name = guess_runtime[name];
            if (state_for_name.first.empty() || snapshot_updated[name] != updated_at) {
                state_for_name = {std::move(points), 0};
                snapshot_updated[name] = updated_at;
                last_switch[name] = now_seconds();
                last_progress[name] = progress;
            }
            if (progress > last_progress[name]) {
                last_progress[name] = progress;
                last_switch[name] = now_seconds();
            } else if (progress < 120 &&
                       now_seconds() - last_switch[name] >= guess_time_limit) {
                state_for_name.second =
                    (state_for_name.second + 1) % state_for_name.first.size();
                last_switch[name] = now_seconds();
                last_progress[name] = progress;
            }
            send_map[name] = state_for_name.first[state_for_name.second];
        }
    }

    {
        std::lock_guard<std::mutex> lock(radio.mutex);
        if (now_seconds() - radio.updated_at <= config.info_wave_timeout_s) {
            for (const auto& item : radio.enemy_points) {
                send_map[item.first] = item.second;
            }
        }
    }
    return send_map;
}

bool distance_trigger(
    const std::map<std::string, SerialPoint>& send_map,
    const vision::AppConfig& config,
    FilterRuntime& runtime,
    char state
) {
    const std::string ally_prefix = prefix_for_state(state);
    const std::string enemy_prefix = opponent_prefix(state);
    for (const std::string& role : {"1", "2", "3", "4", "5", "7"}) {
        const std::string ally = ally_prefix + role;
        bool ally_is_guess = true;
        {
            std::lock_guard<std::mutex> lock(runtime.motion_mutex);
            ally_is_guess = runtime.guess_active[ally];
        }
        if (send_map.at(ally) == SerialPoint{} || ally_is_guess) {
            continue;
        }
        for (const std::string& enemy_role : {"1", "2", "3", "4", "5", "6", "7"}) {
            const std::string enemy = enemy_prefix + enemy_role;
            bool enemy_is_guess = true;
            {
                std::lock_guard<std::mutex> lock(runtime.motion_mutex);
                enemy_is_guess = runtime.guess_active[enemy];
            }
            if (send_map.at(enemy) == SerialPoint{} || enemy_is_guess) {
                continue;
            }
            if (point_distance_meters(send_map.at(ally), send_map.at(enemy)) <
                config.double_vulnerability.distance_threshold) {
                return true;
            }
        }
    }
    return false;
}

bool approaching(
    const SerialPoint& our_position,
    const SerialPoint& enemy_position,
    const std::pair<double, double>& our_velocity,
    const std::pair<double, double>& enemy_velocity,
    const vision::AppConfig& config
) {
    const double dx = (enemy_position.x - static_cast<double>(our_position.x)) / 100.0;
    const double dy = (enemy_position.y - static_cast<double>(our_position.y)) / 100.0;
    const double distance = std::hypot(dx, dy);
    if (distance < config.double_vulnerability.close_distance) {
        return true;
    }
    if (distance > config.double_vulnerability.far_distance || distance < 0.01) {
        return false;
    }

    const double direction_x = dx / distance;
    const double direction_y = dy / distance;
    const double relative_x = enemy_velocity.first - our_velocity.first;
    const double relative_y = enemy_velocity.second - our_velocity.second;
    const double relative_speed = std::hypot(relative_x, relative_y);
    if (relative_speed >= 0.1) {
        const double approach_velocity =
            -(relative_x * direction_x + relative_y * direction_y);
        if (approach_velocity > config.double_vulnerability.approach_speed) {
            return true;
        }
    }
    const double our_speed = std::hypot(our_velocity.first, our_velocity.second);
    if (our_speed > config.double_vulnerability.min_speed) {
        const double our_approach =
            our_velocity.first * direction_x + our_velocity.second * direction_y;
        if (our_approach > config.double_vulnerability.active_speed) {
            return true;
        }
    }
    return false;
}

std::optional<std::pair<double, double>> velocity_from_history(
    const std::deque<std::pair<double, SerialPoint>>& history
) {
    if (history.size() < 2) {
        return std::nullopt;
    }
    const auto& first = history[history.size() - 2];
    const auto& last = history.back();
    const double dt = last.first - first.first;
    if (dt < 0.01) {
        return std::nullopt;
    }
    return std::make_pair(
        (last.second.x - first.second.x) / 100.0 / dt,
        (last.second.y - first.second.y) / 100.0 / dt
    );
}

class RefereeSendWorker {
public:
    RefereeSendWorker(
        const vision::AppConfig& config,
        vision::SerialPort& serial,
        FilterRuntime& runtime,
        RadioCache& radio,
        SharedState& shared,
        std::atomic<bool>& shutdown
    )
        : config_(config),
          serial_(serial),
          runtime_(runtime),
          radio_(radio),
          shared_(shared),
          shutdown_(shutdown) {}

    void start() {
        if (thread_.joinable()) {
            return;
        }
        thread_ = std::thread([this]() { run(); });
    }

    void stop() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        if (!serial_.is_open()) {
            std::cout << "serial send disabled\n";
            return;
        }

        int sequence = 0;
        int chances_flag = 0;
        int last_target = 0;
        double last_trigger_time = 0.0;
        std::optional<double> distance_true_since;
        std::optional<double> motion_true_since;
        std::optional<double> motion_condition_since;
        std::vector<int> current_password = vision::protocol::generate_random_password();
        bool password_sent = false;
        int last_key_modifiable = -1;
        int double_vulnerability_startup_uses = 0;
        std::map<std::string, std::deque<std::pair<double, SerialPoint>>> position_history;
        for (const std::string& name : all_robot_names()) {
            position_history[name] = {};
        }
        const double program_start = now_seconds();

        while (!shutdown_.load()) {
            const auto radar_status = [&]() {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                return shared_.referee;
            }();
            const int target = [&]() {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                return shared_.target;
            }();

            const double guess_time_limit =
                config_.blind_zone.base_time + config_.blind_zone.offset_time;
            auto send_map = send_map_snapshot(
                runtime_,
                config_,
                config_.state,
                radio_,
                guess_time_limit
            );

            const auto radar_data = vision::protocol::build_data_radar_all(send_map, config_.state);
            const auto radar_packet = vision::protocol::build_send_packet(
                radar_data,
                sequence,
                vision::protocol::kRadarAllDataCmd
            );
            serial_.write(radar_packet.bytes);
            sequence = radar_packet.next_sequence;

            bool send_decision = false;
            int decision_command = chances_flag;
            int password_command = 0;
            std::array<int, 6> password{};

            if (!password_sent && (radar_status.key_modifiable == 1 ||
                                   radar_status.key_modifiable == -1)) {
                password_command = 1;
                for (std::size_t i = 0; i < password.size(); ++i) {
                    password[i] = current_password[i];
                }
                send_decision = true;
                password_sent = true;
                last_key_modifiable = radar_status.key_modifiable;
            } else if (password_sent && radar_status.key_modifiable == 1 &&
                       last_key_modifiable != 1) {
                current_password = vision::protocol::generate_random_password();
                password_command = 1;
                for (std::size_t i = 0; i < password.size(); ++i) {
                    password[i] = current_password[i];
                }
                send_decision = true;
            }

            const bool startup_gate_ok =
                config_.double_vulnerability.startup_limit_window_s <= 0.0 ||
                now_seconds() - program_start >=
                    config_.double_vulnerability.startup_limit_window_s ||
                double_vulnerability_startup_uses <
                    config_.double_vulnerability.startup_limit_max_uses;

            auto request_double_vulnerability = [&]() {
                if (radar_status.double_vulnerability_chance > 0 &&
                    radar_status.opponent_double_vulnerability == 0 &&
                    now_seconds() - last_trigger_time > 10.0 &&
                    startup_gate_ok) {
                    ++chances_flag;
                    if (chances_flag > 2) {
                        chances_flag = 0;
                    }
                    decision_command = chances_flag;
                    send_decision = true;
                    last_trigger_time = now_seconds();
                    if (config_.double_vulnerability.startup_limit_window_s > 0.0 &&
                        now_seconds() - program_start <
                            config_.double_vulnerability.startup_limit_window_s) {
                        ++double_vulnerability_startup_uses;
                    }
                    return true;
                }
                return false;
            };

            if (target != last_target && target != 0) {
                last_target = target;
                request_double_vulnerability();
            }

            const std::string mode = config_.double_vulnerability.trigger_mode;
            bool trigger_now = false;
            if (mode == "distance" || mode == "both") {
                if (distance_trigger(send_map, config_, runtime_, config_.state)) {
                    if (!distance_true_since.has_value()) {
                        distance_true_since = now_seconds();
                    } else if (now_seconds() - *distance_true_since >=
                               config_.double_vulnerability.stable_duration_s) {
                        trigger_now = true;
                        distance_true_since.reset();
                    }
                } else {
                    distance_true_since.reset();
                }
            }

            if (mode == "motion" || mode == "both") {
                for (const auto& item : send_map) {
                    bool is_guess = true;
                    {
                        std::lock_guard<std::mutex> lock(runtime_.motion_mutex);
                        is_guess = runtime_.guess_active[item.first];
                    }
                    if (item.second != SerialPoint{} && !is_guess) {
                        position_history[item.first].emplace_back(now_seconds(), item.second);
                        while (position_history[item.first].size() > 5) {
                            position_history[item.first].pop_front();
                        }
                    }
                }
                bool motion_trigger = false;
                const std::string ally_prefix = prefix_for_state(config_.state);
                const std::string enemy_prefix = opponent_prefix(config_.state);
                for (const std::string& role : {"1", "2", "3", "4", "5", "7"}) {
                    const std::string ally = ally_prefix + role;
                    const auto our_velocity = velocity_from_history(position_history[ally]);
                    if (!our_velocity.has_value()) {
                        continue;
                    }
                    for (const std::string& enemy_role : {"1", "2", "3", "4", "5", "6", "7"}) {
                        const std::string enemy = enemy_prefix + enemy_role;
                        const auto enemy_velocity = velocity_from_history(position_history[enemy]);
                        if (!enemy_velocity.has_value()) {
                            continue;
                        }
                        if (approaching(
                                position_history[ally].back().second,
                                position_history[enemy].back().second,
                                *our_velocity,
                                *enemy_velocity,
                                config_
                            )) {
                            motion_trigger = true;
                            break;
                        }
                    }
                    if (motion_trigger) {
                        break;
                    }
                }
                if (motion_trigger) {
                    if (!motion_condition_since.has_value()) {
                        motion_condition_since = now_seconds();
                    } else if (now_seconds() - *motion_condition_since >=
                               config_.double_vulnerability.stable_duration_s) {
                        trigger_now = true;
                        motion_condition_since.reset();
                    }
                } else {
                    motion_condition_since.reset();
                }
            }
            if (trigger_now) {
                request_double_vulnerability();
            }

            if (send_decision) {
                const auto decision = vision::protocol::build_data_decision(
                    decision_command,
                    config_.state,
                    password_command,
                    &password
                );
                const auto packet = vision::protocol::build_send_packet(
                    decision,
                    sequence,
                    vision::protocol::kRadarDecisionCmd
                );
                serial_.write(packet.bytes);
                sequence = packet.next_sequence;
            }
            last_key_modifiable = radar_status.key_modifiable;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    const vision::AppConfig& config_;
    vision::SerialPort& serial_;
    FilterRuntime& runtime_;
    RadioCache& radio_;
    SharedState& shared_;
    std::atomic<bool>& shutdown_;
    std::thread thread_;
};

class RefereeReceiveWorker {
public:
    RefereeReceiveWorker(
        const vision::AppConfig& config,
        vision::SerialPort& serial,
        SharedState& shared,
        std::atomic<bool>& shutdown
    )
        : config_(config), serial_(serial), shared_(shared), shutdown_(shutdown) {}

    void start() {
        if (thread_.joinable()) {
            return;
        }
        thread_ = std::thread([this]() { run(); });
    }

    void stop() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        if (!serial_.is_open()) {
            std::cout << "serial receive disabled\n";
            return;
        }
        vision::protocol::RefereeFrameBuffer buffer;
        double last_stats = now_seconds();
        while (!shutdown_.load()) {
            const auto bytes = serial_.read_all();
            const auto frames = buffer.consume(bytes);
            for (const vision::protocol::ParsedFrame& frame : frames) {
                if (frame.command_id == vision::protocol::kMarkProgressCmd &&
                    frame.data.size() >= 2) {
                    const std::uint16_t mark_progress =
                        static_cast<std::uint16_t>(frame.data[0]) |
                        static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(frame.data[1]) << 8U
                        );
                    const bool countermeasure_active =
                        ((mark_progress >> 13U) & 0x01U) != 0U;
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    if (countermeasure_active && !shared_.gimbal_countermeasure_active) {
                        shared_.gimbal_countermeasure_count =
                            std::min(0xFF, shared_.gimbal_countermeasure_count + 1);
                    }
                    shared_.gimbal_countermeasure_active = countermeasure_active;
                    for (int bit = 0; bit < 6; ++bit) {
                        shared_.vulnerability[static_cast<std::size_t>(bit)] =
                            ((mark_progress >> bit) & 0x01U) != 0U ? 120 : 0;
                    }
                } else if (frame.command_id == vision::protocol::kVulnerabilityCmd &&
                           !frame.data.empty()) {
                    const vision::RefereeStatus status =
                        vision::protocol::parse_radar_decision(frame.data[0]);
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    shared_.referee = status;
                } else if (frame.command_id == vision::protocol::kDartTargetCmd &&
                           frame.data.size() >= 2) {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    shared_.target = (frame.data[1] & 0b01100000U) >> 5U;
                } else if (frame.command_id == vision::protocol::kGameStatusCmd) {
                    const auto status = vision::protocol::parse_game_status(frame.data);
                    if (status.has_value()) {
                        std::cout
                            << "[referee 0x0001] type=" << status->game_type
                            << " progress=" << status->game_progress
                            << " remain=" << status->stage_remain_time
                            << " timestamp=" << status->sync_timestamp
                            << '\n';
                    }
                }
            }

            if (now_seconds() - last_stats >= 1.0) {
                const auto stats = buffer.take_stats();
                std::cout
                    << "[referee serial] bytes=" << stats.bytes
                    << " complete=" << stats.complete_frames
                    << " crc16_ok=" << stats.crc16_ok
                    << " crc16_errors=" << stats.crc16_errors
                    << " crc8_errors=" << stats.header_crc8_errors
                    << '\n';
                last_stats = now_seconds();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    const vision::AppConfig& config_;
    vision::SerialPort& serial_;
    SharedState& shared_;
    std::atomic<bool>& shutdown_;
    std::thread thread_;
};

void send_gimbal_countermeasure(
    vision::SerialPort& gimbal,
    const vision::AppConfig& config,
    SharedState& shared,
    double& last_send
) {
    if (!gimbal.is_open()) {
        return;
    }
    const double now = now_seconds();
    if (now - last_send < config.gimbal_min_interval_s) {
        return;
    }
    int count = 0;
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        count = shared.gimbal_countermeasure_count & 0xFF;
    }
    std::vector<std::uint8_t> packet = {
        0xA5,
        0xA5,
        static_cast<std::uint8_t>(count),
    };
    const std::uint16_t crc = vision::protocol::crc16(packet);
    packet.push_back(static_cast<std::uint8_t>(crc & 0xFF));
    packet.push_back(static_cast<std::uint8_t>((crc >> 8U) & 0xFF));
    packet.push_back(0xFE);
    gimbal.write(packet);
    last_send = now;
}

struct PackedRoi {
    int index = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct Canvas {
    cv::Mat image;
    std::vector<PackedRoi> rois;
};

std::vector<Canvas> pack_rois(
    const std::vector<cv::Mat>& rois,
    const cv::Size& canvas_size = cv::Size(640, 640),
    int padding = 10
) {
    std::vector<Canvas> canvases;
    Canvas current;
    current.image = cv::Mat::zeros(canvas_size, CV_8UC3);
    int x = 0;
    int y = 0;
    int row_height = 0;

    auto finish_canvas = [&]() {
        if (!current.rois.empty()) {
            canvases.push_back(std::move(current));
            current = Canvas{};
            current.image = cv::Mat::zeros(canvas_size, CV_8UC3);
        }
    };

    for (std::size_t index = 0; index < rois.size(); ++index) {
        const cv::Mat& roi = rois[index];
        if (roi.empty()) {
            continue;
        }
        const int padded_width = roi.cols + 2 * padding;
        const int padded_height = roi.rows + 2 * padding;
        if (padded_width > canvas_size.width || padded_height > canvas_size.height) {
            continue;
        }
        if (x + padded_width > canvas_size.width) {
            x = 0;
            y += row_height;
            row_height = 0;
        }
        if (y + padded_height > canvas_size.height) {
            finish_canvas();
            x = 0;
            y = 0;
            row_height = 0;
        }
        roi.copyTo(current.image(cv::Rect(
            x + padding,
            y + padding,
            roi.cols,
            roi.rows
        )));
        current.rois.push_back(PackedRoi{
            static_cast<int>(index),
            x + padding,
            y + padding,
            roi.cols,
            roi.rows,
        });
        x += padded_width;
        row_height = std::max(row_height, padded_height);
    }
    finish_canvas();
    return canvases;
}

int max_game_index() {
    int best = 0;
    if (!fs::exists("save_img")) {
        return best;
    }
    for (const auto& entry : fs::directory_iterator("save_img")) {
        const std::string name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind("game", 0) == 0) {
            try {
                best = std::max(best, std::stoi(name.substr(4)));
            } catch (const std::exception&) {
            }
        }
    }
    return best;
}

std::string timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string make_feedback_json(const vision::AppConfig& config, const SharedState& shared) {
    int level = 1;
    {
        std::lock_guard<std::mutex> lock(const_cast<SharedState&>(shared).mutex);
        level = shared.referee.encryption_level;
    }
    std::ostringstream stream;
    stream << "{\"team\":\"" << (config.state == 'R' ? "red" : "blue")
           << "\",\"level\":" << level
           << ",\"game_progress\":3,\"stage_remain_time\":0,\"radar_id\":"
           << (config.state == 'R' ? 9 : 109) << '}';
    return stream.str();
}

}  // namespace

int main() {
    try {
        const fs::path config_path =
            fs::exists(fs::current_path() / "config.yaml")
                ? fs::current_path() / "config.yaml"
                : vision::default_project_root() / "config.yaml";
        vision::AppConfig config = vision::ConfigLoader::load(config_path);
        std::signal(SIGINT, handle_sigint);

        std::cout
            << "Vision C++ runtime\n"
            << "config=" << config_path << '\n'
            << "camera=" << config.camera_mode << " state=" << config.state << '\n';

        vision::SerialPort referee_serial;
        if (config.use_serial) {
            if (!referee_serial.open(config.serial_port, config.serial_baudrate, config.serial_timeout)) {
                std::cerr << "referee serial open failed; serial disabled\n";
                config.use_serial = false;
            } else {
                referee_serial.reset_input_buffer();
                std::cout << "referee serial connected: " << config.serial_port << '\n';
            }
        }

        vision::SerialPort gimbal_serial;
        if (config.gimbal_enabled) {
            if (!gimbal_serial.open(config.gimbal_port, config.gimbal_baudrate, config.gimbal_timeout)) {
                std::cerr << "gimbal serial open failed; disabled\n";
                config.gimbal_enabled = false;
            }
        }

        FilterRuntime filter_runtime;
        auto motion_callback = [&](const std::string& name, double x, double y, double time) {
            record_guess_motion(filter_runtime, config, config.state, name, x, y, time);
        };
        filter_runtime.main_filter =
            vision::make_filter(config, config.filter_type, motion_callback);
        if (config.filter_compare.enabled &&
            config.filter_compare.compare_type != config.filter_type) {
            std::string compare_type = config.filter_compare.compare_type;
            if (compare_type == "auto") {
                compare_type =
                    config.filter_type == "sliding_window"
                        ? "hybrid_gated_kalman"
                        : "sliding_window";
            }
            if (compare_type != config.filter_type) {
                filter_runtime.compare_filter =
                    vision::make_filter(config, compare_type, {});
            }
        }
        for (const std::string& name : all_robot_names()) {
            filter_runtime.guess_active[name] = true;
            filter_runtime.guess_progress[name] = 0;
        }

        cv::Mat map = cv::imread(config.map_path);
        cv::Mat mask = cv::imread(config.mask_path);
        cv::Mat map_backup = cv::imread(config.map_backup_path);
        if (map.empty() || mask.empty() || map_backup.empty()) {
            throw std::runtime_error(
                "failed to load map image files; check paths in config.yaml"
            );
        }
        const auto calibration = vision::load_calibration_npy(config.calibration_path);
        if (calibration.size() < 2) {
            throw std::runtime_error("calibration file must contain two homography matrices");
        }
        const cv::Mat ground_transform = calibration[0];
        const cv::Mat height_transform = calibration[1];

        vision::CameraSource camera(config);
        camera.start();
        if (!camera.wait_for_first_frame()) {
            throw std::runtime_error("timed out waiting for the first camera frame");
        }
        const cv::Mat first_frame = camera.latest_frame();
        vision::FfmpegRecorder recorder(config, first_frame.size());
        recorder.start();
        camera.set_frame_callback([&](const cv::Mat& frame) { recorder.submit(frame); });

        vision::Detector car_detector(
            config.car_model,
            cv::Size(640, 640),
            static_cast<float>(config.car_confidence),
            0.5F,
            14,
            {"car", "armor", "ignore", "watcher", "base"}
        );
        vision::Detector armor_detector(
            config.armor_model,
            cv::Size(640, 640),
            static_cast<float>(config.armor_confidence),
            0.2F,
            10,
            {"B1", "B2", "B3", "B4", "B5", "B7",
             "R1", "R2", "R3", "R4", "R5", "R7"}
        );
        std::unique_ptr<vision::Detector> uav_detector;
        if (config.uav_enabled && !config.uav_model.empty() && fs::exists(config.uav_model)) {
            uav_detector = std::make_unique<vision::Detector>(
                config.uav_model,
                cv::Size(640, 640),
                static_cast<float>(config.uav_confidence),
                0.4F,
                config.uav_max_det,
                std::vector<std::string>{"plane"}
            );
        }

        SharedState shared;
        RadioCache radio_cache;
        vision::RadioBridge radio_bridge;
        radio_bridge.start();

        RefereeReceiveWorker receive_worker(
            config,
            referee_serial,
            shared,
            shutdown_requested
        );
        RefereeSendWorker send_worker(
            config,
            referee_serial,
            filter_runtime,
            radio_cache,
            shared,
            shutdown_requested
        );
        if (config.use_serial) {
            receive_worker.start();
            send_worker.start();
        }

        const int game_index = max_game_index() + 1;
        if (config.save_img) {
            fs::create_directories(fs::path("save_img") / ("game" + std::to_string(game_index)) / "1");
            fs::create_directories(fs::path("save_img") / ("game" + std::to_string(game_index)) / "2");
        }

        cv::Mat information_ui = cv::Mat::ones(config.info_panel_size, CV_8UC3) * 255;
        int frame_counter = 0;
        int uav_frame_counter = 0;
        double gimbal_last_send = 0.0;
        double last_feedback_time = 0.0;
        std::vector<std::map<std::string, std::string>> uav_draws;

        while (!shutdown_requested.load()) {
            const auto started = Clock::now();
            cv::Mat image = camera.latest_frame();
            if (image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            cv::Mat display = image.clone();

            std::vector<vision::Detection> car_detections =
                car_detector.predict(image);

            if (uav_detector) {
                ++uav_frame_counter;
                if ((uav_frame_counter - 1) % config.uav_detect_interval_frames == 0) {
                    uav_draws.clear();
                    for (const vision::Detection& detection : uav_detector->predict(image)) {
                        std::string class_name = detection.cls;
                        std::transform(
                            class_name.begin(),
                            class_name.end(),
                            class_name.begin(),
                            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
                        );
                        if (std::find(
                                config.uav_target_classes.begin(),
                                config.uav_target_classes.end(),
                                class_name
                            ) == config.uav_target_classes.end()) {
                            continue;
                        }
                        const cv::Point2d camera_point(
                            std::min(
                                static_cast<double>(
                                    detection.box.x + 0.5 * detection.box.width
                                ),
                                static_cast<double>(image.cols)
                            ),
                            std::min(
                                static_cast<double>(
                                    detection.box.y + 0.5 * detection.box.height
                                ),
                                static_cast<double>(image.rows)
                            )
                        );
                        const cv::Point2d map_point = vision::project_camera_point(
                            camera_point,
                            ground_transform,
                            height_transform,
                            mask
                        );
                        const SerialPoint serial = vision::filter_xy_to_serial(
                            map_point.x,
                            map_point.y,
                            config.state
                        );
                        std::map<std::string, std::string> draw;
                        draw["box"] =
                            cv::format(
                                "%d,%d,%d,%d",
                                static_cast<int>(detection.box.x),
                                static_cast<int>(detection.box.y),
                                static_cast<int>(detection.box.x + detection.box.width),
                                static_cast<int>(detection.box.y + detection.box.height)
                            );
                        draw["label"] = cv::format(
                            "%s%d %.2f (%d,%d)",
                            opponent_prefix(config.state).c_str(),
                            6,
                            detection.confidence,
                            serial.x,
                            serial.y
                        );
                        if (serial.y < config.uav_max_y) {
                            const std::string uav_name = opponent_prefix(config.state) + "6";
                            {
                                std::lock_guard<std::mutex> lock(filter_runtime.mutex);
                                filter_runtime.main_filter->add_data(
                                    uav_name,
                                    map_point.x,
                                    map_point.y
                                );
                                if (filter_runtime.compare_filter) {
                                    filter_runtime.compare_filter->add_data(
                                        uav_name,
                                        map_point.x,
                                        map_point.y
                                    );
                                }
                            }
                            {
                                std::lock_guard<std::mutex> lock(filter_runtime.motion_mutex);
                                filter_runtime.guess_active[uav_name] = false;
                            }
                            draw["color"] = "yellow";
                        } else {
                            draw["label"] = cv::format("uav-outside %.2f", detection.confidence);
                            draw["color"] = "gray";
                        }
                        uav_draws.push_back(std::move(draw));
                    }
                }
                for (const auto& draw : uav_draws) {
                    std::stringstream coordinates(draw.at("box"));
                    int x0 = 0;
                    int y0 = 0;
                    int x1 = 0;
                    int y1 = 0;
                    char comma = ',';
                    coordinates >> x0 >> comma >> y0 >> comma >> x1 >> comma >> y1;
                    const cv::Scalar color =
                        draw.at("color") == "yellow"
                            ? cv::Scalar(0, 255, 255)
                            : cv::Scalar(120, 120, 120);
                    cv::rectangle(display, cv::Point(x0, y0), cv::Point(x1, y1), color, 2);
                    cv::putText(
                        display,
                        draw.at("label"),
                        cv::Point(x0, std::max(y0 - 5, 20)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    );
                }
            }

            std::vector<vision::Detection> car_entries;
            std::vector<cv::Mat> rois;
            for (const vision::Detection& detection : car_detections) {
                if (detection.cls != "car") {
                    continue;
                }
                const cv::Rect2f bounded = detection.box &
                    cv::Rect2f(0, 0, image.cols, image.rows);
                const cv::Rect roi = bounded;
                if (roi.width <= 0 || roi.height <= 0) {
                    continue;
                }
                car_entries.push_back(detection);
                rois.push_back(image(roi).clone());
            }
            if (!config.multi_car_recognition && car_entries.size() > 1) {
                std::stable_sort(
                    car_entries.begin(),
                    car_entries.end(),
                    [](const vision::Detection& lhs, const vision::Detection& rhs) {
                        return lhs.confidence > rhs.confidence;
                    }
                );
                car_entries.resize(1);
                rois.resize(1);
            }

            const auto canvases = pack_rois(rois);
            for (const Canvas& canvas : canvases) {
                const auto armor_detections = armor_detector.predict(canvas.image);
                cv::Mat canvas_debug = canvas.image.clone();
                for (const PackedRoi& roi : canvas.rois) {
                    cv::rectangle(
                        canvas_debug,
                        cv::Rect(roi.x, roi.y, roi.width, roi.height),
                        cv::Scalar(0, 0, 255),
                        2
                    );
                }
                for (const PackedRoi& roi : canvas.rois) {
                    std::optional<vision::Detection> best;
                    for (const vision::Detection& detection : armor_detections) {
                        const float center_x = detection.box.x + 0.5F * detection.box.width;
                        const float center_y = detection.box.y + 0.5F * detection.box.height;
                        if (center_x >= roi.x && center_x < roi.x + roi.width &&
                            center_y >= roi.y && center_y < roi.y + roi.height) {
                            if (!best.has_value() ||
                                detection.confidence > best->confidence) {
                                best = detection;
                            }
                        }
                    }
                    if (!best.has_value() || static_cast<std::size_t>(roi.index) >= rois.size()) {
                        continue;
                    }
                    const cv::Rect car_roi =
                        car_entries[roi.index].box &
                        cv::Rect2f(0, 0, image.cols, image.rows);
                    const float x0 =
                        best->box.x - roi.x + static_cast<float>(car_roi.x);
                    const float y0 =
                        best->box.y - roi.y + static_cast<float>(car_roi.y);
                    const cv::Rect2f mapped_box(
                        x0,
                        y0,
                        best->box.width,
                        best->box.height
                    );
                    cv::rectangle(display, mapped_box, cv::Scalar(255, 0, 255), 2);
                    cv::putText(
                        display,
                        best->cls + " " + cv::format("%.2f", best->confidence),
                        cv::Point(static_cast<int>(mapped_box.x), std::max(static_cast<int>(mapped_box.y) - 5, 20)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 0, 255),
                        2
                    );

                    const cv::Point2d camera_point(
                        std::min(
                            x0 + 0.5F * mapped_box.width,
                            static_cast<float>(image.cols)
                        ),
                        std::min(
                            y0 + 1.5F * mapped_box.height,
                            static_cast<float>(image.rows)
                        )
                    );
                    const cv::Point2d map_point = vision::project_camera_point(
                        camera_point,
                        ground_transform,
                        height_transform,
                        mask
                    );
                    {
                        std::lock_guard<std::mutex> lock(filter_runtime.mutex);
                        filter_runtime.main_filter->add_data(
                            best->cls,
                            map_point.x,
                            map_point.y
                        );
                        if (filter_runtime.compare_filter) {
                            filter_runtime.compare_filter->add_data(
                                best->cls,
                                map_point.x,
                                map_point.y
                            );
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(filter_runtime.motion_mutex);
                        filter_runtime.guess_active[best->cls] = false;
                    }
                }
                if (!canvas_debug.empty()) {
                    cv::imshow("canvas_result", canvas_debug);
                }
            }

            send_gimbal_countermeasure(
                gimbal_serial,
                config,
                shared,
                gimbal_last_send
            );

            if (const auto radio_frame = radio_bridge.positions().pop_latest();
                radio_frame.has_value()) {
                std::lock_guard<std::mutex> lock(radio_cache.mutex);
                const std::string enemy = opponent_prefix(config.state);
                std::array<int, 6> ids{1, 2, 3, 4, 6, 7};
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    radio_cache.enemy_points[enemy + std::to_string(ids[index])] =
                        radio_frame->robots[index];
                }
                radio_cache.updated_at = now_seconds();
            }

            std::map<std::string, cv::Point2d> filtered;
            std::map<std::string, cv::Point2d> compare_filtered;
            {
                std::lock_guard<std::mutex> lock(filter_runtime.mutex);
                filtered = filter_runtime.main_filter->get_all_data();
                if (filter_runtime.compare_filter) {
                    compare_filtered = filter_runtime.compare_filter->get_all_data();
                }
            }
            for (const std::string& name : all_robot_names()) {
                std::lock_guard<std::mutex> lock(filter_runtime.motion_mutex);
                filter_runtime.guess_active[name] = filtered.find(name) == filtered.end();
            }

            cv::Mat map_show = map_backup.clone();
            for (const auto& item : filtered) {
                const auto [map_point, serial] =
                    project_to_map_and_serial(item.second, config.state);
                cv::Point point(
                    std::clamp(static_cast<int>(map_point.x), 0, map.cols - 1),
                    std::clamp(static_cast<int>(map_point.y), 0, map.rows - 1)
                );
                SerialPoint display_serial = serial;
                {
                    std::lock_guard<std::mutex> lock(radio_cache.mutex);
                    const auto radio_point = radio_cache.enemy_points.find(item.first);
                    if (now_seconds() - radio_cache.updated_at <=
                            config.info_wave_timeout_s &&
                        radio_point != radio_cache.enemy_points.end() &&
                        radio_point->second != SerialPoint{}) {
                        display_serial = radio_point->second;
                        const auto map_xy = vision::serial_to_map_point(display_serial);
                        point = cv::Point(
                            std::clamp(static_cast<int>(map_xy.first), 0, map.cols - 1),
                            std::clamp(static_cast<int>(map_xy.second), 0, map.rows - 1)
                        );
                    }
                }
                const cv::Scalar color =
                    item.first[0] == 'R' ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                const int thickness =
                    item.first[0] == config.state ? 3 : cv::FILLED;
                cv::circle(map_show, point, 15, color, thickness);
                cv::putText(
                    map_show,
                    item.first,
                    cv::Point(point.x - 5, point.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    2.5,
                    cv::Scalar(255, 255, 255),
                    5
                );
                cv::putText(
                    map_show,
                    cv::format("(%d,%d)", display_serial.x, display_serial.y),
                    cv::Point(point.x - 100, point.y + 60),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.5,
                    cv::Scalar(255, 255, 255),
                    4
                );
            }

            std::array<int, 6> vulnerability{};
            int chance = -1;
            int opponent_double = -1;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                vulnerability = shared.vulnerability;
                chance = shared.referee.double_vulnerability_chance;
                opponent_double = shared.referee.opponent_double_vulnerability;
            }
            cv::Mat information = information_ui.clone();
            vision::draw_information_ui(vulnerability, config.state, information);
            cv::putText(
                information,
                "vulnerability_chances: " + std::to_string(chance),
                cv::Point(10, 350),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );
            cv::putText(
                information,
                "vulnerability_Triggering: " + std::to_string(opponent_double),
                cv::Point(10, 400),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );

            const double elapsed = std::chrono::duration<double>(
                Clock::now() - started
            ).count();
            const double fps = elapsed > 0.0 ? 1.0 / elapsed : 0.0;
            ++frame_counter;
            if (frame_counter % 10 == 0) {
                std::cout << "fps: " << fps << '\n';
            }
            cv::putText(
                display,
                cv::format("FPS: %.1f", fps),
                cv::Point(20, 45),
                cv::FONT_HERSHEY_SIMPLEX,
                1.1,
                cv::Scalar(0, 255, 0),
                3
            );

            cv::Mat map_display;
            cv::resize(map_show, map_display, config.map_display_size);
            cv::Mat image_display;
            cv::resize(display, image_display, config.image_display_size);
            cv::imshow("information_ui", information);
            cv::imshow("map_main", map_display);
            cv::imshow("img", image_display);

            if (!compare_filtered.empty()) {
                cv::Mat compare_map = map_backup.clone();
                for (const auto& item : compare_filtered) {
                    const auto [map_point, serial] =
                        project_to_map_and_serial(item.second, config.state);
                    const cv::Point point(
                        std::clamp(static_cast<int>(map_point.x), 0, map.cols - 1),
                        std::clamp(static_cast<int>(map_point.y), 0, map.rows - 1)
                    );
                    const int thickness =
                        item.first[0] == config.state ? 3 : cv::FILLED;
                    cv::circle(compare_map, point, 15, cv::Scalar(0, 255, 255), thickness);
                    cv::putText(
                        compare_map,
                        item.first,
                        cv::Point(point.x - 5, point.y + 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        2.5,
                        cv::Scalar(255, 255, 255),
                        5
                    );
                    cv::putText(
                        compare_map,
                        cv::format("(%d,%d)", serial.x, serial.y),
                        cv::Point(point.x - 100, point.y + 60),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.5,
                        cv::Scalar(255, 255, 255),
                        4
                    );
                }
                cv::Mat compare_display;
                cv::resize(compare_map, compare_display, config.map_display_size);
                cv::putText(
                    compare_display,
                    "Compare Filter: " + config.filter_compare.compare_type,
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 255),
                    2
                );
                if (config.filter_compare.draw_on_map) {
                    cv::imshow("map_compare", compare_display);
                }
            }

            if (config.save_img) {
                const fs::path base =
                    fs::path("save_img") / ("game" + std::to_string(game_index));
                const std::string stamp = timestamp();
                cv::imwrite((base / "1" / (stamp + ".jpg")).string(), map_display);
                cv::imwrite((base / "2" / (stamp + ".jpg")).string(), image_display);
            }

            if (now_seconds() - last_feedback_time >= 1.0) {
                radio_bridge.publish_feedback(make_feedback_json(config, shared));
                last_feedback_time = now_seconds();
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                shutdown_requested = true;
            }
        }

        send_worker.stop();
        receive_worker.stop();
        radio_bridge.stop();
        camera.stop();
        recorder.stop();
        cv::destroyAllWindows();
        std::cout << "program exited safely\n";
        return 0;
    } catch (const std::exception& exception) {
        shutdown_requested = true;
        std::cerr << "fatal: " << exception.what() << '\n';
        return 1;
    }
}
