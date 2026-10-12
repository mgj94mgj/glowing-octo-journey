#pragma once

#include "vision/types.hpp"

#include <opencv2/video/tracking.hpp>

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace vision {

using MotionCallback = std::function<void(const std::string&, double, double, double)>;

class MotionKalman2D {
public:
    MotionKalman2D(double process_noise = 1e-5, double measurement_noise = 1e-1);

    void update(const cv::Point2d& measurement);
    std::optional<cv::Point2d> estimate() const;
    bool has_measurement() const;

private:
    cv::KalmanFilter kf_;
    bool has_measurement_ = false;
    cv::Point2d last_measurement_;
    double last_update_time_ = 0.0;
};

class PositionFilter {
public:
    virtual ~PositionFilter() = default;
    virtual void add_data(const std::string& name, double x, double y) = 0;
    virtual std::map<std::string, cv::Point2d> get_all_data() = 0;
    virtual void set_motion_callback(MotionCallback callback) = 0;
};

class SlidingWindowFilter final : public PositionFilter {
public:
    SlidingWindowFilter(
        int window_size,
        double max_inactive_time,
        double threshold,
        MotionCallback motion_callback = {}
    );

    void add_data(const std::string& name, double x, double y) override;
    std::map<std::string, cv::Point2d> get_all_data() override;
    void set_motion_callback(MotionCallback callback) override;

private:
    int window_size_;
    double max_inactive_time_;
    double threshold_;
    MotionCallback motion_callback_;
    std::map<std::string, std::deque<cv::Point2d>> windows_;
    std::map<std::string, double> last_update_;
};

class HybridGatedKalmanFilter final : public PositionFilter {
public:
    HybridGatedKalmanFilter(
        int prefilter_window,
        std::string prefilter_mode,
        double max_inactive_time,
        double raw_jump_threshold,
        double pred_jump_threshold,
        double process_noise,
        double measurement_noise,
        MotionCallback motion_callback = {}
    );

    void add_data(const std::string& name, double x, double y) override;
    std::map<std::string, cv::Point2d> get_all_data() override;
    void set_motion_callback(MotionCallback callback) override;

private:
    struct TargetState {
        std::deque<cv::Point2d> raw_window;
        MotionKalman2D kalman;
        double last_update = 0.0;
        std::optional<cv::Point2d> last_raw;
    };

    cv::Point2d prefilter(const std::deque<cv::Point2d>& points) const;

    int prefilter_window_;
    std::string prefilter_mode_;
    double max_inactive_time_;
    double raw_jump_threshold_;
    double pred_jump_threshold_;
    double process_noise_;
    double measurement_noise_;
    MotionCallback motion_callback_;
    std::map<std::string, TargetState> targets_;
};

std::unique_ptr<PositionFilter> make_filter(const AppConfig& config, const std::string& type, MotionCallback callback);

}  // namespace vision
