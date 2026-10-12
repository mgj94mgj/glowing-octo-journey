#include "vision/filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vision {
namespace {

double now_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

double squared_distance(const cv::Point2d& lhs, const cv::Point2d& rhs) {
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

}  // namespace

MotionKalman2D::MotionKalman2D(double process_noise, double measurement_noise)
    : kf_(4, 2, 0, CV_32F), last_update_time_(now_seconds()) {
    cv::setIdentity(kf_.measurementMatrix);
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(process_noise));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(measurement_noise));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));
    kf_.statePost = cv::Mat::zeros(4, 1, CV_32F);
}

void MotionKalman2D::update(const cv::Point2d& measurement) {
    const double current_time = now_seconds();
    const double dt = std::max(current_time - last_update_time_, 1e-3);
    last_update_time_ = current_time;
    const float dtf = static_cast<float>(std::max(dt, 1e-3));
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, dtf, 0,
        0, 1, 0, dtf,
        0, 0, 1, 0,
        0, 0, 0, 1
    );

    if (has_measurement_) {
        const double dx = measurement.x - last_measurement_.x;
        const double dy = measurement.y - last_measurement_.y;
        kf_.processNoiseCov.at<float>(2, 2) =
            static_cast<float>(std::min(0.5, 0.1 + std::abs(dx) / 10.0));
        kf_.processNoiseCov.at<float>(3, 3) =
            static_cast<float>(std::min(0.5, 0.1 + std::abs(dy) / 10.0));
    }

    kf_.predict();
    cv::Mat measurement_mat = (cv::Mat_<float>(2, 1) <<
        static_cast<float>(measurement.x),
        static_cast<float>(measurement.y)
    );
    kf_.correct(measurement_mat);
    last_measurement_ = measurement;
    has_measurement_ = true;
}

std::optional<cv::Point2d> MotionKalman2D::estimate() const {
    if (!has_measurement_) {
        return std::nullopt;
    }
    return cv::Point2d(kf_.statePost.at<float>(0, 0), kf_.statePost.at<float>(1, 0));
}

bool MotionKalman2D::has_measurement() const {
    return has_measurement_;
}

SlidingWindowFilter::SlidingWindowFilter(
    int window_size,
    double max_inactive_time,
    double threshold,
    MotionCallback motion_callback
)
    : window_size_(std::max(1, window_size)),
      max_inactive_time_(max_inactive_time),
      threshold_(threshold),
      motion_callback_(std::move(motion_callback)) {}

void SlidingWindowFilter::add_data(const std::string& name, double x, double y) {
    auto& window = windows_[name];
    const cv::Point2d point(x, y);
    if (!window.empty() && squared_distance(window.back(), point) > threshold_) {
        return;
    }
    while (static_cast<int>(window.size()) >= window_size_) {
        window.pop_front();
    }
    window.push_back(point);
    const double timestamp = now_seconds();
    last_update_[name] = timestamp;
    if (motion_callback_) {
        motion_callback_(name, x, y, timestamp);
    }
}

std::map<std::string, cv::Point2d> SlidingWindowFilter::get_all_data() {
    const double timestamp = now_seconds();
    std::map<std::string, cv::Point2d> output;
    for (auto iterator = windows_.begin(); iterator != windows_.end();) {
        if (timestamp - last_update_[iterator->first] > max_inactive_time_) {
            last_update_.erase(iterator->first);
            iterator = windows_.erase(iterator);
            continue;
        }
        if (static_cast<int>(iterator->second.size()) >= window_size_) {
            cv::Point2d sum;
            for (const cv::Point2d& point : iterator->second) {
                sum += point;
            }
            output[iterator->first] = sum * (1.0 / static_cast<double>(iterator->second.size()));
        }
        ++iterator;
    }
    return output;
}

void SlidingWindowFilter::set_motion_callback(MotionCallback callback) {
    motion_callback_ = std::move(callback);
}

HybridGatedKalmanFilter::HybridGatedKalmanFilter(
    int prefilter_window,
    std::string prefilter_mode,
    double max_inactive_time,
    double raw_jump_threshold,
    double pred_jump_threshold,
    double process_noise,
    double measurement_noise,
    MotionCallback motion_callback
)
    : prefilter_window_(std::max(1, prefilter_window)),
      prefilter_mode_(std::move(prefilter_mode)),
      max_inactive_time_(max_inactive_time),
      raw_jump_threshold_(raw_jump_threshold),
      pred_jump_threshold_(pred_jump_threshold),
      process_noise_(process_noise),
      measurement_noise_(measurement_noise),
      motion_callback_(std::move(motion_callback)) {}

cv::Point2d HybridGatedKalmanFilter::prefilter(const std::deque<cv::Point2d>& points) const {
    if (prefilter_mode_ == "mean") {
        cv::Point2d sum;
        for (const cv::Point2d& point : points) {
            sum += point;
        }
        return sum * (1.0 / static_cast<double>(points.size()));
    }
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    for (const cv::Point2d& point : points) {
        xs.push_back(point.x);
        ys.push_back(point.y);
    }
    const auto middle = xs.size() / 2;
    std::nth_element(xs.begin(), xs.begin() + static_cast<std::ptrdiff_t>(middle), xs.end());
    std::nth_element(ys.begin(), ys.begin() + static_cast<std::ptrdiff_t>(middle), ys.end());
    return cv::Point2d(xs[middle], ys[middle]);
}

void HybridGatedKalmanFilter::add_data(const std::string& name, double x, double y) {
    TargetState& state = targets_.try_emplace(
        name,
        TargetState{
            {},
            MotionKalman2D(process_noise_, measurement_noise_),
            0.0,
            std::nullopt,
        }
    ).first->second;

    const cv::Point2d raw(x, y);
    if (state.last_raw.has_value() &&
        squared_distance(raw, *state.last_raw) > raw_jump_threshold_ * raw_jump_threshold_) {
        return;
    }

    while (static_cast<int>(state.raw_window.size()) >= prefilter_window_) {
        state.raw_window.pop_front();
    }
    state.raw_window.push_back(raw);
    state.last_raw = raw;
    const cv::Point2d filtered_raw = prefilter(state.raw_window);

    if (state.kalman.has_measurement()) {
        const auto prediction = state.kalman.estimate();
        if (prediction.has_value() &&
            squared_distance(filtered_raw, *prediction) >
                pred_jump_threshold_ * pred_jump_threshold_) {
            return;
        }
    }

    state.kalman.update(filtered_raw);
    state.last_update = now_seconds();
    if (motion_callback_) {
        const auto estimate = state.kalman.estimate();
        if (estimate.has_value()) {
            motion_callback_(name, estimate->x, estimate->y, state.last_update);
        }
    }
}

std::map<std::string, cv::Point2d> HybridGatedKalmanFilter::get_all_data() {
    const double timestamp = now_seconds();
    std::map<std::string, cv::Point2d> output;
    for (auto iterator = targets_.begin(); iterator != targets_.end();) {
        if (timestamp - iterator->second.last_update > max_inactive_time_) {
            iterator = targets_.erase(iterator);
            continue;
        }
        const auto estimate = iterator->second.kalman.estimate();
        if (estimate.has_value()) {
            output[iterator->first] = *estimate;
        }
        ++iterator;
    }
    return output;
}

void HybridGatedKalmanFilter::set_motion_callback(MotionCallback callback) {
    motion_callback_ = std::move(callback);
}

std::unique_ptr<PositionFilter> make_filter(
    const AppConfig& config,
    const std::string& type,
    MotionCallback callback
) {
    std::string normalized = type;
    if (normalized == "hybrid") {
        normalized = "hybrid_gated_kalman";
    }
    if (normalized == "sliding_window") {
        return std::make_unique<SlidingWindowFilter>(
            config.sliding_window_size,
            config.sliding_max_inactive_time,
            config.sliding_threshold,
            std::move(callback)
        );
    }
    if (normalized == "hybrid_gated_kalman") {
        return std::make_unique<HybridGatedKalmanFilter>(
            config.hybrid_prefilter_window,
            config.hybrid_prefilter_mode,
            config.hybrid_max_inactive_time,
            config.hybrid_raw_jump_threshold,
            config.hybrid_pred_jump_threshold,
            config.hybrid_process_noise,
            config.hybrid_measurement_noise,
            std::move(callback)
        );
    }
    throw std::runtime_error("unsupported filter type: " + type);
}

}  // namespace vision
