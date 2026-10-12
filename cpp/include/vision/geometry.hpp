#pragma once

#include "vision/types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>

namespace vision {

std::vector<cv::Mat> load_calibration_npy(const std::filesystem::path& path);

cv::Point2d project_camera_point(
    const cv::Point2d& camera_point,
    const cv::Mat& ground_transform,
    const cv::Mat& height_transform,
    const cv::Mat& mask
);

SerialPoint filter_xy_to_serial(double x, double y, char state);
std::pair<double, double> serial_to_map_point(const SerialPoint& point);

}  // namespace vision
