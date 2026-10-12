#pragma once

#include "vision/types.hpp"

#include <opencv2/core.hpp>

#include <array>

namespace vision {

std::array<int, 6> draw_information_ui(
    const std::array<int, 6>& bars,
    char camp,
    cv::Mat& image
);

void draw_detection(cv::Mat& image, const Detection& detection, const cv::Scalar& color);

}  // namespace vision
