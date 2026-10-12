#include "vision/ui.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <string>

namespace vision {

std::array<int, 6> draw_information_ui(
    const std::array<int, 6>& bars,
    char camp,
    cv::Mat& image
) {
    cv::line(image, cv::Point(300, 0), cv::Point(300, 300), cv::Scalar(0, 150, 0), 2);
    std::array<int, 6> highlighted{};
    constexpr std::array<int, 6> robot_numbers = {1, 2, 3, 4, 6, 7};
    const std::string enemy_prefix = camp == 'R' ? "B" : "R";

    constexpr int max_value = 120;
    constexpr int threshold = 100;
    constexpr int max_length = 300;
    const int segment_height = max_length / static_cast<int>(bars.size());

    for (std::size_t index = 0; index < bars.size(); ++index) {
        const int value = std::max(0, bars[index]);
        const int line_length = static_cast<int>(
            (static_cast<double>(value) / max_value) * max_length
        );
        const int line_height = value > threshold ? 8 : 3;
        if (value > threshold) {
            highlighted[index] = 1;
        }

        const cv::Scalar color =
            camp == 'R'
                ? (value > threshold ? cv::Scalar(255, 0, 0) : cv::Scalar(200, 0, 0))
                : (value > threshold ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 200));
        const int center_y = static_cast<int>(index) * segment_height + segment_height / 2;
        cv::line(
            image,
            cv::Point(50, center_y),
            cv::Point(50 + line_length, center_y),
            color,
            line_height,
            cv::LINE_AA
        );
        const std::string robot_name =
            index < robot_numbers.size()
                ? enemy_prefix + std::to_string(robot_numbers[index])
                : enemy_prefix + "?";
        cv::putText(
            image,
            robot_name,
            cv::Point(10, center_y + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA
        );
        cv::putText(
            image,
            std::to_string(value),
            cv::Point(370, center_y + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA
        );
    }
    return highlighted;
}

void draw_detection(cv::Mat& image, const Detection& detection, const cv::Scalar& color) {
    const cv::Rect box = detection.box;
    cv::rectangle(image, box, color, 2);
    cv::putText(
        image,
        detection.cls + " " + cv::format("%.2f", detection.confidence),
        cv::Point(box.x, std::max(box.y - 5, 20)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    );
}

}  // namespace vision
