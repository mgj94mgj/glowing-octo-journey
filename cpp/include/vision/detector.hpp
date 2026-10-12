#pragma once

#include "vision/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vision {

struct LetterboxInfo {
    double ratio = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};

class TensorRtBackend;

class Detector {
public:
    Detector(
        const std::filesystem::path& weights,
        cv::Size input_size,
        float confidence_threshold,
        float iou_threshold,
        int max_detections,
        std::vector<std::string> names = {}
    );
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    std::vector<Detection> predict(const cv::Mat& image);
    cv::Size input_size() const;

private:
    cv::Mat letterbox(const cv::Mat& image, LetterboxInfo& info) const;
    std::vector<Detection> decode(
        const cv::Mat& output,
        const LetterboxInfo& info,
        const cv::Size& original_size
    ) const;

    std::filesystem::path weights_;
    cv::Size input_size_;
    float confidence_threshold_;
    float iou_threshold_;
    int max_detections_;
    std::vector<std::string> names_;
    cv::dnn::Net net_;
    std::unique_ptr<TensorRtBackend> tensorrt_;
};

std::vector<Detection> non_max_suppression(
    const std::vector<Detection>& detections,
    float iou_threshold,
    int max_detections
);

}  // namespace vision
