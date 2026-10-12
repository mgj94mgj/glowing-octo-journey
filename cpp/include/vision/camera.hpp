#pragma once

#include "vision/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace vision {

class CameraSource {
public:
    explicit CameraSource(const AppConfig& config);
    ~CameraSource();

    CameraSource(const CameraSource&) = delete;
    CameraSource& operator=(const CameraSource&) = delete;

    void start();
    void stop();
    void set_frame_callback(std::function<void(const cv::Mat&)> callback);
    bool wait_for_first_frame(double timeout_seconds = 30.0);
    cv::Mat latest_frame() const;

private:
    void run();
    void run_video();
    void run_test();
    void run_hik();
    void publish(const cv::Mat& frame);

    const AppConfig& config_;
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    cv::Mat latest_;
    std::thread worker_;
    std::function<void(const cv::Mat&)> frame_callback_;
};

}  // namespace vision
