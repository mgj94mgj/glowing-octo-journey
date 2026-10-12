#pragma once

#include "vision/types.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace vision {

class FfmpegRecorder {
public:
    FfmpegRecorder(const AppConfig& config, const cv::Size& frame_size);
    ~FfmpegRecorder();

    FfmpegRecorder(const FfmpegRecorder&) = delete;
    FfmpegRecorder& operator=(const FfmpegRecorder&) = delete;

    void start();
    void submit(const cv::Mat& frame);
    void stop();

private:
    void worker_loop();

    bool enabled_ = false;
    std::string output_dir_;
    double fps_ = 30.0;
    int crf_ = 23;
    std::string preset_ = "ultrafast";
    int threads_ = 2;
    cv::Size frame_size_;
    std::size_t queue_limit_ = 20;
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_requested_{false};
    std::FILE* pipe_ = nullptr;
    std::thread worker_;
    std::uint64_t dropped_frames_ = 0;
};

}  // namespace vision
