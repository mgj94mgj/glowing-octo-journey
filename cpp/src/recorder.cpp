#include "vision/recorder.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

namespace fs = std::filesystem;

namespace vision {
namespace {

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &value);
#else
    localtime_r(&value, &local);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string shell_quote(const std::string& value) {
    std::string output = "'";
    for (char character : value) {
        if (character == '\'') {
            output += "'\\''";
        } else {
            output.push_back(character);
        }
    }
    output.push_back('\'');
    return output;
}

bool is_mount_point(const fs::path& path) {
#ifdef _WIN32
    (void)path;
    return true;
#else
    struct stat current {};
    struct stat parent {};
    if (stat(path.c_str(), &current) != 0) {
        return false;
    }
    if (stat(path.parent_path().c_str(), &parent) != 0) {
        return false;
    }
    return current.st_dev != parent.st_dev;
#endif
}

}  // namespace

FfmpegRecorder::FfmpegRecorder(const AppConfig& config, const cv::Size& frame_size)
    : enabled_(config.video_recording_enabled),
      output_dir_(config.video_recording_output_dir),
      fps_(std::max(1.0, config.video_recording_fps)),
      crf_(std::clamp(config.video_recording_crf, 0, 51)),
      preset_(config.video_recording_preset),
      threads_(std::max(1, config.video_recording_threads)),
      frame_size_(frame_size),
      queue_limit_(static_cast<std::size_t>(std::max(1, config.video_recording_queue_size))) {}

FfmpegRecorder::~FfmpegRecorder() {
    stop();
}

void FfmpegRecorder::start() {
    if (!enabled_) {
        std::cout << "video recording disabled\n";
        return;
    }
    if (frame_size_.width <= 0 || frame_size_.height <= 0) {
        enabled_ = false;
        return;
    }

    const fs::path mount_point("/media/ajian/AJian");
    const fs::path output_path(output_dir_);
    if (output_path.string().rfind(mount_point.string() + "/", 0) != 0 ||
        !is_mount_point(mount_point)) {
        std::cerr
            << "video recording not started: USB drive is not mounted at "
            << mount_point << '\n';
        enabled_ = false;
        return;
    }
    std::error_code error;
    fs::create_directories(output_path, error);
    if (error) {
        std::cerr << "failed to create recording directory: " << error.message() << '\n';
        enabled_ = false;
        return;
    }

    const fs::path output_file = output_path / ("radar_" + timestamp() + ".mp4");
    std::ostringstream command;
    command << "ffmpeg -hide_banner -loglevel error -y"
            << " -f rawvideo -pix_fmt bgr24"
            << " -s " << frame_size_.width << "x" << frame_size_.height
            << " -r " << fps_
            << " -i -"
            << " -an -c:v libx264 -preset " << preset_
            << " -crf " << crf_
            << " -threads " << threads_
            << " " << shell_quote(output_file.string());
    pipe_ = popen(command.str().c_str(), "w");
    if (pipe_ == nullptr) {
        std::cerr << "failed to start ffmpeg for video recording\n";
        enabled_ = false;
        return;
    }
    stop_requested_ = false;
    worker_ = std::thread(&FfmpegRecorder::worker_loop, this);
    std::cout << "recording to " << output_file << '\n';
}

void FfmpegRecorder::submit(const cv::Mat& frame) {
    if (!enabled_ || pipe_ == nullptr || frame.empty()) {
        return;
    }
    if (frame.size() != frame_size_ || frame.type() != CV_8UC3) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= queue_limit_) {
        ++dropped_frames_;
        return;
    }
    queue_.push(frame.clone());
    condition_.notify_one();
}

void FfmpegRecorder::worker_loop() {
    while (true) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() {
                return stop_requested_ || !queue_.empty();
            });
            if (stop_requested_ && queue_.empty()) {
                break;
            }
            frame = std::move(queue_.front());
            queue_.pop();
        }
        if (pipe_ != nullptr && !frame.empty()) {
            const std::size_t bytes =
                static_cast<std::size_t>(frame.cols) * frame.rows * frame.channels();
            if (std::fwrite(frame.data, 1, bytes, pipe_) != bytes) {
                std::cerr << "ffmpeg recording pipe write failed\n";
                break;
            }
        }
    }
}

void FfmpegRecorder::stop() {
    if (!enabled_ && pipe_ == nullptr) {
        return;
    }
    stop_requested_ = true;
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    if (pipe_ != nullptr) {
        pclose(pipe_);
        pipe_ = nullptr;
    }
    enabled_ = false;
    if (dropped_frames_ > 0) {
        std::cout << "recording dropped " << dropped_frames_ << " frame(s)\n";
    }
}

}  // namespace vision
