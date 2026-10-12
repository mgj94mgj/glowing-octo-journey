#include "vision/camera.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef VISION_ENABLE_MVS
#include <MvCameraControl.h>
#endif

namespace vision {
namespace {

double fps_delay(double fps, double speed) {
    return 1.0 / std::max(1e-6, fps) / std::max(1e-6, speed);
}

}  // namespace

CameraSource::CameraSource(const AppConfig& config) : config_(config) {}

CameraSource::~CameraSource() {
    stop();
}

void CameraSource::start() {
    if (worker_.joinable()) {
        return;
    }
    stop_requested_ = false;
    worker_ = std::thread(&CameraSource::run, this);
}

void CameraSource::stop() {
    stop_requested_ = true;
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void CameraSource::set_frame_callback(std::function<void(const cv::Mat&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_callback_ = std::move(callback);
}

void CameraSource::run() {
    if (config_.camera_mode == "hik") {
        run_hik();
    } else if (config_.camera_mode == "video") {
        run_video();
    } else {
        run_test();
    }
}

void CameraSource::run_video() {
    cv::VideoCapture capture(0);
    while (!stop_requested_) {
        if (!capture.isOpened()) {
            std::cerr << "USB camera open failed; retrying\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            capture.open(0);
            continue;
        }
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "USB camera read failed; reconnecting\n";
            capture.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        publish(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    capture.release();
}

void CameraSource::run_test() {
    if (!config_.test_video_path.empty() && std::filesystem::exists(config_.test_video_path)) {
        cv::VideoCapture capture(config_.test_video_path);
        if (!capture.isOpened()) {
            throw std::runtime_error("failed to open test video: " + config_.test_video_path);
        }
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) {
            fps = 30.0;
        }
        capture.set(cv::CAP_PROP_POS_FRAMES, std::max(0, config_.test_start_frame));
        while (!stop_requested_) {
            cv::Mat frame;
            if (capture.read(frame) && !frame.empty()) {
                publish(frame);
                const auto wait = std::chrono::duration<double>(
                    fps_delay(fps, config_.test_playback_speed)
                );
                std::this_thread::sleep_for(wait);
                continue;
            }
            if (!config_.test_loop) {
                break;
            }
            capture.set(cv::CAP_PROP_POS_FRAMES, std::max(0, config_.test_start_frame));
        }
        return;
    }

    cv::Mat image = cv::imread(config_.test_image_path);
    if (image.empty()) {
        throw std::runtime_error("failed to open test image: " + config_.test_image_path);
    }
    while (!stop_requested_) {
        publish(image);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void CameraSource::run_hik() {
#ifndef VISION_ENABLE_MVS
    throw std::runtime_error("hik camera mode requires the MVS SDK and VISION_ENABLE_MVS=ON");
#else
    MV_CC_DEVICE_INFO_LIST device_list{};
    const int enumerate_result = MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE,
        &device_list
    );
    if (enumerate_result != MV_OK || device_list.nDeviceNum == 0) {
        throw std::runtime_error("no Hikrobot camera found");
    }

    auto serial = [](const MV_CC_DEVICE_INFO* info) -> std::string {
        char buffer[64]{};
        if (info->nTLayerType == MV_GIGE_DEVICE) {
            std::memcpy(
                buffer,
                info->SpecialInfo.stGigEInfo.chSerialNumber,
                sizeof(info->SpecialInfo.stGigEInfo.chSerialNumber)
            );
        } else if (info->nTLayerType == MV_USB_DEVICE) {
            std::memcpy(
                buffer,
                info->SpecialInfo.stUsb3VInfo.chSerialNumber,
                sizeof(info->SpecialInfo.stUsb3VInfo.chSerialNumber)
            );
        }
        return std::string(buffer);
    };

    std::size_t selected = 0;
    if (!config_.camera_device_serial.empty()) {
        bool found = false;
        for (unsigned int index = 0; index < device_list.nDeviceNum; ++index) {
            if (serial(device_list.pDeviceInfo[index]) == config_.camera_device_serial) {
                selected = index;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "configured Hikrobot camera serial was not found: " +
                config_.camera_device_serial
            );
        }
    }

    void* handle = nullptr;
    int result = MV_CC_CreateHandle(&handle, device_list.pDeviceInfo[selected]);
    if (result != MV_OK) {
        throw std::runtime_error("MV_CC_CreateHandle failed");
    }
    result = MV_CC_OpenDevice(handle, MV_ACCESS_Exclusive, 0);
    if (result != MV_OK) {
        MV_CC_DestroyHandle(handle);
        throw std::runtime_error("MV_CC_OpenDevice failed");
    }

    MV_CC_SetFloatValue(handle, "ExposureTime", static_cast<float>(config_.camera_exposure_time));
    MV_CC_SetFloatValue(handle, "Gain", static_cast<float>(config_.camera_gain));
    result = MV_CC_StartGrabbing(handle);
    if (result != MV_OK) {
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        throw std::runtime_error("MV_CC_StartGrabbing failed");
    }

    while (!stop_requested_) {
        MV_FRAME_OUT frame{};
        result = MV_CC_GetImageBuffer(handle, &frame, 1000);
        if (result != MV_OK || frame.pBufAddr == nullptr) {
            std::cerr << "Hikrobot frame timeout/error: 0x" << std::hex << result << std::dec << '\n';
            continue;
        }

        cv::Mat image;
        const int width = static_cast<int>(frame.stFrameInfo.nWidth);
        const int height = static_cast<int>(frame.stFrameInfo.nHeight);
        const auto pixel_type = static_cast<unsigned int>(frame.stFrameInfo.enPixelType);
        try {
            if (pixel_type == 17301505U) {
                cv::Mat gray(height, width, CV_8UC1, frame.pBufAddr);
                cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
            } else if (pixel_type == 17301513U) {
                cv::Mat bayer(height, width, CV_8UC1, frame.pBufAddr);
                cv::cvtColor(bayer, image, cv::COLOR_BayerRG2BGR);
            } else if (pixel_type == 35127316U) {
                cv::Mat rgb(height, width, CV_8UC3, frame.pBufAddr);
                cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);
            } else if (pixel_type == 34603039U) {
                cv::Mat yuv422(height, width, CV_8UC2, frame.pBufAddr);
                cv::cvtColor(yuv422, image, cv::COLOR_YUV2BGR_Y422);
            } else {
                std::cerr << "unsupported Hikrobot pixel type: " << pixel_type << '\n';
            }
        } catch (const cv::Exception& exception) {
            std::cerr << "Hikrobot frame conversion failed: " << exception.what() << '\n';
        }
        MV_CC_FreeImageBuffer(handle, &frame);
        if (!image.empty()) {
            publish(image);
        }
    }

    MV_CC_StopGrabbing(handle);
    MV_CC_CloseDevice(handle);
    MV_CC_DestroyHandle(handle);
#endif
}

void CameraSource::publish(const cv::Mat& frame) {
    std::function<void(const cv::Mat&)> callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = frame.clone();
        callback = frame_callback_;
    }
    condition_.notify_all();
    if (callback) {
        callback(frame);
    }
}

bool CameraSource::wait_for_first_frame(double timeout_seconds) {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock,
        std::chrono::duration<double>(timeout_seconds),
        [this]() { return !latest_.empty() || stop_requested_; }
    ) && !latest_.empty();
}

cv::Mat CameraSource::latest_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_.clone();
}

}  // namespace vision
