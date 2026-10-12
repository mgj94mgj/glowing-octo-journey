#pragma once

#include "vision/types.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace vision {

class LatestQueue {
public:
    void push(const RadioPositionFrame& frame);
    std::optional<RadioPositionFrame> pop_latest();

private:
    std::mutex mutex_;
    std::optional<RadioPositionFrame> value_;
};

class RadioBridge {
public:
    RadioBridge();
    ~RadioBridge();

    RadioBridge(const RadioBridge&) = delete;
    RadioBridge& operator=(const RadioBridge&) = delete;

    void start();
    void stop();
    LatestQueue& positions();
    void publish_feedback(const std::string& json);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision
