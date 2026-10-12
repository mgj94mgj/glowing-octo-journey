#include "vision/radio.hpp"

#include "vision/protocol.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef VISION_ENABLE_ROS2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#endif

namespace vision {
namespace {

std::uint16_t read_u16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8U);
}

std::optional<RadioPositionFrame> parse_info_wave(const std::vector<std::uint8_t>& raw) {
    constexpr std::uint8_t kInfoWaveId = 0xD5;
    constexpr std::size_t kInfoWaveLength = 94;
    constexpr std::uint8_t kPositionValid = 0x01;
    if (raw.size() != kInfoWaveLength || raw[0] != kInfoWaveId) {
        return std::nullopt;
    }
    const std::uint16_t received_crc = read_u16(raw.data() + raw.size() - 2);
    if (protocol::crc16(raw.data(), raw.size() - 2) != received_crc) {
        return std::nullopt;
    }
    if ((raw[1] & kPositionValid) == 0U) {
        return std::nullopt;
    }

    RadioPositionFrame frame;
    frame.valid_mask = raw[1];
    std::size_t offset = 2;
    for (SerialPoint& point : frame.robots) {
        point.x = read_u16(raw.data() + offset);
        point.y = read_u16(raw.data() + offset + 2);
        offset += 4;
    }
    return frame;
}

}  // namespace

void LatestQueue::push(const RadioPositionFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = frame;
}

std::optional<RadioPositionFrame> LatestQueue::pop_latest() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::optional<RadioPositionFrame> output = std::move(value_);
    value_.reset();
    return output;
}

class RadioBridge::Impl {
public:
    LatestQueue positions;

    void start() {
#ifdef VISION_ENABLE_ROS2
        if (started) {
            return;
        }
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
            owns_context = true;
        }
        listener = std::make_shared<ListenerNode>(&positions);
        feedback = std::make_shared<FeedbackNode>();
        executor.add_node(listener);
        executor.add_node(feedback);
        thread = std::thread([this]() { executor.spin(); });
        started = true;
#else
        std::cout << "ROS2 radio bridge disabled at build time; /Radio is not subscribed\n";
#endif
    }

    void stop() {
#ifdef VISION_ENABLE_ROS2
        if (!started) {
            return;
        }
        executor.cancel();
        if (thread.joinable()) {
            thread.join();
        }
        executor.remove_node(listener);
        executor.remove_node(feedback);
        listener.reset();
        feedback.reset();
        if (owns_context && rclcpp::ok()) {
            rclcpp::shutdown();
        }
        started = false;
        owns_context = false;
#endif
    }

    void publish_feedback(const std::string& json) {
#ifdef VISION_ENABLE_ROS2
        if (feedback) {
            feedback->set_feedback(json);
        }
#else
        (void)json;
#endif
    }

private:
#ifdef VISION_ENABLE_ROS2
    class ListenerNode final : public rclcpp::Node {
    public:
        explicit ListenerNode(LatestQueue* queue)
            : Node("radio_listener"), queue_(queue) {
            subscription_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
                "/Radio",
                10,
                [this](std_msgs::msg::UInt8MultiArray::SharedPtr message) {
                    std::vector<std::uint8_t> raw;
                    raw.reserve(message->data.size());
                    for (std::uint8_t value : message->data) {
                        raw.push_back(value);
                    }
                    const auto frame = parse_info_wave(raw);
                    if (frame.has_value()) {
                        queue_->push(*frame);
                    }
                }
            );
            RCLCPP_INFO(get_logger(), "Listening /Radio ...");
        }

    private:
        LatestQueue* queue_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr subscription_;
    };

    class FeedbackNode final : public rclcpp::Node {
    public:
        FeedbackNode() : Node("lower_feedback_publisher") {
            publisher_ = create_publisher<std_msgs::msg::UInt8MultiArray>("/RadioFeedback", 1);
            timer_ = create_wall_timer(
                std::chrono::seconds(1),
                [this]() {
                    std::string json;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        json = feedback_;
                    }
                    if (json.empty()) {
                        return;
                    }
                    std_msgs::msg::UInt8MultiArray message;
                    message.data.reserve(json.size());
                    for (unsigned char byte : json) {
                        message.data.push_back(byte);
                    }
                    publisher_->publish(message);
                    RCLCPP_INFO(get_logger(), "send /RadioFeedback: %s", json.c_str());
                }
            );
        }

        void set_feedback(const std::string& value) {
            std::lock_guard<std::mutex> lock(mutex_);
            feedback_ = value;
        }

    private:
        rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
        std::mutex mutex_;
        std::string feedback_;
    };

    bool started = false;
    bool owns_context = false;
    rclcpp::executors::SingleThreadedExecutor executor;
    std::shared_ptr<ListenerNode> listener;
    std::shared_ptr<FeedbackNode> feedback;
    std::thread thread;
#endif
};

RadioBridge::RadioBridge() : impl_(std::make_unique<Impl>()) {}

RadioBridge::~RadioBridge() {
    stop();
}

void RadioBridge::start() {
    impl_->start();
}

void RadioBridge::stop() {
    impl_->stop();
}

LatestQueue& RadioBridge::positions() {
    return impl_->positions;
}

void RadioBridge::publish_feedback(const std::string& json) {
    impl_->publish_feedback(json);
}

}  // namespace vision
