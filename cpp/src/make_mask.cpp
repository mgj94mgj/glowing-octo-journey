#include "vision/config.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct MaskTool {
    cv::Mat original;
    cv::Mat scaled;
    cv::Mat mask;
    cv::Mat preview;
    std::vector<cv::Point> points;
    bool drawing = false;
    bool green_mode = true;
    double scale = 1.0;
};

void on_mouse(int event, int x, int y, int, void* user_data) {
    auto* tool = static_cast<MaskTool*>(user_data);
    if (tool == nullptr || tool->scaled.empty()) {
        return;
    }
    const cv::Point point(
        std::clamp(static_cast<int>(std::round(x / tool->scale)), 0, tool->original.cols - 1),
        std::clamp(static_cast<int>(std::round(y / tool->scale)), 0, tool->original.rows - 1)
    );
    if (event == cv::EVENT_LBUTTONDOWN) {
        tool->drawing = true;
        tool->points = {point};
    } else if (event == cv::EVENT_MOUSEMOVE && tool->drawing) {
        tool->points.push_back(point);
    } else if (event == cv::EVENT_LBUTTONUP && tool->drawing) {
        tool->drawing = false;
        tool->points.push_back(point);
        if (tool->points.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours{tool->points};
            cv::fillPoly(
                tool->mask,
                contours,
                tool->green_mode ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0)
            );
        }
    }
    if (!tool->scaled.empty()) {
        cv::Mat scaled_mask;
        cv::resize(tool->mask, scaled_mask, tool->scaled.size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::Mat preview;
        cv::addWeighted(
            tool->scaled,
            0.5,
            scaled_mask,
            0.5,
            0.0,
            preview
        );
        tool->preview = preview;
    }
}

}  // namespace

int main() {
    try {
        const fs::path config_path =
            fs::exists(fs::current_path() / "config.yaml")
                ? fs::current_path() / "config.yaml"
                : vision::default_project_root() / "config.yaml";
        const vision::AppConfig config = vision::ConfigLoader::load(config_path);

        MaskTool tool;
        tool.original = cv::imread(config.map_path);
        if (tool.original.empty()) {
            throw std::runtime_error("failed to load map image: " + config.map_path);
        }
        constexpr int target_width = 430;
        tool.scale = static_cast<double>(target_width) / tool.original.cols;
        cv::resize(
            tool.original,
            tool.scaled,
            cv::Size(target_width, static_cast<int>(tool.original.rows * tool.scale))
        );
        tool.mask = cv::Mat::zeros(tool.original.size(), CV_8UC3);
        tool.preview = tool.scaled.clone();

        const std::string window = "Image";
        cv::namedWindow(window);
        cv::setMouseCallback(window, on_mouse, &tool);
        std::cout
            << "drag to draw; n switches green/blue; q saves "
            << config.mask_path << '\n';

        while (true) {
            cv::imshow(window, tool.preview);
            const int key = cv::waitKey(30);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
            if (key == 'n' || key == 'N') {
                tool.green_mode = !tool.green_mode;
            }
        }
        cv::destroyAllWindows();
        if (!cv::imwrite(config.mask_path, tool.mask)) {
            throw std::runtime_error("failed to write mask: " + config.mask_path);
        }
        std::cout << "mask saved: " << config.mask_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "fatal: " << exception.what() << '\n';
        return 1;
    }
}
