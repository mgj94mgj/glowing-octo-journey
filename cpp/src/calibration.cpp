#include "vision/camera.hpp"
#include "vision/config.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CalibrationUi {
    std::array<std::array<std::optional<cv::Point2d>, 4>, 2> image_points;
    std::array<std::array<std::optional<cv::Point2d>, 4>, 2> map_points;
    int height = 0;
    int image_count = 0;
    int map_count = 0;
    bool capturing = true;
    cv::Mat camera_frame;
    cv::Mat camera_preview;
    cv::Mat map_image;
    cv::Mat map_preview;
    double camera_scale_x = 1.0;
    double camera_scale_y = 1.0;
    double map_scale_x = 1.0;
    double map_scale_y = 1.0;
};

void draw_status(CalibrationUi& ui) {
    cv::putText(
        ui.camera_preview,
        ui.capturing ? "Press s to freeze; drag on both views; c saves; q quits"
                     : "Click 4 camera and 4 map points; h changes height; c saves",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        ui.capturing ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255),
        2
    );
    cv::putText(
        ui.camera_preview,
        "height=" + std::to_string(ui.height + 1) +
            " image=" + std::to_string(ui.image_count % 4) +
            " map=" + std::to_string(ui.map_count % 4),
        cv::Point(10, 65),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255, 255, 255),
        2
    );
}

void on_camera_mouse(int event, int x, int y, int, void* user_data) {
    auto* ui = static_cast<CalibrationUi*>(user_data);
    if (ui == nullptr || ui->capturing || event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    const cv::Point2d point(x * ui->camera_scale_x, y * ui->camera_scale_y);
    ui->image_points[static_cast<std::size_t>(ui->height)]
                    [static_cast<std::size_t>(ui->image_count % 4)] = point;
    const cv::Scalar color =
        ui->height == 0 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 255, 0);
    cv::circle(ui->camera_preview, cv::Point(x, y), 5, color, cv::FILLED);
    cv::putText(
        ui->camera_preview,
        std::to_string(ui->image_count % 4),
        cv::Point(x + 6, y - 6),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    );
    ++ui->image_count;
}

void on_map_mouse(int event, int x, int y, int, void* user_data) {
    auto* ui = static_cast<CalibrationUi*>(user_data);
    if (ui == nullptr || ui->capturing || event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    const cv::Point2d point(x * ui->map_scale_x, y * ui->map_scale_y);
    ui->map_points[static_cast<std::size_t>(ui->height)]
                  [static_cast<std::size_t>(ui->map_count % 4)] = point;
    const cv::Scalar color =
        ui->height == 0 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 255, 0);
    cv::circle(ui->map_preview, cv::Point(x, y), 5, color, cv::FILLED);
    cv::putText(
        ui->map_preview,
        std::to_string(ui->map_count % 4),
        cv::Point(x + 6, y - 6),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    );
    ++ui->map_count;
}

bool valid_points(
    const std::array<std::optional<cv::Point2d>, 4>& image_points,
    const std::array<std::optional<cv::Point2d>, 4>& map_points
) {
    for (const auto* points : {&image_points, &map_points}) {
        for (const auto& point : *points) {
            if (!point.has_value()) {
                return false;
            }
        }
        std::vector<cv::Point2f> contour;
        contour.reserve(4);
        for (const auto& point : *points) {
            contour.emplace_back(
                static_cast<float>(point->x),
                static_cast<float>(point->y)
            );
        }
        if (std::abs(cv::contourArea(contour)) < 1.0F) {
            return false;
        }
    }
    return true;
}

void write_scalar(std::ofstream& stream, const std::string& value) {
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_npy(const fs::path& path, const std::array<cv::Mat, 2>& matrices) {
    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3, 3), }";
    std::string header_text = header.str();
    const std::size_t preamble = 10;
    const std::size_t padding =
        (16 - ((preamble + header_text.size() + 1) % 16)) % 16;
    header_text.append(padding, ' ');
    header_text.push_back('\n');
    if (header_text.size() > 0xFFFFU) {
        throw std::runtime_error("npy header is too large");
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to write calibration: " + path.string());
    }
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    stream.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const std::uint16_t header_length = static_cast<std::uint16_t>(header_text.size());
    const unsigned char header_size[] = {
        static_cast<unsigned char>(header_length & 0xFFU),
        static_cast<unsigned char>((header_length >> 8U) & 0xFFU),
    };
    stream.write(reinterpret_cast<const char*>(header_size), sizeof(header_size));
    write_scalar(stream, header_text);
    for (const cv::Mat& matrix : matrices) {
        cv::Mat converted;
        matrix.convertTo(converted, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const double value = converted.at<double>(row, column);
                stream.write(
                    reinterpret_cast<const char*>(&value),
                    sizeof(value)
                );
            }
        }
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

        vision::CameraSource camera(config);
        camera.start();
        if (!camera.wait_for_first_frame()) {
            throw std::runtime_error("timed out waiting for camera frame");
        }

        CalibrationUi ui;
        ui.camera_frame = camera.latest_frame();
        ui.camera_preview = ui.camera_frame.clone();
        ui.map_image = cv::imread(config.map_path);
        if (ui.map_image.empty()) {
            throw std::runtime_error("failed to load map image: " + config.map_path);
        }
        ui.map_preview = ui.map_image.clone();
        ui.camera_scale_x =
            static_cast<double>(ui.camera_frame.cols) / ui.camera_preview.cols;
        ui.camera_scale_y =
            static_cast<double>(ui.camera_frame.rows) / ui.camera_preview.rows;
        ui.map_scale_x = static_cast<double>(ui.map_image.cols) / ui.map_preview.cols;
        ui.map_scale_y = static_cast<double>(ui.map_image.rows) / ui.map_preview.rows;

        const std::string camera_window = "calibration_camera";
        const std::string map_window = "calibration_map";
        cv::namedWindow(camera_window);
        cv::namedWindow(map_window);
        cv::setMouseCallback(camera_window, on_camera_mouse, &ui);
        cv::setMouseCallback(map_window, on_map_mouse, &ui);

        while (true) {
            if (ui.capturing) {
                ui.camera_frame = camera.latest_frame();
                if (!ui.camera_frame.empty()) {
                    ui.camera_preview = ui.camera_frame.clone();
                    ui.camera_scale_x =
                        static_cast<double>(ui.camera_frame.cols) / ui.camera_preview.cols;
                    ui.camera_scale_y =
                        static_cast<double>(ui.camera_frame.rows) / ui.camera_preview.rows;
                }
            }
            draw_status(ui);
            cv::imshow(camera_window, ui.camera_preview);
            cv::imshow(map_window, ui.map_preview);

            const int key = cv::waitKey(30);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
            if (key == 's' || key == 'S') {
                ui.capturing = false;
                std::cout << "calibration frozen\n";
            }
            if (key == 'h' || key == 'H') {
                ui.height = (ui.height + 1) % 2;
                ui.image_count = 0;
                ui.map_count = 0;
                std::cout << "height layer " << (ui.height + 1) << '\n';
            }
            if (key == 'c' || key == 'C') {
                std::array<cv::Mat, 2> transforms;
                bool valid = true;
                for (int index = 0; index < 2; ++index) {
                    if (!valid_points(
                            ui.image_points[static_cast<std::size_t>(index)],
                            ui.map_points[static_cast<std::size_t>(index)]
                        )) {
                        valid = false;
                        break;
                    }
                    std::vector<cv::Point2f> image_points;
                    std::vector<cv::Point2f> map_points;
                    for (int point = 0; point < 4; ++point) {
                        image_points.emplace_back(
                            static_cast<float>(
                                ui.image_points[static_cast<std::size_t>(index)]
                                               [static_cast<std::size_t>(point)]->x
                            ),
                            static_cast<float>(
                                ui.image_points[static_cast<std::size_t>(index)]
                                               [static_cast<std::size_t>(point)]->y
                            )
                        );
                        map_points.emplace_back(
                            static_cast<float>(
                                ui.map_points[static_cast<std::size_t>(index)]
                                             [static_cast<std::size_t>(point)]->x
                            ),
                            static_cast<float>(
                                ui.map_points[static_cast<std::size_t>(index)]
                                             [static_cast<std::size_t>(point)]->y
                            )
                        );
                    }
                    transforms[static_cast<std::size_t>(index)] =
                        cv::getPerspectiveTransform(image_points, map_points);
                }
                if (!valid) {
                    std::cerr
                        << "each height needs four distinct, non-collinear points "
                           "on both views\n";
                    continue;
                }
                write_npy(config.calibration_path, transforms);
                std::cout << "calibration saved: " << config.calibration_path << '\n';
                break;
            }
        }

        camera.stop();
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "fatal: " << exception.what() << '\n';
        return 1;
    }
}
