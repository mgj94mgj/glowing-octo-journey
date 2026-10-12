#include "vision/geometry.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace vision {
namespace {

template <typename T>
T read_scalar(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("truncated npy data");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

cv::Point2d transform_point(const cv::Point2d& point, const cv::Mat& transform) {
    std::vector<cv::Point2d> input{point};
    std::vector<cv::Point2d> output;
    cv::perspectiveTransform(input, output, transform);
    return output.front();
}

cv::Point2d clamp_map_point(const cv::Point2d& point, const cv::Mat& mask) {
    const int max_x = std::max(0, mask.cols - 1);
    const int max_y = std::max(0, mask.rows - 1);
    return cv::Point2d(
        std::clamp(static_cast<int>(point.x), 0, max_x),
        std::clamp(static_cast<int>(point.y), 0, max_y)
    );
}

}  // namespace

std::vector<cv::Mat> load_calibration_npy(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open calibration file: " + path.string());
    }

    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    if (bytes.size() < 10 || bytes[0] != 0x93 ||
        std::string(bytes.begin() + 1, bytes.begin() + 6) != "NUMPY") {
        throw std::runtime_error("not a NumPy .npy file: " + path.string());
    }

    const int major = bytes[6];
    std::size_t header_offset = 10;
    std::size_t header_length = bytes[8] | (static_cast<std::size_t>(bytes[9]) << 8U);
    if (major >= 2) {
        if (bytes.size() < 12) {
            throw std::runtime_error("invalid npy v2 header");
        }
        header_length = static_cast<std::size_t>(bytes[8]) |
                        (static_cast<std::size_t>(bytes[9]) << 8U) |
                        (static_cast<std::size_t>(bytes[10]) << 16U) |
                        (static_cast<std::size_t>(bytes[11]) << 24U);
        header_offset = 12;
    }
    if (header_offset + header_length > bytes.size()) {
        throw std::runtime_error("truncated npy header");
    }

    const std::string header(
        reinterpret_cast<const char*>(bytes.data() + header_offset),
        header_length
    );
    if (header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-order npy files are not supported");
    }

    std::smatch descr_match;
    if (!std::regex_search(header, descr_match, std::regex("'descr'\\s*:\\s*'([^']+)'"))) {
        throw std::runtime_error("missing descr in npy header");
    }
    const std::string descr = descr_match[1].str();

    std::smatch shape_match;
    if (!std::regex_search(header, shape_match, std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)"))) {
        throw std::runtime_error("missing shape in npy header");
    }
    std::vector<int> shape;
    const std::string shape_text = shape_match[1].str();
    std::regex number_pattern("-?\\d+");
    for (std::sregex_iterator iterator(shape_text.begin(), shape_text.end(), number_pattern), end;
         iterator != end;
         ++iterator) {
        shape.push_back(std::stoi(iterator->str()));
    }
    if (shape.empty()) {
        throw std::runtime_error("invalid npy shape");
    }

    std::size_t total = 1;
    for (int dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("negative npy dimension");
        }
        total *= static_cast<std::size_t>(dimension);
    }

    std::vector<double> values(total);
    std::size_t offset = header_offset + header_length;
    if (descr == "<f8" || descr == "|f8") {
        for (std::size_t i = 0; i < total; ++i) {
            values[i] = read_scalar<double>(bytes, offset + i * 8);
        }
    } else if (descr == "<f4" || descr == "|f4") {
        for (std::size_t i = 0; i < total; ++i) {
            values[i] = read_scalar<float>(bytes, offset + i * 4);
        }
    } else if (descr == "|u1") {
        for (std::size_t i = 0; i < total; ++i) {
            values[i] = read_scalar<std::uint8_t>(bytes, offset + i);
        }
    } else {
        throw std::runtime_error("unsupported npy dtype: " + descr);
    }

    if (shape.size() == 1) {
        return {cv::Mat(shape[0], 1, CV_64F, values.data()).clone()};
    }
    if (shape.size() == 2) {
        return {cv::Mat(shape[0], shape[1], CV_64F, values.data()).clone()};
    }
    if (shape.size() == 3) {
        std::vector<cv::Mat> matrices;
        matrices.reserve(shape[0]);
        const std::size_t matrix_size = static_cast<std::size_t>(shape[1] * shape[2]);
        for (int index = 0; index < shape[0]; ++index) {
            matrices.emplace_back(
                shape[1],
                shape[2],
                CV_64F,
                values.data() + static_cast<std::size_t>(index) * matrix_size
            );
        }
        return matrices;
    }
    throw std::runtime_error("npy arrays with more than 3 dimensions are not supported");
}

cv::Point2d project_camera_point(
    const cv::Point2d& camera_point,
    const cv::Mat& ground_transform,
    const cv::Mat& height_transform,
    const cv::Mat& mask
) {
    cv::Point2d mapped = clamp_map_point(transform_point(camera_point, ground_transform), mask);
    const cv::Vec3b color = mask.at<cv::Vec3b>(
        static_cast<int>(mapped.y),
        static_cast<int>(mapped.x)
    );
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        return mapped;
    }
    return clamp_map_point(transform_point(camera_point, height_transform), mask);
}

SerialPoint filter_xy_to_serial(double x, double y, char state) {
    if (state == 'R') {
        return SerialPoint{
            static_cast<int>(2800.0 - y),
            static_cast<int>(1500.0 - x),
        };
    }
    return SerialPoint{static_cast<int>(y), static_cast<int>(x)};
}

std::pair<double, double> serial_to_map_point(const SerialPoint& point) {
    return {static_cast<double>(point.x), static_cast<double>(1500 - point.y)};
}

}  // namespace vision
