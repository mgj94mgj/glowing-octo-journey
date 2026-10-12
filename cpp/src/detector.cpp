#include "vision/detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef VISION_ENABLE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace fs = std::filesystem;

namespace vision {
namespace {

cv::Rect2f clamp_box(const cv::Rect2f& box, const cv::Size& size) {
    const float x1 = std::clamp(box.x, 0.0F, static_cast<float>(size.width - 1));
    const float y1 = std::clamp(box.y, 0.0F, static_cast<float>(size.height - 1));
    const float x2 = std::clamp(box.x + box.width, 0.0F, static_cast<float>(size.width - 1));
    const float y2 = std::clamp(box.y + box.height, 0.0F, static_cast<float>(size.height - 1));
    return cv::Rect2f(x1, y1, std::max(0.0F, x2 - x1), std::max(0.0F, y2 - y1));
}

float intersection_over_union(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const float intersection = (lhs & rhs).area();
    const float union_area = lhs.area() + rhs.area() - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

cv::Mat output_to_matrix(const cv::Mat& output) {
    if (output.dims <= 2) {
        return output;
    }
    std::vector<int> shape;
    for (int i = 0; i < output.dims; ++i) {
        if (output.size[i] != 1) {
            shape.push_back(output.size[i]);
        }
    }
    if (shape.empty()) {
        return output.reshape(1, 1);
    }
    if (shape.size() == 1) {
        return output.reshape(1, shape[0]);
    }
    if (shape.size() == 2) {
        return output.reshape(1, shape[0]);
    }
    const int total = static_cast<int>(output.total());
    if (shape.size() == 3) {
        if (shape[0] == 1) {
            return output.reshape(1, shape[1]);
        }
        if (shape[1] == 1) {
            return output.reshape(1, shape[0]);
        }
    }
    return output.reshape(1, total);
}

}  // namespace

#ifdef VISION_ENABLE_TENSORRT

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", message);
        }
    }
};

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result)
        );
    }
}

#endif

class TensorRtBackend {
public:
    explicit TensorRtBackend(
        const fs::path& engine_path,
        std::size_t class_count
    );
    ~TensorRtBackend();

    TensorRtBackend(const TensorRtBackend&) = delete;
    TensorRtBackend& operator=(const TensorRtBackend&) = delete;

    std::vector<cv::Mat> infer(const cv::Mat& blob);

private:
#ifdef VISION_ENABLE_TENSORRT
    struct OutputBinding {
        std::size_t index = 0;
        std::vector<std::int64_t> dimensions;
        std::size_t element_count = 0;
        void* device = nullptr;
        std::vector<float> host;
    };

    TensorRtLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::size_t input_index_ = 0;
    std::size_t input_elements_ = 0;
    void* input_device_ = nullptr;
    std::vector<OutputBinding> outputs_;
#endif
    std::size_t class_count_ = 0;
};

TensorRtBackend::TensorRtBackend(const fs::path& engine_path, std::size_t class_count)
    : class_count_(class_count) {
#ifndef VISION_ENABLE_TENSORRT
    (void)engine_path;
    throw std::runtime_error(
        "TensorRT support was not enabled at build time; configure with "
        "-DVISION_ENABLE_TENSORRT=ON or use an ONNX model"
    );
#else
    std::ifstream stream(engine_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open TensorRT engine: " + engine_path.string());
    }
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    if (bytes.empty()) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_path.string());
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) {
        throw std::runtime_error("failed to create TensorRT runtime");
    }
    engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
    if (engine_ == nullptr) {
        throw std::runtime_error("failed to deserialize TensorRT engine");
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        throw std::runtime_error("failed to create TensorRT execution context");
    }

    std::vector<void*> bindings(static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
    for (int binding_index = 0; binding_index < engine_->getNbBindings(); ++binding_index) {
        const char* name = engine_->getBindingName(binding_index);
        const nvinfer1::Dims dimensions = engine_->getBindingDimensions(binding_index);
        const std::size_t elements = static_cast<std::size_t>(
            std::accumulate(
                dimensions.d,
                dimensions.d + dimensions.nbDims,
                std::int64_t{1},
                [](std::int64_t product, std::int64_t value) {
                    return product * std::max<std::int64_t>(value, 1);
                }
            )
        );

        if (engine_->bindingIsInput(binding_index)) {
            if (input_device_ != nullptr) {
                throw std::runtime_error("TensorRT engine has more than one input binding");
            }
            input_index_ = static_cast<std::size_t>(binding_index);
            input_elements_ = elements;
            check_cuda(cudaMalloc(&input_device_, input_elements_ * sizeof(float)), "cudaMalloc input");
            bindings[input_index_] = input_device_;
            std::fprintf(stderr, "[TensorRT] input=%s elements=%zu\n", name, input_elements_);
            continue;
        }

        OutputBinding output;
        output.index = static_cast<std::size_t>(binding_index);
        output.dimensions.reserve(static_cast<std::size_t>(dimensions.nbDims));
        for (int dimension = 0; dimension < dimensions.nbDims; ++dimension) {
            output.dimensions.push_back(static_cast<std::int64_t>(dimensions.d[dimension]));
        }
        output.element_count = elements;
        output.host.resize(elements);
        check_cuda(
            cudaMalloc(&output.device, output.element_count * sizeof(float)),
            "cudaMalloc output"
        );
        bindings[output.index] = output.device;
        outputs_.push_back(std::move(output));
    }

    if (input_device_ == nullptr || outputs_.empty()) {
        throw std::runtime_error("TensorRT engine does not expose an input/output binding");
    }

    // TensorRT 8.6 exposes executeV2. The tensor addresses are valid for this context.
    if (!context_->executeV2(bindings.data())) {
        throw std::runtime_error("TensorRT warmup execution failed");
    }
#endif
}

TensorRtBackend::~TensorRtBackend() {
#ifdef VISION_ENABLE_TENSORRT
    for (OutputBinding& output : outputs_) {
        if (output.device != nullptr) {
            cudaFree(output.device);
        }
    }
    if (input_device_ != nullptr) {
        cudaFree(input_device_);
    }
    if (context_ != nullptr) {
        context_->destroy();
    }
    if (engine_ != nullptr) {
        engine_->destroy();
    }
    if (runtime_ != nullptr) {
        runtime_->destroy();
    }
#endif
}

std::vector<cv::Mat> TensorRtBackend::infer(const cv::Mat& blob) {
#ifndef VISION_ENABLE_TENSORRT
    (void)blob;
    return {};
#else
    if (blob.type() != CV_32F || blob.total() != input_elements_) {
        throw std::runtime_error("TensorRT input blob shape does not match engine input");
    }
    check_cuda(
        cudaMemcpy(
            input_device_,
            blob.ptr<float>(),
            input_elements_ * sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy input"
    );

    std::vector<void*> bindings(static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
    bindings[input_index_] = input_device_;
    for (const OutputBinding& output : outputs_) {
        bindings[output.index] = output.device;
    }
    if (!context_->executeV2(bindings.data())) {
        throw std::runtime_error("TensorRT execution failed");
    }
    for (OutputBinding& output : outputs_) {
        check_cuda(
            cudaMemcpy(
                output.host.data(),
                output.device,
                output.element_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy output"
        );
    }

    std::vector<cv::Mat> result;
    result.reserve(outputs_.size());
    for (const OutputBinding& output : outputs_) {
        std::vector<int> shape;
        for (std::int64_t dimension : output.dimensions) {
            if (dimension > 1) {
                shape.push_back(static_cast<int>(dimension));
            }
        }
        if (shape.empty()) {
            shape.push_back(static_cast<int>(output.element_count));
        }

        cv::Mat matrix;
        if (shape.size() == 1) {
            matrix = cv::Mat(shape[0], 1, CV_32F);
        } else if (shape.size() == 2) {
            matrix = cv::Mat(shape[0], shape[1], CV_32F);
        } else if (shape.size() == 3) {
            matrix = cv::Mat(shape[1], shape[2], CV_32F);
            if (shape[0] != 1) {
                matrix = cv::Mat(shape[0], shape[1] * shape[2], CV_32F);
            }
        } else {
            matrix = cv::Mat(1, static_cast<int>(output.element_count), CV_32F);
        }
        if (matrix.total() != output.element_count) {
            matrix = cv::Mat(1, static_cast<int>(output.element_count), CV_32F);
        }
        std::memcpy(matrix.ptr(), output.host.data(), output.element_count * sizeof(float));
        result.push_back(std::move(matrix));
    }
    return result;
#endif
}

Detector::Detector(
    const fs::path& weights,
    cv::Size input_size,
    float confidence_threshold,
    float iou_threshold,
    int max_detections,
    std::vector<std::string> names
)
    : weights_(weights),
      input_size_(input_size),
      confidence_threshold_(confidence_threshold),
      iou_threshold_(iou_threshold),
      max_detections_(max_detections),
      names_(std::move(names)) {
    if (!fs::exists(weights_)) {
        throw std::runtime_error("model file does not exist: " + weights_.string());
    }

    std::string extension = weights_.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (extension == ".onnx") {
        net_ = cv::dnn::readNetFromONNX(weights_.string());
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else if (extension == ".engine") {
        tensorrt_ = std::make_unique<TensorRtBackend>(weights_, names_.size());
    } else if (extension == ".pt") {
        throw std::runtime_error(
            "C++ runtime does not load PyTorch .pt checkpoints directly; export it to "
            ".onnx or .engine with the retained Python export tools"
        );
    } else {
        throw std::runtime_error("unsupported model format: " + weights_.string());
    }
}

Detector::~Detector() = default;

cv::Mat Detector::letterbox(const cv::Mat& image, LetterboxInfo& info) const {
    const double ratio = std::min(
        static_cast<double>(input_size_.width) / image.cols,
        static_cast<double>(input_size_.height) / image.rows
    );
    const int new_width = static_cast<int>(std::round(image.cols * ratio));
    const int new_height = static_cast<int>(std::round(image.rows * ratio));
    info.ratio = ratio;
    info.pad_x = (input_size_.width - new_width) / 2;
    info.pad_y = (input_size_.height - new_height) / 2;

    cv::Mat resized;
    if (image.cols != new_width || image.rows != new_height) {
        cv::resize(image, resized, cv::Size(new_width, new_height), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    const int right = input_size_.width - new_width - info.pad_x;
    const int bottom = input_size_.height - new_height - info.pad_y;
    cv::Mat output;
    cv::copyMakeBorder(
        resized,
        output,
        info.pad_y,
        bottom,
        info.pad_x,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );
    return output;
}

std::vector<Detection> Detector::predict(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    LetterboxInfo info;
    const cv::Mat letterboxed = letterbox(image, info);

    cv::Mat blob;
    cv::dnn::blobFromImage(
        letterboxed,
        blob,
        1.0 / 255.0,
        cv::Size(letterboxed.cols, letterboxed.rows),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    std::vector<cv::Mat> raw_outputs;
    if (tensorrt_) {
        raw_outputs = tensorrt_->infer(blob);
    } else {
        net_.setInput(blob);
        raw_outputs.push_back(net_.forward());
    }

    std::vector<cv::Mat> candidates;
    for (const cv::Mat& output : raw_outputs) {
        cv::Mat matrix = output_to_matrix(output);
        if (matrix.dims == 2 && matrix.rows >= 5 && matrix.cols >= 5) {
            candidates.push_back(std::move(matrix));
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("model output does not contain a YOLO detection tensor");
    }

    cv::Mat selected = candidates.front();
    if (!names_.empty()) {
        for (const cv::Mat& candidate : candidates) {
            const int low = std::min(candidate.rows, candidate.cols);
            const int high = std::max(candidate.rows, candidate.cols);
            if (low == static_cast<int>(names_.size()) + 4 ||
                low == static_cast<int>(names_.size()) + 5 ||
                high == static_cast<int>(names_.size()) + 4 ||
                high == static_cast<int>(names_.size()) + 5) {
                selected = candidate;
                break;
            }
        }
    }

    std::vector<Detection> decoded = decode(selected, info, image.size());
    return non_max_suppression(decoded, iou_threshold_, max_detections_);
}

std::vector<Detection> Detector::decode(
    const cv::Mat& output,
    const LetterboxInfo& info,
    const cv::Size& original_size
) const {
    const int rows = output.rows;
    const int cols = output.cols;
    const bool channels_first = rows < cols && rows >= 5;
    const int anchor_count = channels_first ? cols : rows;
    const int channel_count = channels_first ? rows : cols;
    if (channel_count < 5) {
        return {};
    }

    const bool has_objectness = channel_count >= static_cast<int>(names_.size()) + 5;
    const int class_count = has_objectness
                                ? channel_count - 5
                                : channel_count - 4;
    if (class_count <= 0) {
        return {};
    }

    auto value = [&](int anchor, int channel) -> float {
        return channels_first
                   ? output.at<float>(channel, anchor)
                   : output.at<float>(anchor, channel);
    };

    std::vector<Detection> detections;
    detections.reserve(static_cast<std::size_t>(anchor_count));
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        const float objectness = has_objectness ? value(anchor, 4) : 1.0F;
        int best_class = 0;
        float best_score = -1.0F;
        const int score_offset = has_objectness ? 5 : 4;
        for (int class_index = 0; class_index < class_count; ++class_index) {
            const float score = value(anchor, score_offset + class_index);
            if (score > best_score) {
                best_score = score;
                best_class = class_index;
            }
        }
        const float confidence = objectness * best_score;
        if (confidence < confidence_threshold_) {
            continue;
        }

        const float center_x = value(anchor, 0);
        const float center_y = value(anchor, 1);
        const float width = value(anchor, 2);
        const float height = value(anchor, 3);
        const float x = (center_x - 0.5F * width - static_cast<float>(info.pad_x)) /
                        static_cast<float>(info.ratio);
        const float y = (center_y - 0.5F * height - static_cast<float>(info.pad_y)) /
                        static_cast<float>(info.ratio);
        const float scaled_width = width / static_cast<float>(info.ratio);
        const float scaled_height = height / static_cast<float>(info.ratio);

        const std::string class_name =
            best_class < static_cast<int>(names_.size())
                ? names_[static_cast<std::size_t>(best_class)]
                : std::to_string(best_class);
        detections.push_back(Detection{
            class_name,
            clamp_box(
                cv::Rect2f(x, y, scaled_width, scaled_height),
                original_size
            ),
            confidence,
        });
    }
    return detections;
}

cv::Size Detector::input_size() const {
    return input_size_;
}

std::vector<Detection> non_max_suppression(
    const std::vector<Detection>& detections,
    float iou_threshold,
    int max_detections
) {
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return detections[static_cast<std::size_t>(lhs)].confidence >
               detections[static_cast<std::size_t>(rhs)].confidence;
    });

    std::vector<Detection> output;
    std::vector<bool> removed(detections.size(), false);
    for (std::size_t position = 0;
         position < order.size() && static_cast<int>(output.size()) < max_detections;
         ++position) {
        const int index = order[position];
        if (removed[static_cast<std::size_t>(index)]) {
            continue;
        }
        const Detection& candidate = detections[static_cast<std::size_t>(index)];
        output.push_back(candidate);
        for (std::size_t next = position + 1; next < order.size(); ++next) {
            const int other_index = order[next];
            if (removed[static_cast<std::size_t>(other_index)]) {
                continue;
            }
            const Detection& other = detections[static_cast<std::size_t>(other_index)];
            if (candidate.cls == other.cls &&
                intersection_over_union(candidate.box, other.box) > iou_threshold) {
                removed[static_cast<std::size_t>(other_index)] = true;
            }
        }
    }
    return output;
}

}  // namespace vision
