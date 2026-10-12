# Vision C++ Runtime

This directory contains the C++ migration of the online Vision runtime. It
keeps the Python export/training tools, but the live radar process no longer
requires Python, PyTorch, PyQt5, or `pyserial`.

## Covered

- YAML configuration loading with `yaml-cpp`
- YOLO `.onnx` inference through OpenCV DNN
- optional TensorRT `.engine` inference through the TensorRT C++ API
- robot ROI batching and armor-board detection
- sliding-window and hybrid gated Kalman filters
- blind-zone candidate ranking and motion history
- RoboMaster referee protocol, CRC8/CRC16, decision packets, and gimbal packets
- Hikrobot MVS camera support, USB camera support, test video, and test images
- optional ROS2 `/Radio` and `/RadioFeedback` bridge
- information UI, map rendering, image saving, and FFmpeg recording
- C++ calibration and map-mask tools

The following Python files intentionally remain because they are offline
training/export utilities rather than runtime code:

- `export.py`
- `onnx2engine.py`
- `models/*.py`
- `utils/**`

The runtime consumes `.onnx` or `.engine` files produced by those tools.

## Build

Ubuntu 22.04 example with system packages:

```bash
sudo apt install build-essential cmake libopencv-dev libyaml-cpp-dev
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

If OpenCV and yaml-cpp are installed in a Conda prefix:

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build cpp/build -j
```

Optional features:

```bash
# TensorRT engine inference
cmake -S cpp -B cpp/build \
  -DVISION_ENABLE_TENSORRT=ON \
  -DTENSORRT_INCLUDE_DIR=/path/to/tensorrt/include \
  -DTENSORRT_LIBRARY=/path/to/libnvinfer.so \
  -DCUDA_RUNTIME_LIBRARY=/path/to/libcudart.so

# ROS2 Humble /radio bridge
source /opt/ros/humble/setup.bash
cmake -S cpp -B cpp/build -DVISION_ENABLE_ROS2=ON
```

MVS support is enabled automatically when `MvCameraControl.h` and
`libMvCameraControl.so` are found. Set `MVCAM_SDK_PATH` when the SDK is installed
in a non-standard prefix.

## Run

The binaries use the existing root `config.yaml`, images, calibration files,
and models. Run them from the repository root:

```bash
./cpp/build/vision
./cpp/build/vision_calibrate
./cpp/build/vision_make_mask
```

`vision_calibrate` controls:

- `s`: freeze the current camera frame and start selecting points
- `h`: switch between ground and central-highland layers
- `c`: validate and save both homographies
- `q` / Esc: exit

`vision_make_mask` controls:

- drag with the left mouse button to draw a region
- `n`: switch green/blue mask mode
- `q`: save and exit

## Test

```bash
cmake -S cpp -B cpp/build -DVISION_BUILD_TESTS=ON
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

The tests verify CRC compatibility, packet framing, game-status parsing,
filter behavior, class-aware NMS, and NumPy calibration loading.
