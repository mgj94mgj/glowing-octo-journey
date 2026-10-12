cmake_minimum_required(VERSION 3.16)

project(VisionCpp VERSION 1.0.0 LANGUAGES CXX)

option(VISION_BUILD_TESTS "Build C++ unit tests" ON)
option(VISION_ENABLE_TENSORRT "Enable TensorRT .engine inference" OFF)
option(VISION_ENABLE_MVS "Enable Hikrobot MVS camera support when the SDK is installed" ON)
option(VISION_ENABLE_ROS2 "Enable ROS2 radio bridge nodes" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio video dnn highgui)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

add_library(vision_core
    src/camera.cpp
    src/config.cpp
    src/detector.cpp
    src/filters.cpp
    src/geometry.cpp
    src/protocol.cpp
    src/radio.cpp
    src/recorder.cpp
    src/serial.cpp
    src/ui.cpp
)

target_include_directories(vision_core PUBLIC include)
target_link_libraries(vision_core PUBLIC
    ${OpenCV_LIBS}
    $<IF:$<TARGET_EXISTS:yaml-cpp::yaml-cpp>,yaml-cpp::yaml-cpp,yaml-cpp>
    Threads::Threads
)

# Conda's OpenCV packages are built with a newer libstdc++ than Ubuntu 22.04.
# Link the matching runtime when OpenCV is resolved from a Conda prefix.
if(OpenCV_DIR MATCHES "conda")
    get_filename_component(_opencv_prefix "${OpenCV_DIR}/../../.." ABSOLUTE)
    find_library(_opencv_stdcxx NAMES stdc++ HINTS "${_opencv_prefix}/lib" NO_DEFAULT_PATH)
    if(_opencv_stdcxx)
        target_link_libraries(vision_core PUBLIC "${_opencv_stdcxx}")
    endif()
endif()

target_compile_definitions(vision_core PUBLIC
    $<$<BOOL:${VISION_ENABLE_TENSORRT}>:VISION_ENABLE_TENSORRT=1>
    $<$<BOOL:${VISION_ENABLE_MVS}>:VISION_ENABLE_MVS=1>
    $<$<BOOL:${VISION_ENABLE_ROS2}>:VISION_ENABLE_ROS2=1>
)

if(VISION_ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
    find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h)
    find_library(TENSORRT_LIBRARY nvinfer)
    find_library(CUDA_RUNTIME_LIBRARY cudart)
    if(NOT TENSORRT_INCLUDE_DIR OR NOT CUDA_INCLUDE_DIR OR
       NOT TENSORRT_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR "TensorRT/CUDA development files were not found")
    endif()
    target_include_directories(vision_core PRIVATE
        "${TENSORRT_INCLUDE_DIR}"
        "${CUDA_INCLUDE_DIR}"
    )
    target_link_libraries(vision_core PRIVATE "${TENSORRT_LIBRARY}" "${CUDA_RUNTIME_LIBRARY}")
endif()

if(VISION_ENABLE_MVS)
    find_path(MVS_INCLUDE_DIR MvCameraControl.h
        HINTS "$ENV{MVCAM_SDK_PATH}/include" "/opt/MVS/include" "$ENV{HOME}/.local/opt/MVS/include"
    )
    find_library(MVS_LIBRARY NAMES MvCameraControl
        HINTS "$ENV{MVCAM_SDK_PATH}/lib/64" "/opt/MVS/lib/64" "$ENV{HOME}/.local/opt/MVS/lib/64"
    )
    if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
        target_include_directories(vision_core PRIVATE "${MVS_INCLUDE_DIR}")
        target_link_libraries(vision_core PRIVATE "${MVS_LIBRARY}")
    else()
        target_compile_definitions(vision_core PUBLIC VISION_ENABLE_MVS=0)
        message(STATUS "MVS SDK not found; hik camera mode will report a clear error")
    endif()
endif()

if(VISION_ENABLE_ROS2)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    target_link_libraries(vision_core PRIVATE rclcpp::rclcpp ${std_msgs_TARGETS})
endif()

add_executable(vision src/main.cpp)
target_link_libraries(vision PRIVATE vision_core)

add_executable(vision_calibrate src/calibration.cpp)
target_link_libraries(vision_calibrate PRIVATE vision_core)

add_executable(vision_make_mask src/make_mask.cpp)
target_link_libraries(vision_make_mask PRIVATE vision_core)

if(VISION_BUILD_TESTS)
    enable_testing()
    add_executable(vision_tests tests/tests.cpp)
    target_link_libraries(vision_tests PRIVATE vision_core)
    target_compile_definitions(vision_tests PRIVATE
        VISION_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    add_test(NAME vision_tests COMMAND vision_tests)
endif()
