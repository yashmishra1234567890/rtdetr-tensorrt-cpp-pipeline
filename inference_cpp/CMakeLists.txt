cmake_minimum_required(VERSION 3.18)
project(rtdetr_infer)

set(CMAKE_CXX_STANDARD 17)

# ─── CUDA ──────────────────────────────────────────────
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ─── TensorRT ──────────────────────────────────────────
set(TRT_ROOT "D:/TensorRT/TensorRT-10.16.1.11")
include_directories(${TRT_ROOT}/include)
link_directories(${TRT_ROOT}/lib)

# ─── OpenCV ────────────────────────────────────────────
set(OpenCV_DIR "D:/opencv/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ─── Executable ────────────────────────────────────────
add_executable(rtdetr_infer main.cpp)

target_link_libraries(rtdetr_infer
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)