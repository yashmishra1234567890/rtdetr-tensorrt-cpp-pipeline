#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string label;
};

static const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

class RTDETR {
public:
    RTDETR(const std::string& engine_path, float conf_threshold = 0.5f);
    ~RTDETR();
    std::vector<Detection> detect(const cv::Mat& image);
    cv::Mat draw(const cv::Mat& image, const std::vector<Detection>& dets);

private:
    void loadEngine(const std::string& path);
    void preprocess(const cv::Mat& image, float* input_buf, int64_t* size_buf);

    Logger logger_;
    nvinfer1::IRuntime*          runtime_  = nullptr;
    nvinfer1::ICudaEngine*       engine_   = nullptr;
    nvinfer1::IExecutionContext* context_  = nullptr;

    void* gpu_buffers_[5];
    int64_t* cpu_labels_ = nullptr;
    float*   cpu_boxes_  = nullptr;
    float*   cpu_scores_ = nullptr;
    float conf_thresh_;
};