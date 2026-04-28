#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "rtdetr.hpp"

// Macro to catch Out-of-Memory (OOM) or CUDA errors instantly
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA FATAL ERROR] " << #call << " failed: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)


// 1. Engine Loading

void RTDETR::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[ERROR] Cannot open engine: " << path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_  = runtime_->deserializeCudaEngine(buffer.data(), size);
    context_ = engine_->createExecutionContext();
    std::cout << "[INFO] Engine loaded successfully!" << std::endl;
}


// 2. Constructor & Destructor

RTDETR::RTDETR(const std::string& engine_path, float conf_threshold)
    : conf_thresh_(conf_threshold) {
    
    // Zero-out array to prevent garbage memory
    memset(gpu_buffers_, 0, sizeof(gpu_buffers_));
    
    loadEngine(engine_path);
    
    std::cout << "[INFO] Allocating GPU VRAM..." << std::endl;
    // Safely allocate memory.
    CHECK_CUDA(cudaMalloc(&gpu_buffers_[0], 1*3*640*640*sizeof(float)));   // images
    CHECK_CUDA(cudaMalloc(&gpu_buffers_[1], 1*2*sizeof(int64_t)));         // orig_target_sizes
    CHECK_CUDA(cudaMalloc(&gpu_buffers_[2], 1*300*sizeof(int64_t)));       // labels
    CHECK_CUDA(cudaMalloc(&gpu_buffers_[3], 1*300*4*sizeof(float)));       // boxes
    CHECK_CUDA(cudaMalloc(&gpu_buffers_[4], 1*300*sizeof(float)));         // scores

    cpu_labels_ = new int64_t[300];
    cpu_boxes_  = new float[300*4];
    cpu_scores_ = new float[300];
}

RTDETR::~RTDETR() {
    for (auto& buf : gpu_buffers_) {
        if (buf) cudaFree(buf);
    }
    delete[] cpu_labels_;
    delete[] cpu_boxes_;
    delete[] cpu_scores_;
    delete context_;
    delete engine_;
    delete runtime_;
}

// 3. Image Preprocessing
void RTDETR::preprocess(const cv::Mat& image, float* input_buf, int64_t* size_buf) {
    cv::Mat resized, rgb;
    cv::resize(image, resized, cv::Size(640, 640));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f/255.0f);
    
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++) {
        memcpy(input_buf + c*640*640, channels[c].data, 640*640*sizeof(float));
    }
    
    size_buf[0] = image.rows;
    size_buf[1] = image.cols;
}


// 4. Core Inference (TensorRT 10 Native)

std::vector<Detection> RTDETR::detect(const cv::Mat& image) {
    std::vector<float> input_data(3 * 640 * 640);
    int64_t size_data[2];
    
    preprocess(image, input_data.data(), size_data);

    // 1. Copy image to GPU
    CHECK_CUDA(cudaMemcpy(gpu_buffers_[0], input_data.data(), 3*640*640*sizeof(float), cudaMemcpyHostToDevice));

    // 2. Safely handle orig_target_sizes data type
    if (engine_->getTensorDataType("orig_target_sizes") == nvinfer1::DataType::kINT32) {
        int32_t size_data32[2] = {(int32_t)image.rows, (int32_t)image.cols};
        CHECK_CUDA(cudaMemcpy(gpu_buffers_[1], size_data32, 2*sizeof(int32_t), cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA(cudaMemcpy(gpu_buffers_[1], size_data, 2*sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    // 3. TRT 10 CRITICAL FIX: Explicitly set input shapes FIRST!
    context_->setInputShape("images", nvinfer1::Dims4{1, 3, 640, 640});
    context_->setInputShape("orig_target_sizes", nvinfer1::Dims2{1, 2});

    // 4. Set Addresses SECOND!
    bool ok = true;
    ok &= context_->setTensorAddress("images", gpu_buffers_[0]);
    ok &= context_->setTensorAddress("orig_target_sizes", gpu_buffers_[1]);
    ok &= context_->setTensorAddress("labels", gpu_buffers_[2]);
    ok &= context_->setTensorAddress("boxes", gpu_buffers_[3]);
    ok &= context_->setTensorAddress("scores", gpu_buffers_[4]);

    if (!ok) {
        std::cerr << "\n[FATAL ERROR] Engine tensor names mismatch or shapes are invalid!" << std::endl;
        std::cerr << "Engine contains the following I/O Tensors:" << std::endl;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            std::cerr << " -> " << engine_->getIOTensorName(i) << std::endl;
        }
        exit(1);
    }

    // 5. Execute using enqueueV3 
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    bool status = context_->enqueueV3(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t2 = std::chrono::high_resolution_clock::now();

    if (!status) {
        std::cerr << "[FATAL ERROR] GPU aborted execution! Check shapes/types." << std::endl;
        exit(1);
    }

    float ms = std::chrono::duration<float,std::milli>(t2-t1).count();
    std::cout << "[INFO] Inference time: " << ms << " ms" << std::endl;

    // 6. Safely handle Output labels data type
    if (engine_->getTensorDataType("labels") == nvinfer1::DataType::kINT32) {
        std::vector<int32_t> cpu_labels32(300);
        CHECK_CUDA(cudaMemcpy(cpu_labels32.data(), gpu_buffers_[2], 300*sizeof(int32_t), cudaMemcpyDeviceToHost));
        for(int i=0; i<300; i++) cpu_labels_[i] = cpu_labels32[i]; 
    } else {
        CHECK_CUDA(cudaMemcpy(cpu_labels_, gpu_buffers_[2], 300*sizeof(int64_t), cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaMemcpy(cpu_boxes_,  gpu_buffers_[3], 300*4*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(cpu_scores_, gpu_buffers_[4], 300*sizeof(float),   cudaMemcpyDeviceToHost));

    cudaStreamDestroy(stream);

    // Parse Outputs
    std::vector<Detection> results;
    for (int i = 0; i < 300; i++) {
        float score = cpu_scores_[i];
        if (score < conf_thresh_) continue;
        
        Detection d;
        d.x1 = cpu_boxes_[i*4+0];
        d.y1 = cpu_boxes_[i*4+1];
        d.x2 = cpu_boxes_[i*4+2];
        d.y2 = cpu_boxes_[i*4+3];
        d.confidence = score;
        d.class_id   = (int)cpu_labels_[i];
        
        if (d.class_id >= 0 && d.class_id < (int)COCO_CLASSES.size()) {
            d.label = COCO_CLASSES[d.class_id];
        } else {
            d.label = "unknown";
        }
        
        results.push_back(d);
    }
    return results;
}


// 5. Drawing Bounding Boxes

cv::Mat RTDETR::draw(const cv::Mat& image, const std::vector<Detection>& dets) {
    cv::Mat out = image.clone();
    for (auto& d : dets) {
        cv::rectangle(out,
            cv::Point((int)d.x1, (int)d.y1),
            cv::Point((int)d.x2, (int)d.y2),
            cv::Scalar(0, 255, 0), 2);
            
        std::string txt = d.label + " " + std::to_string((int)(d.confidence*100)) + "%";
        cv::putText(out, txt,
            cv::Point((int)d.x1, (int)d.y1 - 8),
            cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 255, 0), 2);
    }
    return out;
}

// 6. Main Execution

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: rtdetr_infer <engine> <image>" << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Loading RT-DETR TensorRT Engine..." << std::endl;
    RTDETR detector(argv[1], 0.048f); 

    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot read image!" << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Running inference..." << std::endl;
    auto detections = detector.detect(image);
    
    std::cout << "[INFO] Detections: " << detections.size() << std::endl;
    for (auto& d : detections) {
        std::cout << "  -> " << d.label
                  << " (" << (int)(d.confidence*100) << "%)"
                  << " [" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]"
                  << std::endl;
    }

    cv::Mat result = detector.draw(image, detections);
    cv::imwrite("D:/RT-DETR-Project/results/output.jpg", result);
    std::cout << "[INFO] Saved: D:/RT-DETR-Project/results/output.jpg" << std::endl;
    
    return 0;
}