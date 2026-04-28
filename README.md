# RT-DETR TensorRT C++ Inference Pipeline

End-to-end real-time object detection pipeline using RT-DETR, optimized with TensorRT FP16, and deployed via high-performance C++ inference.

---

## Overview

This project implements a complete production-style AI vision pipeline:

- Dataset preparation (COCO128, YOLO → COCO format conversion)
- Model training with PyTorch (AMP mixed precision)
- COCO metric evaluation (mAP, AR)
- ONNX export from trained checkpoint
- TensorRT FP16 engine build
- C++ inference with TensorRT 10 runtime

---

## Model

| Property | Value |
|----------|-------|
| Architecture | RT-DETR (Real-Time Detection Transformer) |
| Backbone | ResNet-18 |
| Framework | PyTorch |
| Dataset | COCO128 (subset of MS COCO) |
| Input size | 640 × 640 |
| Precision | FP16 (TensorRT) |

---

## Pipeline

```
PyTorch (.pth)  →  ONNX (.onnx)  →  TensorRT (.engine)  →  C++ Inference
```

---

## Evaluation Results (COCO128)

| Metric | Value |
|--------|-------|
| AP (0.5:0.95) | 0.012 |
| AP50 | 0.026 |
| AP75 | 0.008 |
| AP small | 0.025 |
| AP medium | 0.022 |
| AP large | 0.027 |
| AR (maxDets=100) | 0.094 |

> Low mAP is expected — COCO128 (128 images) was used for pipeline validation, not accuracy maximization. The same pipeline scales to full MS COCO (~118k images).

---

## Inference Performance

| Implementation | Inference Time |
|----------------|----------------|
| PyTorch (Python) | baseline |
| TensorRT FP16 (C++) | 71.86 ms |

Tested on RTX 3050 laptop GPU.

---

## Project Structure

```
RT-DETR-Project/
├── inference_cpp/
│   ├── main.cpp          # TensorRT inference pipeline
│   ├── rtdetr.hpp        # RTDETR class definition
│   └── CMakeLists.txt    # CMake build config
├── notebook/
│   └── rtgetr.ipynb      # Training + evaluation notebook
├── docs/
│   └── RT-DETR_Documentation.pdf
├── results/              # Sample output images
└── README.md
```

---

## Requirements

### Software

| Component | Version |
|-----------|---------|
| CUDA | 11.7+ |
| cuDNN | 8.x |
| Python | 3.8+ |
| PyTorch | 1.12+ (CUDA) |
| TensorRT | 10.x |
| OpenCV | 4.x |
| CMake | 3.18+ |

### Python dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### 1. Clone RT-DETR

```bash
git clone https://github.com/lyuwenyu/RT-DETR.git
cd RT-DETR/rtdetr_pytorch
```

### 2. Train model

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --amp
```

### 3. Evaluate

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -r output/rtdetr_r18vd_6x_coco/checkpoint.pth \
  --test-only
```

### 4. Export to ONNX

```bash
python tools/export_onnx.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -r output/rtdetr_r18vd_6x_coco/checkpoint.pth \
  --check
```

### 5. Build TensorRT engine

```python
profile.set_shape("images", (1,3,640,640), (1,3,640,640), (1,3,640,640))
config.set_flag(trt.BuilderFlag.FP16)
engine_bytes = builder.build_serialized_network(network, config)
```

### 6. Build C++ inference binary

```bash
cd inference_cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 7. Run inference

```bash
./rtdetr_infer model.engine input.jpg output.jpg
```

---

## Key Technical Details

**TensorRT 10 API** — uses `enqueueV3()` and explicit tensor address binding (not the deprecated `enqueueV2` / buffer array approach).

**Dynamic type handling** — runtime detection of INT32 vs INT64 for `orig_target_sizes` and `labels` tensors, ensuring compatibility across different export configurations.

**Preprocessing** — BGR → RGB conversion, resize to 640×640, normalization to [0,1], CHW channel layout via `cv::split`.

**Post-processing** — direct bounding box output in pixel coordinates (no NMS required — RT-DETR is an end-to-end detector).

---

## Notes

- Large files (weights, ONNX, TensorRT engines) are excluded from the repo
- Training was performed on Linux; engine build and C++ inference on Windows (RTX 3050)
- Full MS COCO (~20 GB) not used due to compute constraints — pipeline is fully scalable

---

## Future Work

- Train on full MS COCO for production-level mAP
- Add INT8 quantization with calibration dataset
- Multi-image batch inference
- Triton Inference Server integration
- Adapt pipeline for custom industrial defect detection datasets

---

