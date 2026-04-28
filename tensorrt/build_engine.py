import tensorrt as trt
import os

# Change to onnx directory first
os.chdir(r"D:\RT-DETR-Project\onnx")

onnx_path = r"D:\RT-DETR-Project\onnx\model.onnx"
engine_path = r"D:\RT-DETR-Project\models\model_local.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

config = builder.create_builder_config()
profile = builder.create_optimization_profile()

profile.set_shape("images", (1,3,640,640), (1,3,640,640), (1,3,640,640))
profile.set_shape("orig_target_sizes", (1,2), (1,2), (1,2))
config.add_optimization_profile(profile)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 Enabled for RTX 3050!")

print("Building engine for RTX 3050... (~3-5 min)")
engine_bytes = builder.build_serialized_network(network, config)

if engine_bytes:
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print("Local engine saved:", engine_path)
else:
    print("Engine build failed!")