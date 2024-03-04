import numpy as np
import tensorrt as trt
 
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
 
def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    # builder.max_workspace_size = 1 << 30
    # # we have only one image in batch
    # builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine, context

if __name__ == "__main__":
    engine, context = build_engine("/home/bdi/Mammo_FDA/TensorRT/LatexOCR/triton/tutorials/Conceptual_Guide/LatexOCR_Triton/model_repository/encoder/1/model.onnx")