from PIL import Image
import os
from typing import  Tuple
from contextlib import suppress
import logging
import yaml

import torch.nn as nn
import numpy as np
import torch
from munch import Munch

import onnx
import onnxruntime

# from img2tex.dataset.latex2png import tex2pil
from img2tex.models import get_model
from img2tex.utils import *
# from img2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints
# import bentoml


def minmax_size(img, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None):
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)] #44,672 ratio: tỉ lệ gữa size ảnh thật và size max
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios) #(44,14)
            img = img.resize(size.astype(int), Image.BILINEAR)

    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)] #(w, h)
        if padded_size != list(img.size):  # assert hypothesis
            x = (padded_size[0]- img.size[0])//2
            y = (padded_size[1]- img.size[1])//2
            padded_im = Image.new('RGB', padded_size, color = 'white')
            padded_im.paste(img,(x,y))
            img = padded_im
    return img

def div32(img):
    w, h = img.size
    w = (w//32 + 1)*32 if w%32!=0 else w
    h = (h//32 + 1)*32 if h%32!=0 else h
    x = (w- img.size[0])//2
    y = (h- img.size[1])//2
    padded_im = Image.new('RGB', (w, h), color = 'white')
    padded_im.paste(img,(x,y))
    img = padded_im
    return img




# @in_model_path()        
def get_model4_onnx(arguments=None):
    if arguments is None:
        arguments = Munch({'config': r'/home/bdi/Mammo_FDA/TensorRT/LatexOCR/img2tex/model/settings/config.yaml', 'checkpoint': r'/home/bdi/Mammo_FDA/TensorRT/LatexOCR/img2tex/model/checkpoints/checkpoint.pth', 'no_cuda': True, 'no_resize': True})
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    # if not os.path.exists(args.checkpoint):
    #     download_checkpoints()
    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    # model = bentoml.pytorch.load("latexocr:latest")
    # bentoml.pytorch.save_model("latexocr", model)
    model.eval()
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def export_encoder2onnx(encoder, input_example):
    print("================== Exporting ... ==================")
    torch.onnx.export(
        encoder,
        input_example,
        "encoder_onnx.onnx",
        verbose=False,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={
            "input":{0:"batch_size", 2:"height", 3:"width"}, #[batch, 1, h, w]
            "output":{0:"batch_size", 1:"length"}
            },
        do_constant_folding=False,
        opset_version=11
        )
    print("============ Exporting successfully ===============")
    
    onnx_encoder = onnx.load_model("encoder_onnx.onnx")
    onnx.checker.check_model(onnx_encoder)
    print("Load model successfully")
    

if __name__ == "__main__":
    
    # dumpy_input = torch.rand(1,1,32,32)
    # model = Model_new()
    # out = model(dumpy_input)
    # print(out.shape)

    
    #export encoder
    model = get_model4_onnx()
    encoder = model.encoder
    
    #================== initilization dumpy input =============
    dumpy_input = torch.rand(1,1,64,32)
    out_put = encoder(dumpy_input)
    print(out_put)
    
    export_encoder2onnx(encoder, dumpy_input)
    
    #======= Start runtime model ==========
    ort_session = onnxruntime.InferenceSession("encoder_onnx.onnx", providers=["CPUExecutionProvider"])
    ort_inputs = {"input":to_numpy(dumpy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])
    
    np.testing.assert_allclose(to_numpy(out_put), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("\nExported model has been tested with ONNXRuntime, and the result looks good!")
