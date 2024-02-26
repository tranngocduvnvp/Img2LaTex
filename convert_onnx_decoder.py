from PIL import Image
import os
from typing import  Tuple
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
        arguments = Munch({'config': r'C:\Users\dutn\Documents\LatexOCR\LatexOCR\img2tex\model\settings/config.yaml', 'checkpoint': r'C:\Users\dutn\Documents\LatexOCR\LatexOCR\img2tex\model\checkpoints\checkpoint.pth', 'no_cuda': True, 'no_resize': True})
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

class Decoder_onnx(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = get_model4_onnx().decoder.net
    def forward(self, x, mask, context):
        logits = self.decoder(x, mask=mask, context=context)
        return logits

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_decoder2onnx(decoder_net, x, mask, context):
    print("================== Exporting ... ==================")
    torch.onnx.export(
        decoder_net,
        (   x,
            mask,
            context
        ),
        "decoder_onnx.onnx",
        verbose=False,
        input_names=[
            "x",
            "mask",
            "context",
        ],
        output_names=["output"],
        dynamic_axes={
            "x":{0:"batch_size", 1:"step"},
            "mask":{0:"batch_size", 1:"step"},
            "context":{0:"batch_size", 1:"length"},
            "output":{0:"batch_size", 1:"step"}
            
        }
        
    )
    print("============ Exporting successfully ===============")
    #======= Load model onnx =========
    
    onnx_model = onnx.load("decoder_onnx.onnx")
    onnx.checker.check_model(onnx_model)
    
    print("Load model successfully")

if __name__ == "__main__":
    
    
 
    decoder_net = Decoder_onnx().eval()
    
    
    #======== initialization input ==================
    dumpy = torch.rand(2,1,32,32)
    out = torch.LongTensor([1]*dumpy.shape[0])[:, None]
    mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)
    max_seq_len = 512
    
    x = out[:, -max_seq_len:]
    mask = mask[:, -max_seq_len:]
    context = torch.rand(2, 17, 256)
    
    # print("x.shape:", x.shape)
    # print("mask.shape:", mask.shape)
    # print("context.shape:", context.shape)
    
    logits = decoder_net(x, mask=mask, context = context)
    print(logits.shape)
    print("=================================================")
    # export_decoder2onnx(decoder_net, x, mask, context)

    #======= Start runtime model ==========
    ort_session = onnxruntime.InferenceSession("decoder_onnx.onnx", providers=["CPUExecutionProvider"])
    ort_inputs = {"x": to_numpy(x), "mask": to_numpy(mask), "context": to_numpy(context)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0].shape, type(ort_outs), "||", type(to_numpy(x)), type(to_numpy(mask)), type(to_numpy(context)), to_numpy(x).dtype)
    
    np.testing.assert_allclose(to_numpy(logits), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("\nExported model has been tested with ONNXRuntime, and the result looks good!")
