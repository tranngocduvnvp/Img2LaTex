import onnxruntime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerFast
import re
from typing import Tuple
import albumentations as alb

def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    ind = np.argpartition(logits, -k, axis=-1)[:, -k:]
    
    # Tạo mảng zeros có cùng hình dạng với logits
    probs = np.full_like(logits, -np.inf)
    
    # Gán giá trị từ logits vào probs tại các vị trí được xác định bởi ind
    row_indices = np.arange(logits.shape[0])[:, np.newaxis]
    probs[row_indices, ind] = logits[row_indices, ind]
    
    return probs

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def softmax(logits, axis=-1):
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def multinomial_sample(probs, num_samples=1):
    m, n = probs.shape
    cumulative_probs = np.cumsum(probs, axis=1)
    
    rand_values = np.random.rand(m, num_samples)
    rand_values = np.tile(rand_values, (1, n)).reshape(m, n, num_samples)

    samples = (rand_values < cumulative_probs[:, :, np.newaxis]).argmax(axis=1)
    
    return samples

def pad_numpy(mask, pad_width=(0, 1), value=True):
    padded_mask = np.pad(mask, ((0, 0), pad_width), mode='constant', constant_values=value)
    return padded_mask

def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]

def generate(image, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        num_dims = len(start_tokens.shape)
        max_seq_len = 512
        
        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out = start_tokens
        mask = kwargs.pop('mask', None)
        ort_encoder = kwargs.pop("ort_encoder_session", None)
        ort_decoder = kwargs.pop("ort_decoder_session", None)
        
        assert (ort_encoder is not None) or (ort_decoder is not None), "ort_session missing"
        if mask is None:
           mask = np.full_like(out, True, dtype=bool)

        ort_image_inputs = {"input":image}
        context = ort_encoder.run(None, ort_image_inputs)[0]
        
        for _ in range(seq_len):
            x = out[:, -max_seq_len:]
            mask = mask[:, -max_seq_len:]
            
            
            ort_decoder_inputs = {"x": x, "mask": mask, "context": context}

            # logits = net(x, mask=mask, **kwargs)[:, -1, :]
            logits = ort_decoder.run(None, ort_decoder_inputs)[0][:, -1, :] #logits numpy

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = softmax(filtered_logits / temperature, axis=-1)

            sample = multinomial_sample(probs, 1)

            # out = torch.cat((out, sample), dim=-1)
            out = np.concatenate([out, sample], axis=-1)
            mask = pad_numpy(mask, (0, 1), value=True)

            if eos_token is not None and (np.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break
    


        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        return out
    
def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s

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


def resize4test(path, scale):
    img = Image.open(path)
    w, h = img.size
    w = w*scale
    h = h*scale
    img = img.resize((round(w), round(h)))
    img = div32(minmax_size(img, (672, 192), (32, 32)))
    return img

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
    ]
)

if __name__ == "__main__":
    
    #========== Create runtime ONNX ============

    ort_encoder_session = onnxruntime.InferenceSession("/home/bdi/Mammo_FDA/TensorRT/LatexOCR/triton/tutorials/Conceptual_Guide/LatexOCR_Triton/model_repository/encoder/1/model.onnx", providers=["CPUExecutionProvider"])
    ort_decoder_session = onnxruntime.InferenceSession("/home/bdi/Mammo_FDA/TensorRT/LatexOCR/triton/tutorials/Conceptual_Guide/LatexOCR_Triton/model_repository/decoder/1/model.onnx", providers=["CPUExecutionProvider"])


    #======== initialization input ==================
    # dumpy = torch.rand(3,1,64,64)
    # dumpy = np.random.rand(3,1,64,64).astype(np.float32)
    img = resize4test(r"/home/bdi/Mammo_FDA/TensorRT/LatexOCR/triton/tutorials/Conceptual_Guide/LatexOCR_Triton/img/math1.jpg", scale=1.0)
    dumpy = np.array(img.convert('RGB'))
    dumpy = np.expand_dims(np.transpose(test_transform(image=dumpy)['image'][:,:,:1], (2,0,1)), axis=0)
    print(dumpy.shape, dumpy.dtype)
    
    start_tokens = np.ones((dumpy.shape[0], 1), dtype=np.int64)

    
    dec = generate(dumpy, start_tokens, ort_encoder_session=ort_encoder_session, ort_decoder_session=ort_decoder_session)
    print(dec.shape)
    print(type(dec))
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"C:\Users\dutn\Documents\LatexOCR\LatexOCR\img2tex\model\dataset\tokenizer.json")
    
    pred = post_process(token2str(dec, tokenizer)[0])
    print(pred)
    