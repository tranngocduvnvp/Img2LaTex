from typing import Tuple
import albumentations as alb
from PIL import Image
import numpy as np

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
