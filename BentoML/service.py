from img2tex.resizeimg4test import  resize4api
import bentoml
from img2tex.dataset.transforms import test_transform
import numpy as np
import logging
from munch import Munch
from img2tex.utils import *
import yaml
from transformers import PreTrainedTokenizerFast

import os
cwd = os.getcwd()
# print(cwd)

@in_model_path()
def getargs(arguments=None):
    """Initialize a LatexOCR model

    Args:
        arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
    """
    # cwd = os.getcwd()
    # print(cwd)
    if arguments is None:
        arguments = Munch({'config': 'settings/config.yaml', 'no_cuda': True, 'no_resize': True})
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    return args

args = getargs()

tokenizer = PreTrainedTokenizerFast(tokenizer_file= cwd + "/img2tex/model/" + args.tokenizer)
runner = bentoml.pytorch.get("latexocr:gmuh2vun3omjtigt").to_runner()
svc = bentoml.Service(name="LatextOCR", runners=[runner])

@svc.api(input=bentoml.io.Image(), output=bentoml.io.Text())
def inference(image):
    img = resize4api(image, scale=1)
    img = np.array(img.convert('RGB'))
    t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to(args.device)
    math = runner.run(im, args.get('temperature', .25))
    pred = post_process(token2str(math, tokenizer)[0])
    return pred
