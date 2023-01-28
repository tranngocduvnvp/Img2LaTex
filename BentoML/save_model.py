import bentoml
from img2tex.models import get_model4serve
import torch
import logging
from munch import Munch
import yaml
from img2tex.utils import *
import os

@in_model_path()
def getmodel4serve(arguments=None):
    """Initialize a LatexOCR model

    Args:
        arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
    """
    cwd = os.getcwd()
    if arguments is None:
        arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': f'{cwd}/checkpoints/mixed_e07_step1871.pth', 'no_cuda': True, 'no_resize': True})
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
    model = get_model4serve(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    # model = bentoml.pytorch.load("latexocr:latest")
    # bentoml.pytorch.save_model("latexocr", model)
    model.eval()
    
    return model, args

model, args = getmodel4serve()

if __name__ == "__main__":
    save_model = bentoml.pytorch.save_model("LatexOCR", model, signatures={
        "__call__":{"batchable":True}
    })
    print(save_model.path)