
"""
select config.yaml  by model
"""

from yacs.config import CfgNode as CN

def config_path(args):
    if args.model == "IRNet-2":
        path = "./experiments/IRNet_COSINE.yaml"
    elif args.model == "IRNet-1-48":
        path = "./experiments/IRNet_COSINE.yaml"
    elif args.model == "SRITM-IRNet-5":
        path = "./experiments/IRNet_COSINE.yaml"
    else:
        path = None
    
    return path
