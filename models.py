"""
using args select model 
"""
from model.architecture import IRNet_1, IRNet_2
from model.architecture_SRITM import SRITM_IRNet_5


def Net(args, cfg=None):
    if args.model=="IRNet-2":
        model = IRNet_2(upscale=4)
    elif args.model == '"IRNet-1-48"':
        model = IRNet_1(upscale=4, nf=48)
    elif args.model == 'SRITM-IRNet-5':
        model = SRITM_IRNet_5(upscale=4)
    else:
        raise NotImplementedError("model is not available")
    return model
