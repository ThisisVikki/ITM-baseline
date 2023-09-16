import torch

def Optimizer(model,args,cfg=None):
    try:
        WEIGHT_DECAY = cfg.OPTIMIZER.WEIGHT_DECAY
    except AttributeError:
        WEIGHT_DECAY = 0
    
    try:
        if args.model == 'IRNet-2' or 'IRNet-1-48' or 'SRITM-IRNet-5':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE)
            
    except AttributeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer
