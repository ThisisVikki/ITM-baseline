import torch
 
def Scheduler(optimizer,args,cfg=None):
    try:
        if args.model == 'IRNet-2' or 'IRNet-1-48' or 'SRITM-IRNet-5':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_mult=cfg.SCHEDULER.T_MULT, T_0=cfg.SCHEDULER.T_0, eta_min=cfg.SCHEDULER.eta_min)

    except AttributeError:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1)
    return scheduler
