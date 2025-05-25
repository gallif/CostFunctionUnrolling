from .flow_loss import unFlowLoss, unSequenceLoss

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'sequence':
        loss = unSequenceLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
