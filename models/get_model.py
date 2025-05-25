from .pwc_admm import PWCLite
from .raft_admm import RAFT

def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    elif cfg.type == 'smurf':
        model = RAFT(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
