import collections
from PIL import Image
import os

def dump_results(img, val_idx, samp_idx, res, samp_type="", path=""):
    err_str = "_".join(map('{:.2f}'.format, res))
    Image.fromarray(img).save(os.path.join(path, "Valid_{}_samp_{}_{}__epe_{}.png".format(val_idx, samp_idx, samp_type, err_str)))
    return

def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val
    return orig_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)

class Configurator(object):
    def __init__(self,cfg_fns_dict={}) -> None:
        self.cfg_fns_dict = cfg_fns_dict
        pass

    def __call__(self, cfg, iter):
        for key, _fn in self.cfg_fns_dict.items():
            cfg[key] = _fn(cfg[key],iter)
        pass

class ExponentialLRFactor(object):
    def __init__(self, *args) -> None:
        self.lr_decay_after_num_steps, self.lr_decay_steps, self.factor = args
        pass
    
    def __call__(self, iter):
        effective_step = max(iter - self.lr_decay_after_num_steps + 1, 0)
        lr_step_ratio = float(effective_step) / float(self.lr_decay_steps)
        return self.factor ** lr_step_ratio


class LinearIncrement(object):
    def __init__(self, *args) -> None:
        self.a, self.b, delta = args
        self.dx = self.b - self.a
        self.dy = delta
        pass
    
    def __call__(self, w, iter):            
        if self.a <= iter < self.b:
            #w += self.dy / self.dx
            w = (iter - self.a) / self.dx * self.dy
        elif iter > self.b:
            w = self.dy
        return w

class ToggleFlag(object):
    def __init__(self, *args) -> None:
        self.milestones = args
        pass

    def __call__(self, w, iter):
        if iter in self.milestone:
            w = not w
        return w

def get_lr_func(cfg):
    if cfg.func == 'exponential':
        return ExponentialLRFactor(*cfg.args)
    else:
        raise NotImplementedError(cfg.func)

def get_cfgtor(cfg):
    cfg_fns_dict = {}
    for p_, f_dict in cfg.items():
        for f_, args_ in zip(f_dict['funcs'], f_dict['args']):
            if f_ == 'lin_increment':
                f = LinearIncrement(*args_)
            elif f_ == 'toggle':
                f = ToggleFlag(*args_)
            else:
                raise NotImplementedError(f_)
            cfg_fns_dict.update({p_ : f})
    return Configurator(cfg_fns_dict)