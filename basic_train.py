import torch
from utils.torch_utils import init_seed
from utils.misc_utils import LinearIncrement

from datasets.get_dataset import get_dataset
from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer

import torch.multiprocessing as mp



def main(cfg):
    init_seed(cfg.seed)

    print("=> fetching img pairs.")
    train_set, valid_set = get_dataset(cfg)
    print('{} samples found, {} train samples and {} test samples '.format(
        len(valid_set) + len(train_set),
        len(train_set),
        len(valid_set)))

    model = get_model(cfg.model)
    loss = get_loss(cfg.loss)
    trainer = get_trainer(cfg.trainer)(
        train_set, valid_set, model, loss, cfg.save_root, cfg)

    # run DDP
    world_size = min(torch.cuda.device_count(),cfg.train.n_gpu)
    mp.spawn(trainer.train,
             args=(world_size,),
             nprocs=world_size,
             join=True)