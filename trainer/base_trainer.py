from pickletools import optimize
import torch
import torch.nn.functional as F
import numpy as np
import pprint
import os
from abc import abstractmethod
from logger import init_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from utils.torch_utils import bias_parameters, weight_parameters, load_checkpoint, save_checkpoint, AdamW
from utils.misc_utils import get_cfgtor, get_lr_func

#from utils.flow_utils import save_flow, np_resize_flow, flow_to_image


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_set, valid_set, model, loss_func, save_root, config_all):

        self.cfg_all = config_all
        self.cfg = config_all.train
        self.save_root = save_root
        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

        # create config functions
        self.cfgtor = get_cfgtor(config_all.train.cfg_funcs if 'cfg_funcs' in config_all.train.keys() else {})

        # set datasets
        self.train_set, self.valid_set = train_set, valid_set

        # set model and loss
        self.model = self._init_model(model)
        self.loss_func = loss_func
        
        # create optimizers and schedulers
        self.optimizer, self.scheduler = self._create_optimizer()
        
    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def train(self, rank, world_size):
        self._init_rank(rank, world_size)

        for epoch in range(self.i_epoch,self.cfg.epoch_num):
            self._run_one_epoch()
            
            if self.rank == 0 and self.i_epoch % self.cfg.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)
        
        try:
            self.cleanup()
        except Exception:
            pass
        pass

    def _init_rank(self, rank, world_size):
        self.setup(rank, world_size)
        self.world_size = world_size
        self.rank = rank
        print('DDP: Rank {} initialized'.format(rank))

        # init logger
        if self.rank == 0:
            self._log = init_logger(log_dir=self.save_root, filename=self.cfg.model + '.log')
            self._log.info('=> Rank {}: will save everything to {}'.format(self.rank, self.save_root))

            # show configurations
            cfg_str = pprint.pformat(self.cfg_all)
            self._log.info('=> configurations \n ' + cfg_str)
            self._log.info('{} training samples found'.format(len(self.train_set)))
            self._log.info('{} validation samples found'.format(len(self.valid_set)))
            if self.cfg.pretrained_model:
                self._log.info("=> using pre-trained weights {}.".format(self.cfg.pretrained_model))
            else:
                self._log.info("=> Train from scratch")
            if self.cfg.resume_run:
                    self._log.info("=> resuing run: self.i_epoch = {}, self.i_iter = {}.".format(self.i_epoch, self.i_iter))
            else:
                    self._log.info("=> new run: self.i_epoch = {}, self.i_iter = {}.".format(self.i_epoch, self.i_iter))


            self.summary_writer = SummaryWriter(str(self.save_root))
        
        self.train_loader, self.valid_loader = self._get_dataloaders(self.train_set, self.valid_set)
        self.cfg.epoch_size = min(self.cfg.epoch_size, len(self.train_loader))

        torch.cuda.set_device(self.rank)
        self.model = self.model.to(self.rank)
        if self.world_size > 1:
            if self.rank == 0:
                self._log.info(f'DDP Wrapping the model')
            self.model = DDP(self.model, device_ids=[self.rank])

        self.loss = self.loss_func.to(self.rank)
        
        if hasattr(self,'sp_transform'):
            self.sp_transform = self.sp_transform.to(self.rank)
        if hasattr(self,'seq_selfsup'):
            self.seq_selfsup = self.seq_selfsup.to(self.rank)

        #self.loss_modules = {loss_: module_.to(self.rank) for loss_, module_ in self.loss_modules.items()}
        #self.loss_module = self.loss_module.to(self.rank)
        #if hasattr(self, 'mwl_module'):
        #    self.mwl_module = self.mwl_module.to(self.rank)
        #if hasattr(self, 'cyc_module'):
        #    self.cyc_module = self.cyc_module.to(self.rank)

        pass

    def _get_dataloaders(self,train_set,valid_set):
        if self.cfg.dump_en:
            max_test_batch = 1
        else:
            max_test_batch = 16

        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	                    train_set,
                            shuffle=True,
    	                    num_replicas = self.cfg.n_gpu,
    	                    rank=self.rank)
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_set,
                            batch_size=self.cfg.batch_size_train,
                            shuffle=False,
                            num_workers=self.cfg.workers,
                            pin_memory=True,
                            sampler=train_sampler)
        #valid_loader = torch.utils.data.DataLoader(
        #                    dataset=valid_set,
        #                    batch_size=1,
        #                    shuffle=False,
        #                    num_workers=0,
        #                    pin_memory=True)
        
        #train_loader = torch.utils.data.DataLoader(
        #    train_set, batch_size=cfg.train.batch_size_train,
        #    num_workers=cfg.train.workers, pin_memory=True, shuffle=True)
        
        if type(valid_set) is torch.utils.data.ConcatDataset:
            valid_loader = [torch.utils.data.DataLoader(
                s, batch_size=min(max_test_batch, self.cfg.batch_size_eval),
                num_workers=min(4, self.cfg.workers),
                pin_memory=True, shuffle=self.cfg.val_shuffle) for s in valid_set.datasets]
            valid_size = sum([len(l) for l in valid_loader])
        else:
            valid_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=min(max_test_batch, self.cfg.batch_size_eval),
                num_workers=min(4, self.cfg.workers),
                pin_memory=True, shuffle=self.cfg.val_shuffle)
            valid_size = len(valid_loader)

        if self.cfg.epoch_size == 0:
            self.cfg.epoch_size = len(train_loader)
        if self.cfg.valid_size == 0:
            self.cfg.valid_size = valid_size
        self.cfg.epoch_size = min(self.cfg.epoch_size, len(train_loader))
        self.cfg.valid_size = min(self.cfg.valid_size, valid_size)

        return train_loader, valid_loader

    
    def _init_model(self, model):        
        if self.cfg.pretrained_model:
            print("=> using pre-trained weights {}.".format(self.cfg.pretrained_model))
            epoch, iter, weights, optimizer, scheduler = load_checkpoint(self.cfg.pretrained_model)
            if self.cfg.resume_run and epoch is not None:
                self.i_epoch = epoch
            if self.cfg.resume_run and iter is not None:
                self.i_iter = iter
            if self.cfg.resume_run and optimizer is not None:
                self.optimizer_state_dict = optimizer
            if self.cfg.resume_run and scheduler is not None:
                self.scheduler_state_dict = scheduler

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)

        else:
            print("=> Train from scratch")
            #model.apply(model.init_weights)        
        return model

    def _create_optimizer(self,):
        print('=> setting Adam solver')
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': self.cfg.bias_decay},
            {'params': weight_parameters(self.model),
             'weight_decay': self.cfg.weight_decay}]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)
        
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)
        
        if 'lr_decay_func' in self.cfg.keys():
            lr_fun = get_lr_func(self.cfg.lr_decay_func)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_fun)
            if self.scheduler_state_dict is not None:
                scheduler.load_state_dict(self.scheduler_state_dict)
        else:
            scheduler = None

        return optimizer, scheduler

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def save_model(self, error, name):
        is_best = error < self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                'iter': self.i_iter,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best)

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

