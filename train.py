import os
import json
import datetime
import argparse
from path import Path
from easydict import EasyDict

import basic_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='Flow/configs/raft/raft_chairs_l1.json')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-d', '--dump', action='store_true')
    parser.add_argument('-s', '--server', action='store_true')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--admm_lambda', type=float, default=None)
    parser.add_argument('--admm_rho', type=float, default=None)
    parser.add_argument('--w_admm', type=float, default=None)
    parser.add_argument('--w_smooth', type=float, default=None)
    parser.add_argument('--sm_mode', default=None)
    parser.add_argument('--sm_alpha', type=float, default=None)
    parser.add_argument('--run_fw', action='store_true')
    parser.add_argument('--diter', type=int, default=None)
    parser.add_argument('--cuda_devices', default="0,1")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    if args.evaluate:
        args.n_gpu = 1
        cfg.train.update({
            'eval_only': True,
            'epoch_num': 1,
            'epoch_size': -1,
            'valid_size': 0,
            'workers': 1,
            'val_epoch_size': 1,
            'val_shuffle': False
        })

    if args.dump:
        cfg.train.update({
            'dump_en': True,
            'val_shuffle': False
        })
    
    if args.diter is not None:
        cfg.train.update({
            'dump_iter': args.diter,
        })

    if args.split:
        cfg.data.update({ 
            "train_subsplit": "train",
            "val_subsplit": "val"
        })

    if args.run_fw:
        cfg.loss.update({'run_fw':True})
        cfg.train.update({'run_fw':True})

    if args.sm_alpha is not None:
        cfg.loss.update({'alpha':args.sm_alpha})
        if 'admm_args' in cfg.model.keys():
            cfg.model.admm_args.update({'alpha':args.sm_alpha})
    
    if args.admm_lambda is not None:
        cfg.model.admm_args.update({'lamb':args.admm_lambda})

    if args.admm_rho is not None:
        cfg.model.admm_args.update({'rho':args.admm_rho})
        cfg.loss.update({'admm_rho':args.admm_rho})

    if args.w_admm is not None:
        cfg.loss.update({'w_admm':args.w_admm})

    if args.w_smooth is not None:
        cfg.loss.update({'w_smooth':args.w_smooth})

    if args.sm_mode is not None:
        cfg.loss.update({'sm_mode':args.sm_mode})

    if args.model is not None:
        cfg.train.pretrained_model = args.model
    if args.resume:
        cfg.train.resume_run = True
    cfg.train.n_gpu = args.n_gpu
    cfg.data.n_gpu = args.n_gpu

    # store files day by day
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    if args.server:
        cfg.save_root = Path('/mnt/storage/datasets/glifshitz_user_data/CostUnrolling/outputs/flow') / curr_time[:6] / curr_time[6:]
    else:
        cfg.save_root = Path('./outputs/flow') / curr_time[:6] / curr_time[6:]
    cfg.save_root.makedirs_p()

    ## init logger
    #_log = init_logger(log_dir=cfg.save_root, filename=curr_time[6:] + '.log')
    #_log.info('=> will save everything to {}'.format(cfg.save_root))
#
    ## show configurations
    #cfg_str = pprint.pformat(cfg)
    #_log.info('=> configurations \n ' + cfg_str)

    basic_train.main(cfg)
