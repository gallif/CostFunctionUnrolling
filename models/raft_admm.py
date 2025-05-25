import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .admm import ADMMSolverBlock, MaskGenerator
from .raft_core.modules.update import BasicUpdateBlock, SmallUpdateBlock
from .raft_core.modules.extractor import BasicEncoder, SmallEncoder
from .raft_core.modules.corr import CorrBlock
from .raft_core.utils.utils import bilinear_sampler, coords_grid, upflow8


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.admm_solver = args.admm_solver
        if args.admm_solver:
            self.admm_block = ADMMSolverBlock(rho=args.admm_args.rho, lamb=args.admm_args.lamb, eta=args.admm_args.eta, 
                grad=args.admm_args.grad, T=args.admm_args.T)
            self.mask_gen = MaskGenerator(alpha=args.admm_args.alpha)

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args.keys():
            args['dropout'] = 0

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args['dropout'])        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args['dropout'])
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args['dropout'])        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args['dropout'])
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self, imgs, iters=12, with_bk=False, upsample=True):
        """ Estimate optical flow between pair of frames """

        #image1 = 2 * (image1 / 255.0) - 1.0
        #image2 = 2 * (image2 / 255.0) - 1.0

        # run the feature network
        image1, image2 = 2 * imgs[:, :3] - 1.0, 2 * imgs[:, 3:] -1
        fmap1, fmap2 = self.fnet([image1, image2])
        
        if self.admm_solver:
            masks = [self.mask_gen(img, scale=1/8) for img in [image1, image2]]
        else:
            masks = [None]*2

        res_dict = {}
        res_dict['flows_fw'] = self.forward_2_frames(image1, fmap1, fmap2, masks[0], iters, upsample)        
        if with_bk:
                res_dict['flows_bw'] = self.forward_2_frames(image2, fmap2, fmap1, masks[1], iters, upsample)
        return res_dict
    
    def forward_2_frames(self, image1, fmap1, fmap2, mask, iters, upsample):
        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)

        # if dropout is being used reset mask
        self.update_block.reset_mask(net, inp)
        coords0, coords1 = self.initialize_flow(image1)

        flow_predictions = []
        if self.admm_solver:
            aux_vars = {"q":        [],
                        "c":        [],
                        "betas":    [],
                        "masks":    mask
                        }
        else:
            aux_vars = {}

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # Apply ADMM Solver
            if self.admm_solver:
                Q, C, Betas = self.admm_block(coords1 - coords0, aux_vars["masks"])
                aux_vars["q"].append(Q)
                aux_vars["c"].append(C)
                aux_vars["betas"].append(Betas)
            
            if upsample:
                flow_up = upflow8(coords1 - coords0)
                flow_predictions.append(flow_up)
            
            else:
                flow_predictions.append(coords1 - coords0)

        return flow_predictions, aux_vars




