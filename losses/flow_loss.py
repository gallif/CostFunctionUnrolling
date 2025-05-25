import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward
import matplotlib.pyplot as plt

def save_fig_for_debug(im,name):
    plt.imshow(im.squeeze().cpu().detach().numpy().transpose(1,2,0))
    plt.savefig(name)
    plt.close()
    return


class SelfSequenceLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(SelfSequenceLoss, self).__init__()
        self.cfg = cfg
        
    def forward(self, output, target, noc):
        n_predictions = len(output)
        i_weights = [self.cfg.gamma**(n_predictions - i - 1) for i in range(n_predictions)]
        i_losses = [((flow_pred - target).abs() + self.cfg.ar_eps) ** self.cfg.ar_q for flow_pred in output] 
        i_losses = [(loss_ * noc).mean() / (noc.mean() + 1e-7) for loss_ in i_losses]
        return sum([w * l for w,l in zip(i_weights, i_losses)])


class unSequenceLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unSequenceLoss, self).__init__()
        self.cfg = cfg

    def loss_admm(self, Q, C, Betas, w_admm):
        loss = []

        if w_admm > 0:
            T = len(Q)
            loss += [(q - c + beta)**2 / T for q,c,beta in zip(Q,C,Betas)]
        l_admm = w_admm * self.cfg.admm_rho / 2 * sum(loss)
        #l_admm.register_hook(print_hook)
        return l_admm.mean()

    def loss_photometric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha, self.cfg.sm_mode)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target, aux, pos=None):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        sequence_flows = output
        n_predictions = len(sequence_flows)
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]
        aux_12 = aux[0]
        aux_21 = aux[1]

        sequence_admm_losses = []
        sequence_smooth_losses = []
        sequence_warp_losses = []
        sequence_weights = []

        b, _, h, w = sequence_flows[-1].size()
        B, _, H, W = target.size()

        # resize images to match the size of flow
        if self.cfg.run_fw:
            im1_cropped = im1_origin[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]
            im2_cropped = im2_origin[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]
            im1_scaled = F.interpolate(im1_cropped, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_cropped, (h, w), mode='area')
        else:            
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

        # occlusion mask estimation using last iteration
        output_flow = sequence_flows[-1]

        if self.cfg.occ_from_back:
            occu_mask1 = 1 - get_occu_mask_backward(output_flow[:, 2:], th=0.2)
            occu_mask2 = 1 - get_occu_mask_backward(output_flow[:, :2], th=0.2)
            
            if self.cfg.run_fw:
                output_flow_full = F.pad(output_flow,[pos[1], W - w - pos[1],pos[0], H - h - pos[0]])
                occu_mask1_full = 1 - get_occu_mask_backward(output_flow_full[:, 2:], th=0.2)
                occu_mask2_full = 1 - get_occu_mask_backward(output_flow_full[:, :2], th=0.2)
                occu_mask1_fw = occu_mask1_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]
                occu_mask2_fw = occu_mask2_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]

        else:
            occu_mask1 = 1 - get_occu_mask_bidirection(output_flow[:, :2], flow[:, 2:])
            occu_mask2 = 1 - get_occu_mask_bidirection(output_flow[:, 2:], flow[:, :2])

            if self.cfg.run_fw:
                occu_mask1_full = 1 - get_occu_mask_bidirection(output_flow_full[:, 2:], th=0.2)
                occu_mask2_full = 1 - get_occu_mask_bidirection(output_flow_full[:, :2], th=0.2)
                occu_mask1_fw = occu_mask1_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]
                occu_mask2_fw = occu_mask2_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]

        s = 1.
        for i, flow in enumerate(sequence_flows):
            sequence_weights.append(self.cfg.gamma**(n_predictions - i - 1))
            
            if self.cfg.run_fw:
                # warp using full size images
                flow_full = F.pad(flow,[pos[1], W - w - pos[1], pos[0], H - h - pos[0]])
                im1_recons_full = flow_warp(im2_origin, flow_full[:, :2], pad=self.cfg.warp_pad)
                im2_recons_full = flow_warp(im1_origin, flow_full[:, 2:], pad=self.cfg.warp_pad)

                # crop                    
                im1_recons_fw = im1_recons_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]
                im2_recons_fw = im2_recons_full[:, :, pos[0]:pos[0]+h, pos[1]:pos[1]+w]

                loss_warp = self.loss_photometric(im1_scaled, im1_recons_fw, occu_mask1_fw.to(im1_recons_fw.device))

            else:
                im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
                im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)
                
                loss_warp = self.loss_photometric(im1_scaled, im1_recons, occu_mask1.to(im1_recons.device))

            if i == 0:
                s = min(h, w)

            if self.cfg.w_smooth > 0:
                loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)
            else:
                loss_smooth = torch.zeros(1, device=loss_warp.device)

            if self.cfg.w_admm > 0:
                loss_admm = self.loss_admm(aux_12["q"][i], aux_12["c"][i], aux_12["betas"][i], self.cfg.w_admm)
            else:
                loss_admm = torch.zeros(1, device=loss_warp.device)

            if self.cfg.with_bk:
                if self.cfg.run_fw:
                    loss_warp += self.loss_photometric(im2_scaled, im2_recons_fw, occu_mask2_fw.to(im2_recons_fw.device))
                else:
                    loss_warp += self.loss_photometric(im2_scaled, im2_recons, occu_mask2.to(im2_recons.device))
                
                if self.cfg.w_smooth > 0:
                    loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                if self.cfg.w_admm > 0:
                    loss_admm += self.loss_admm(aux_21["q"][i], aux_21["c"][i], aux_21["betas"][i], self.cfg.w_admm)

                loss_warp /= 2.
                loss_smooth /= 2.
                loss_admm /= 2.
            

            sequence_warp_losses.append(loss_warp)
            sequence_smooth_losses.append(loss_smooth)
            sequence_admm_losses.append(loss_admm)

        sequence_warp_losses = [l * w for l, w in zip(sequence_warp_losses, sequence_weights)]
        sequence_smooth_losses = [l * w for l, w in zip(sequence_smooth_losses, sequence_weights)]
        sequence_admm_losses = [l * w for l, w in zip(sequence_admm_losses, sequence_weights)]

        warp_loss = sum(sequence_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(sequence_smooth_losses)
        admm_loss = sum(sequence_admm_losses)
        total_loss = warp_loss + smooth_loss + admm_loss

        return total_loss, warp_loss, smooth_loss, admm_loss, sequence_flows[0].abs().mean(), occu_mask1

class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_admm(self, Q, C, Betas, w_admm):
        loss = []

        if w_admm > 0:
            T = len(Q)
            loss += [(q - c + beta)**2 / T for q,c,beta in zip(Q,C,Betas)]
        l_admm = w_admm * self.cfg.admm_rho / 2 * sum(loss)
        #l_admm.register_hook(print_hook)
        return l_admm.mean()

    def loss_photometric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha, self.cfg.sm_mode)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target, aux):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]
        aux_12 = aux[0]
        aux_21 = aux[1]

        pyramid_admm_losses = []
        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        pyramid_occu_mask1 = []
        pyramid_occu_mask2 = []


        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])

                pyramid_occu_mask1 = occu_mask1
                pyramid_occu_mask2 = occu_mask2
            else:
                occu_mask1 = F.interpolate(pyramid_occu_mask1, (h, w), mode='nearest')
                occu_mask2 = F.interpolate(pyramid_occu_mask2, (h, w), mode='nearest')

            loss_warp = self.loss_photometric(im1_scaled, im1_recons, occu_mask1.to(im1_recons.device))

            if i == 0:
                s = min(h, w)

            if self.cfg.w_smooth > 0:
                loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)
            else:
                loss_smooth = torch.zeros(1, device=loss_warp.device)

            if self.cfg.w_admm[i] > 0:
                loss_admm = self.loss_admm(aux_12["q"][i], aux_12["c"][i], aux_12["betas"][i], self.cfg.w_admm[i])
            else:
                loss_admm = torch.zeros(1, device=loss_warp.device)

            if self.cfg.with_bk:
                loss_warp += self.loss_photometric(im2_scaled, im2_recons, occu_mask2.to(im2_recons.device))
                
                if self.cfg.w_smooth > 0:
                    loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                if self.cfg.w_admm[i] > 0:
                    loss_admm += self.loss_admm(aux_21["q"][i], aux_21["c"][i], aux_21["betas"][i], self.cfg.w_admm[i])

                loss_warp /= 2.
                loss_smooth /= 2.
                loss_admm /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)
            pyramid_admm_losses.append(loss_admm)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        admm_loss = sum(pyramid_admm_losses)
        total_loss = warp_loss + smooth_loss + admm_loss

        return total_loss, warp_loss, smooth_loss, admm_loss, pyramid_flows[0].abs().mean(), pyramid_occu_mask1