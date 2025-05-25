import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.flow_utils import dump_flow

class ADMMSolverBlock(nn.Module):
    def __init__(self,rho,lamb,eta,grad="1st",T=1):
        super(ADMMSolverBlock, self).__init__()
        # params
        self.T = T
        self.grad = grad
        # variables
        self.beta = None
        self.Q = None
        self.count = 0
        # blocks
        self.get_gradients = Sobel()
        self.apply_threshold = SoftThresholding(rho,lamb)
        self.update_multipliers = MultiplierUpdate(eta)

    def forward(self, F, masks):
        # get masked grads
        dF = self.get_gradients(F) #[dF/dx, dF/dy]
        if self.grad == "2nd":
            dF2 = [self.get_gradients(df) for df in dF] #[[dF/dxx, dF/dxy], [dF/dyx, dF/dyy]]
            dF = [dF2[0][0], dF2[1][1]] #[dF/dxx, dF/dyy]
        c = [df * m for df, m in zip(dF, masks)]
        c = torch.cat(c, dim = 1) #[B,4,H,W]
        # initialize 
        beta = torch.zeros_like(c)
        q = torch.zeros_like(c)
        
        Q = [q]
        C = [c]
        Betas = [beta]

        # update q and beta
        for t in range(self.T):
            q = self.apply_threshold(c,beta)
            beta = self.update_multipliers(q,c,beta)
            Q.append(q)
            C.append(c)
            Betas.append(beta)

        self.count += 1
        return Q, C, Betas
    
class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb):
        super(SoftThresholding, self).__init__()
        self.lamb = lamb
        self.rho = rho
    
    def forward(self,C, beta):
        th = self.lamb / self.rho

        mask = (C - beta).abs() >= th
        Q = (C - beta - th * torch.sign(C - beta)) * mask
        
        return Q

class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, C, beta):
        beta = beta + self.eta * (Q - C)
        
        return beta

class MaskGenerator(nn.Module):
    def __init__(self,alpha):
        super(MaskGenerator,self).__init__()
        self.alpha = alpha
        self.sobel = Sobel()
    
    def rgb2gray(self, im_rgb):
        im_gray = (im_rgb * torch.tensor([[[[0.2989]],[[0.5870]],[[0.1140]]]], dtype = torch.float, device = im_rgb.device)).sum(dim=1,keepdim=True)
        return im_gray

    def forward(self, image, scale=1/4):
        image = F.interpolate(image, scale_factor=scale, mode='area')
        im_grads = self.sobel(image) #[dx, dy]
        masks = [torch.exp(-torch.mean(torch.abs(grad), 1, keepdim=True) * self.alpha) for grad in im_grads]
        return masks

class Sobel(nn.Module):
    def __init__(self,  f_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        f_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]):
        super(Sobel, self).__init__()
        Dx = torch.tensor(f_x, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        Dy = torch.tensor(f_y, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        self.D = nn.Parameter(torch.cat((Dx, Dy), dim = 0), requires_grad = False)
    
    def forward(self,image):
        # apply filter over each channel seperately
        im_ch = torch.split(image, 1, dim = 1)
        grad_ch = [F.conv2d(ch, self.D, padding = 1) for ch in im_ch]
        dx = torch.cat([g[:,0:1,:,:] for g in grad_ch], dim=1)
        dy = torch.cat([g[:,1:2,:,:] for g in grad_ch], dim=1)

        return [dx, dy]