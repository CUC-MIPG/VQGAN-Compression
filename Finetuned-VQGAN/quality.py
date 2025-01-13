import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_msssim import MS_SSIM
from lpips import LPIPS
from DISTS_pytorch import DISTS

# ---------- quality model ----------

class Quality_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ms_ssim = MS_SSIM(data_range=1.0)
        self.lpips = LPIPS(net="alex").eval()
        self.dists = DISTS().eval()
    
    def psnr(self, a, b):
        mse = torch.mean((a - b) ** 2)
        return -10 * torch.log10(mse)
    
    @torch.no_grad()
    def forward(self, a, b):
        psnr = self.psnr(a, b).item()
        msssim = self.ms_ssim(a, b).item()
        lpips = self.lpips(a * 2 - 1, b * 2 - 1).item()
        dists = self.dists(a, b).item()
        return {
            "psnr": psnr,
            "msssim": msssim,
            "lpips": lpips,
            "dists": dists
        }