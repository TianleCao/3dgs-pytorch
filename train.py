import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torchmetrics.image import StructuralSimilarityIndexMeasure
from rasterizer import Rasterizer
from gaussian import GaussianModel
from dataset import Camera


def train_step(rasterizer: Rasterizer, gaussians: GaussianModel, img: torch.Tensor, cam: Camera,
               optimizer: Optimizer, bg_color: torch.Tensor, lambda_dssim: float, active_sh_deg: int,
               grad_accum: torch.Tensor, grad_denom: torch.Tensor,
               ssim: StructuralSimilarityIndexMeasure):
    optimizer.zero_grad()
    rendered = rasterizer(gaussians, cam, bg_color, active_sh_deg) #(H,W,3)
    rendered_nchw = rendered.permute(2,0,1).unsqueeze(0) # (1,3,H,W)
    loss = F.l1_loss(img, rendered_nchw)
    if lambda_dssim > 0:
        loss = loss + lambda_dssim * (1 - ssim(img, rendered_nchw))
    loss.backward()
    viewspace_grad_norm = rasterizer.viewspace_position_grad.norm(dim=-1)
    grad_accum += viewspace_grad_norm # (N,)
    grad_denom += (viewspace_grad_norm>0).float() # (N,), anything with valid grad will contribute 1 to the denominator
    optimizer.step()
    return loss.item()
