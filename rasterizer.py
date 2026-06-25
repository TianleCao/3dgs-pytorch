from dataset import Camera
import torch
import torch.nn as nn
from gaussian import GaussianModel

class Rasterizer(nn.Module):
    def __init__(self):
        super().__init__()

    # def render_single_gaussian(self, gaussian_center_2d: torch.Tensor, gaussian_cov_2d: torch.Tensor, opacity: float, color: torch.Tensor, pixel_locations: torch.Tensor, T: torch.tensor):
    #     """
    #     render a single gaussian
    #     Arugments:
    #         pixel_location: [H,W,2]
    #         T: accumulated opacity [H,W]
    #     returns:
    #         color: image after rendering the gaussian[H,W,3]
    #         T: updated opacity [H,W]
    #     """
    #     center_dist = pixel_locations - gaussian_center_2d # [H,W,2] - [2] -> [H,W,2]
    #     a, b, c = gaussian_cov_2d[0,0], gaussian_cov_2d[0,1], gaussian_cov_2d[1,1] # scalars
    #     det = a*c-b*b
    #     xv, yv = center_dist[...,0], center_dist[...,1] # [H,W]
    #     power = -0.5/det*(c*xv**2 - 2*b*xv*yv + a*yv**2) # [H,W]
    #     power = torch.clamp(power, max=0) # numerical precision
    #     alpha = torch.clamp(torch.exp(power)* opacity, max=0.99) # [H,W], clamp alpha with 0.99, as in Appenedix C of  3DGS paper
    #     alpha = alpha * (alpha>=1/255)  # filter out negligible contribution(<1/255), as in Appenedix C of  3DGS paper
    #     T_new = T*(1-alpha) # [H,W]
    #     color_new = color * alpha.unsqueeze(-1) * T.unsqueeze(-1) # [H,W,3]
    #     return color_new, T_new
    
    def forward(self, gaussians: GaussianModel, camera: Camera, bg_color: torch.Tensor, active_sh_deg:int):
        device = gaussians.mean.device
        h, w = camera.height, camera.width

        depth = gaussians.get_mean_cam(camera.w2c)[:,2] # [N]
        sort_ind = torch.argsort(depth)

        xv, yv = torch.meshgrid(torch.arange(w,device=device), torch.arange(h,device=device), indexing = 'xy')
        gaussians_center_2d, gaussians_cov_2d, visible = gaussians.transform_to_2dframe(camera) # (N,2), (N,2,2)
        # we need to keep the grad of nonleaf center (i.e. "view-space position gradients" in 5.2 of paper), to assess under-reconstruction and over-reconstruction
        gaussians_center_2d.retain_grad()
        self.last_means_2d = gaussians_center_2d # expose to training loop
        sort_ind = sort_ind[visible[sort_ind]]## skips gaussian too close to the detector frame

        opacities = gaussians.get_opacity
        colors = gaussians.get_color(camera.c2w, active_sh_deg) # (N,3)
        x_pixel_dist_to_centers = xv - gaussians_center_2d[sort_ind,None,None,0]# (N,1,1) - (H,W) -> (N,H,W)
        y_pixel_dist_to_centers = yv - gaussians_center_2d[sort_ind,None,None,1]# (N,1,1) - (H,W) -> (N,H,W)
        a, b, c = gaussians_cov_2d[sort_ind,0:1,0:1], gaussians_cov_2d[sort_ind,0:1,1:2], gaussians_cov_2d[sort_ind,1:2,1:2] # (N,1,1)
        det = a*c-b*b # (N,1,1)
        power = -0.5/det*(c*x_pixel_dist_to_centers**2 -2*b*x_pixel_dist_to_centers*y_pixel_dist_to_centers + a*y_pixel_dist_to_centers**2) # (N,1,1) * (N,H,W) -> (N,H,W)
        power = torch.clamp(power, max=0)
        alpha = torch.clamp(torch.exp(power)*opacities[sort_ind,None,None], max=0.99) # (N,H,W)
        alpha = alpha * (alpha>=1/255)
        T = torch.cumprod(1-torch.cat((torch.zeros(1,h,w,device=device),alpha),0),0) # (N+1,H,W)
        image = torch.sum(T[:-1,...,None]*alpha[...,None]*colors[sort_ind,None,None,:],0) #(N,H,W,1) * (N,1,1,3) -> (H,W,3)
        image += T[-1,...,None] * bg_color # fill in remaining transmittance with background color

        return image
    
    @property
    def viewspace_position_grad(self):
        return self.last_means_2d.grad
