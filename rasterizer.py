from dataset import Camera
import torch
import torch.nn as nn
from gaussian import GaussianModel

class Rasterizer(nn.Module):
    def __init__(self):
        super().__init__()

    def render_single_gaussian(self, gaussian: GaussianModel, pixel_locations: torch.Tensor, T: torch.tensor, camera: Camera):
        """
        render a single gaussian
        Arugments:
            pixel_location: [H,W,2]
            T: accumulated opacity [H,W]
        returns:
            color: image after rendering the gaussian[H,W,3]
            T: updated opacity [H,W]
        """
        gaussian_center_2d, gaussian_cov_2d = gaussian.transform_to_2dframe(camera.fx, camera.fy, camera.w2c)
        center_dist = pixel_locations - gaussian_center_2d # [H,W,2] - [2] -> [H,W,2]
        a, b, c = gaussian_cov_2d[0,0], gaussian_cov_2d[0,1], gaussian_cov_2d[1,1] # scalars
        det = a*c-b*b
        xv, yv = center_dist[...,0], center_dist[...,1] # [H,W]
        power = -0.5/det*(c*xv**2 - 2*b*xv*yv + a*yv**2) # [H,W]
        power[power>0] = 0 # numerical precision
        alpha = torch.clamp(torch.exp(power)* torch.sigmoid(gaussian.opacity), max=0.99) # [H,W], clamp alpha with 0.99, as in Appenedix C of  3DGS paper
        alpha[alpha<1/255] = 0 # filter out negligible contribution, as in Appenedix C of  3DGS paper
        T_new = T*(1-alpha) # [H,W]
        color = gaussian.get_color(camera.c2w) # [3]
        color = color * alpha.unsqueeze(-1) * T.unsqueeze(-1) # [H,W,3]
        return color, T_new
    
    def forward(self, gaussian_list: list[GaussianModel], camera: Camera, bg_color: torch.Tensor):
        h, w = camera.height, camera.width
        image = torch.zeros((h,w,3))
        T = torch.ones((h,w)) # remaining transmittance, initially at 1

        depth = torch.stack([g.get_mean_cam(camera.w2c)[-1] for g in gaussian_list])
        sort_ind = torch.argsort(depth)

        xv, yv = torch.meshgrid(torch.arange(w), torch.arange(h), indexing = 'xy')
        pixel_locations = torch.stack((xv,yv),-1)
        for ind in sort_ind:
            if torch.max(T) < 1/255:
                ## cheap global early stopping
                break
            color, T = self.render_single_gaussian(gaussian_list[ind], pixel_locations, T, camera)
            image += color
        image += T.unsqueeze(-1)*bg_color # fill in remaining transmittance with background color
        return torch.clip(image, min=0, max=1)
    