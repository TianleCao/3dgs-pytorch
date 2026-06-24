import torch.nn as nn
import torch
from utils import quaternion_to_rotation_matrix, inverse_sigmoid
from sh import computeColorFromSH
from dataset import Camera
class GaussianModel(nn.Module):
    def __init__(self, N):
        """
        N: number of gaussians
        """
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(N,3))
        self.scale = nn.Parameter(torch.zeros(N,3)) # in log-space to ensure positivity
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(N,1)) # quaternions, initialized as identity rotation. shape: (N,4)
        self.opacity = nn.Parameter(inverse_sigmoid(torch.ones(N)*0.01)) # will go through sigmoid to ensure [0,1] range
        self.sh_coeff = nn.Parameter(torch.rand(N,16,3))
    
    @property
    def get_scale(self):
        return torch.exp(self.scale)
    
    @property
    def get_opacity(self):
        return torch.sigmoid(self.opacity)
    
    def get_cov_world_square_root(self):
        # (N,3,3) * (N,3,3) -> (N,3,3)
        return quaternion_to_rotation_matrix(self.rotation)@torch.diag_embed(self.get_scale)
    
    def get_cov_world(self):
        # (N,3,3) * (N,3,3) * (N,3,3) -> (N,3,3)
        return quaternion_to_rotation_matrix(self.rotation)@torch.diag_embed(self.get_scale**2)@quaternion_to_rotation_matrix(self.rotation).transpose(-1, -2)
    
    def get_mean_cam(self, w2c: torch.Tensor):
        R, t = w2c[:3, :3], w2c[:3, 3]
        # (N,3) * (3,3) + (N,3) -> (N,3)
        return self.mean@R.T + t
    
    def get_cov_cam(self, w2c: torch.Tensor):
        # (3,3) * (N,3,3) * (3,3) -> (N,3,3)
        R = w2c[:3, :3]
        return R@self.get_cov_world()@R.T
    
    def get_color(self, c2w: torch.Tensor, active_sh_deg):
        cam_center_world = c2w[:3,3]
        return computeColorFromSH(active_sh_deg, self.mean, cam_center_world, self.sh_coeff)
    
    def transform_to_2dframe(self, camera:Camera):
        fx, fy, cx, cy, w2c = camera.fx, camera.fy, camera.cx, camera.cy, camera.w2c
        mean_3d_cam = self.get_mean_cam(w2c)
        x, y, z = mean_3d_cam[:,0], mean_3d_cam[:,1], mean_3d_cam[:,2] # (N,)
        cov_3d_cam = self.get_cov_cam(w2c)
        zeros = torch.zeros_like(x)
        J = torch.stack([fx/z, zeros, -fx*x/z**2, zeros, fy/z, -fy*y/z**2], -1).reshape(-1,2,3) # [N,2,3]
        # [N,2,3] * (N, 3, 3) * (N,3,2) -> (N,2,2)
        cov_2d = J @ cov_3d_cam @ J.transpose(-1,-2)
        mean_2d = torch.stack([fx*x/z + cx, fy*y/z + cy], -1) # (N,2)
        return mean_2d, cov_2d
