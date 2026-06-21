import torch.nn as nn
import torch
from utils import quaternion_to_rotation_matrix
from sh import computeColorFromSH
class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(3))
        self.scale = nn.Parameter(torch.ones(3))
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.opacity = nn.Parameter(torch.tensor(0.0))
        self.sh_coeff = nn.Parameter(torch.rand(16,3))

    def get_cov_world(self):
        return quaternion_to_rotation_matrix(self.rotation)@torch.diag(self.scale**2)@quaternion_to_rotation_matrix(self.rotation).T
    
    def get_mean_cam(self, w2c: torch.Tensor):
        R, t = w2c[:3, :3], w2c[:3, 3]
        return R@self.mean + t
    
    def get_cov_cam(self, w2c: torch.Tensor):
        R = w2c[:3, :3]
        return R@self.get_cov_world()@R.T
    
    def get_color(self, c2w: torch.Tensor):
        cam_center_world = c2w[:3,3]
        return computeColorFromSH(3, self.mean, cam_center_world, self.sh_coeff)
    
    def transform_to_2dframe(self, fx, fy, w2c: torch.Tensor):
        mean_3d_cam = self.get_mean_cam(w2c)
        x, y, z = mean_3d_cam
        cov_3d_cam = self.get_cov_cam(w2c)
        J = torch.tensor([[fx/z, 0, -fx*x/z**2],[0, fy/z, -fy*y/z**2]]) #[2,3]
        cov_2d = J @ cov_3d_cam @ J.T
        return mean_3d_cam[:2], cov_2d
