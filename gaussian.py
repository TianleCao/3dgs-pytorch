import torch.nn as nn
import torch
from utils import quaternion_to_rotation_matrix, inverse_sigmoid
from sh import computeColorFromSH
from dataset import Camera
from scipy.spatial import KDTree
import numpy as np

class GaussianModel(nn.Module):
    def __init__(self, N):
        """
        N: number of gaussians
        """
        super().__init__()
        means = np.random.random((N,3))* 2.6 - 1.3 # from readNerfSyntheticInfo() of reference repo (dataset_readers.py)
        tree = KDTree(means)
        dists, _ = tree.query(means, k=4)
        mean_dist = np.sqrt(np.mean(dists[:,1:]**2,-1))
        self.mean = nn.Parameter(torch.tensor(means, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(np.log(mean_dist), dtype=torch.float32).unsqueeze(-1).repeat(1,3) ) # in log-space to ensure positivity
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(N,1)) # quaternions, initialized as identity rotation. shape: (N,4)
        self.opacity = nn.Parameter(inverse_sigmoid(torch.ones(N)*0.1)) # will go through sigmoid to ensure [0,1] range; based on create_from_pcd() of reference repo
        sh = np.zeros((N,16,3))
        sh[:,0,:] = np.random.random((N,3))/255.0 # match readNerfSyntheticInfo() function in reference repo, although this is almost the same all zeros init
        self.sh_coeff = nn.Parameter(torch.tensor(sh, dtype=torch.float32))
    
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
        visible = z>=0.2 # relatively far away from plane, one can see similar filters inside reference repo - cuda_rasterizer/auxilliary.h
        z_safe= torch.clamp(z, min=0.2)
        cov_3d_cam = self.get_cov_cam(w2c)
        zeros = torch.zeros_like(x)
        J = torch.stack([fx/z_safe, zeros, -fx*x/z_safe**2, zeros, fy/z_safe, -fy*y/z_safe**2], -1).reshape(-1,2,3) # [N,2,3]
        # [N,2,3] * (N, 3, 3) * (N,3,2) -> (N,2,2)
        cov_2d = J @ cov_3d_cam @ J.transpose(-1,-2)
        h_var = 0.3 # this is the EWA splatting low-pass filter, also seen in preprocessCUDA() function inside reference repo - cuda_rasterizer/forward.cu
        det_before = cov_2d[:,0,0] * cov_2d[:,1,1] - cov_2d[:,0,1]**2
        det_after = (cov_2d[:,0,0] + h_var) * (cov_2d[:,1,1] + h_var) - cov_2d[:,0,1]**2
        cov_2d_filtered = cov_2d + h_var* torch.eye(2, device = cov_2d.device, dtype=cov_2d.dtype)
        opacity_scale = torch.sqrt(torch.clamp(det_before/det_after, min=0.000025)) # anti-aliasing. Also taken from preprocessCUDA() function, for numerical stability
        mean_2d = torch.stack([fx*x/z_safe + cx, fy*y/z_safe + cy], -1) # (N,2)
        return mean_2d, cov_2d_filtered, visible, opacity_scale
