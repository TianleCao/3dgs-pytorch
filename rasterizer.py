from dataset import Camera
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from gaussian import GaussianModel


class Rasterizer(nn.Module):
    # Empirically each Gaussian in a chunk holds ~20 (H, W) float32 tensors alive
    # during forward+backward of one checkpointed chunk (alpha, T_before, T_after,
    # weight, contrib, x_dist, y_dist, power, plus autograd duplicates).
    _BYTES_PER_GAUSSIAN_PIXEL = 20 * 4

    def __init__(self, chunk_size: int | None = None, memory_fraction: float = 0.75,
                 min_chunk: int = 50, max_chunk: int = 5000):
        """
        chunk_size: explicit chunk size, or None to auto-size from free VRAM on first call.
        memory_fraction: fraction of free GPU memory to budget for one chunk.
        min_chunk / max_chunk: clamp the auto-sized chunk to a sane range.
        """
        super().__init__()
        self._explicit_chunk_size = chunk_size
        self.memory_fraction = memory_fraction
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self._auto_chunk_size: int | None = None

    def _resolve_chunk_size(self, h: int, w: int, device: torch.device) -> int:
        if self._explicit_chunk_size is not None:
            return self._explicit_chunk_size
        if self._auto_chunk_size is not None:
            return self._auto_chunk_size

        bytes_per_gaussian = self._BYTES_PER_GAUSSIAN_PIXEL * h * w
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = int(free_bytes * self.memory_fraction)
        else:
            budget = 2 * 1024**3  # 2 GB on CPU
        K = max(self.min_chunk, min(self.max_chunk, budget // bytes_per_gaussian))
        self._auto_chunk_size = K
        print(f"[Rasterizer] auto chunk_size = {K} for {h}x{w} "
              f"(budget {budget/1e9:.2f} GB, free {free_bytes/1e9:.2f} GB)"
              if device.type == "cuda" else
              f"[Rasterizer] auto chunk_size = {K} for {h}x{w} (CPU)")
        return K

    def _render_chunk(self, means_2d, cov_2d, opacities, colors, xv, yv, T_global, h, w):
        """Vectorized rendering of one chunk of Gaussians, composited onto T_global."""
        # alpha computation, same math as before but on K Gaussians
        x_dist = xv - means_2d[:, None, None, 0]                                   # (K, H, W)
        y_dist = yv - means_2d[:, None, None, 1]                                   # (K, H, W)
        a = cov_2d[:, 0:1, 0:1]                                                    # (K, 1, 1)
        b = cov_2d[:, 0:1, 1:2]
        c = cov_2d[:, 1:2, 1:2]
        det = a * c - b * b
        power = -0.5 / det * (c * x_dist**2 - 2 * b * x_dist * y_dist + a * y_dist**2)
        power = power.clamp(max=0)
        alpha = (torch.exp(power) * opacities[:, None, None]).clamp(max=0.99)      # (K, H, W), clamp alpha with 0.99, as in Appenedix C of  3DGS paper
        alpha = alpha * (alpha >= 1/255) # filter out negligible contribution(<1/255), as in Appenedix C of  3DGS paper

        # local cumprod gives T relative to the start of this chunk;
        # multiply by T_global to get transmittance from camera up to each Gaussian.
        T_chunk_after = torch.cumprod(1 - alpha, dim=0)                            # (K, H, W)
        T_chunk_before = torch.cat(
            [torch.ones(1, h, w, device=alpha.device), T_chunk_after[:-1]], dim=0
        )                                                                          # (K, H, W)
        weight = alpha * T_chunk_before * T_global.unsqueeze(0)                    # (K, H, W)
        contrib = (weight.unsqueeze(-1) * colors[:, None, None, :]).sum(dim=0)     # (H, W, 3)

        T_global_new = T_global * T_chunk_after[-1]                                # (H, W)
        return contrib, T_global_new

    def forward(self, gaussians: GaussianModel, camera: Camera,
                bg_color: torch.Tensor, active_sh_deg: int):
        device = gaussians.mean.device
        h, w = camera.height, camera.width

        depth = gaussians.get_mean_cam(camera.w2c)[:, 2] # (N,)
        sort_ind = torch.argsort(depth)
        means_2d, cov_2d, visible = gaussians.transform_to_2dframe(camera) # (N,2), (N,2,2), (N,)

        # retain_grad on the FULL means_2d so adaptive control can read it.
        # gradients from each chunk's slice flow back here through autograd.
        if means_2d.requires_grad:
            means_2d.retain_grad()
        self.last_means_2d = means_2d

        sort_ind = sort_ind[visible[sort_ind]]   # drop near-plane culled, keep depth order

        opacities = gaussians.get_opacity # (N,)
        colors    = gaussians.get_color(camera.c2w, active_sh_deg) # (N, 3)

        # Reorder by depth once, slice into chunks below
        means_2d_s  = means_2d[sort_ind]
        cov_2d_s    = cov_2d[sort_ind]
        opacities_s = opacities[sort_ind]
        colors_s    = colors[sort_ind]

        xv, yv = torch.meshgrid(
            torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy'
        ) # (H, W)

        image    = torch.zeros(h, w, 3, device=device)
        T_global = torch.ones(h, w, device=device)

        n_visible = sort_ind.shape[0]
        use_checkpoint = self.training and means_2d.requires_grad
        chunk_size = self._resolve_chunk_size(h, w, device)

        for k in range(0, n_visible, chunk_size):
            args = (
                means_2d_s[k:k+chunk_size],
                cov_2d_s[k:k+chunk_size],
                opacities_s[k:k+chunk_size],
                colors_s[k:k+chunk_size],
                xv, yv, T_global, h, w,
            )
            if use_checkpoint:
                contrib, T_global = checkpoint(self._render_chunk, *args, use_reentrant=False)
            else:
                contrib, T_global = self._render_chunk(*args)
            image = image + contrib

        image = image + T_global.unsqueeze(-1) * bg_color # fill in remaining transmittance with background color
        return image

    @property
    def viewspace_position_grad(self):
        return self.last_means_2d.grad
