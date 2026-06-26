import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from .rasterizer import Rasterizer
from .gaussian import GaussianModel


@torch.no_grad()
def evaluate(rasterizer: Rasterizer, gaussians: GaussianModel, dataloader: DataLoader,
             bg_color: torch.Tensor, active_sh_deg: int, device, max_frames: int = None):
    """Render each frame, return mean PSNR / SSIM and one sample render for logging."""
    ssim = SSIM(data_range=1.0).to(device)
    psnrs, ssims = [], []
    sample_pred, sample_gt = None, None

    for i, (img, cam) in enumerate(dataloader):
        if max_frames is not None and i >= max_frames:
            break
        img = img.to(device)
        cam = cam.to(device)

        rendered = rasterizer(gaussians, cam, bg_color, active_sh_deg)  # (H,W,3)
        pred = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)        # (1,3,H,W)

        mse = torch.mean((pred - img) ** 2)
        psnrs.append(-10.0 * torch.log10(mse).item())
        ssims.append(ssim(pred, img).item())

        if i == 0:
            sample_pred, sample_gt = pred.squeeze(0).cpu(), img.squeeze(0).cpu()

    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "n": len(psnrs),
        "sample_pred": sample_pred,
        "sample_gt": sample_gt,
    }
