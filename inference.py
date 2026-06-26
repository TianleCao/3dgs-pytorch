"""Render a small test-view gallery from a trained 3DGS checkpoint.

Usage (CLI):
    python inference.py --checkpoint checkpoints/final.pt --data_dir data/lego \
                        --output checkpoints/gallery.png --num_views 3

Usage (notebook / Python):
    from inference import render_gallery
    render_gallery("checkpoints/final.pt", "data/lego", "gallery.png", num_views=3)
"""
import argparse
import matplotlib
import torch

from gaussian import GaussianModel
from rasterizer import Rasterizer
from dataset import BlenderDataset


def render_gallery(checkpoint_path: str, data_dir: str, output_path: str,
                   num_views: int = 3, active_sh_deg: int = 3) -> dict:
    """Render `num_views` evenly-spaced test images and save a GT-vs-render gallery.

    Returns a dict with the per-view and mean PSNR.
    """
    # Set non-interactive backend lazily so importing this module never grabs a display.
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print(f"Loaded checkpoint: N={ckpt['N']}, downscale={ckpt['downscale']}")

    gaussians = GaussianModel(ckpt["N"]).to(device)
    gaussians.load_state_dict(ckpt["gaussian_state"])

    rasterizer = Rasterizer().to(device)
    rasterizer.eval()

    test_dataset = BlenderDataset(data_dir, "test", downscale=ckpt["downscale"])
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

    if num_views == 1:
        indices = [len(test_dataset) // 2]
    else:
        n = len(test_dataset)
        indices = [int(i * (n - 1) / (num_views - 1)) for i in range(num_views)]

    fig, axes = plt.subplots(num_views, 2, figsize=(8, 4 * num_views), squeeze=False)
    psnrs = []

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, cam = test_dataset[idx]
            cam = cam.to(device)
            rendered = rasterizer(gaussians, cam, bg_color, active_sh_deg=active_sh_deg).clamp(0, 1).cpu()
            gt = img.permute(1, 2, 0)
            psnr = -10.0 * torch.log10(((rendered - gt) ** 2).mean()).item()
            psnrs.append(psnr)

            axes[row, 0].imshow(gt);       axes[row, 0].set_title(f"GT (view {idx})");          axes[row, 0].axis("off")
            axes[row, 1].imshow(rendered); axes[row, 1].set_title(f"Render (PSNR {psnr:.2f})"); axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_psnr = sum(psnrs) / len(psnrs)
    print(f"Saved gallery to {output_path}")
    print(f"PSNR per view: {[f'{p:.2f}' for p in psnrs]}  mean={mean_psnr:.2f}")
    return {"psnrs": psnrs, "mean_psnr": mean_psnr, "output_path": output_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--data_dir", default="data/lego")
    parser.add_argument("--output", default="checkpoints/inference_gallery.png")
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--sh_deg", type=int, default=3)
    args = parser.parse_args()
    render_gallery(args.checkpoint, args.data_dir, args.output,
                   num_views=args.num_views, active_sh_deg=args.sh_deg)


if __name__ == "__main__":
    main()
