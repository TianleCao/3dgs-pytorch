import argparse
import os

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from tqdm import trange

from gs.dataset import BlenderDataset, compute_scene_extent
from gs.gaussian import GaussianModel
from gs.train import train_step
from gs.rasterizer import Rasterizer
from gs.control import adaptive_control, reset_opacity
from gs.evaluate import evaluate


def collate_fn(batch):
    imgs, cams = zip(*batch)
    assert len(imgs) == 1, "only batch size = 1 is supported"
    return torch.stack(imgs, 0), cams[0]


def get_position_lr(step: int, cfg: dict) -> float:
    t = step / cfg["position_lr_max_steps"]
    return cfg["position_lr_init"] * (cfg["position_lr_final"] / cfg["position_lr_init"]) ** t


def cycle(dataloader: DataLoader):
    while True:
        for data in dataloader:
            yield data


def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BlenderDataset(cfg["data_dir"], "train", downscale=cfg["downscale"])
    val_dataset   = BlenderDataset(cfg["data_dir"], "val",   downscale=cfg["downscale"])
    test_dataset  = BlenderDataset(cfg["data_dir"], "test",  downscale=cfg["downscale"])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_dataloader   = DataLoader(val_dataset,   batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_dataloader  = DataLoader(test_dataset,  batch_size=1, shuffle=True, collate_fn=collate_fn)

    scene_extent = compute_scene_extent(train_dataset)
    gaussians = GaussianModel(cfg["N_gaussians"]).to(device)
    rasterizer = Rasterizer()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer = optim.Adam([
        {"name": "mean",     "params": [gaussians.mean],     "lr": cfg["position_lr_init"] * scene_extent},
        {"name": "scale",    "params": [gaussians.scale],    "lr": cfg["scaling_lr"]},
        {"name": "rotation", "params": [gaussians.rotation], "lr": cfg["rotation_lr"]},
        {"name": "opacity",  "params": [gaussians.opacity],  "lr": cfg["opacity_lr"]},
        {"name": "sh_coeff", "params": [gaussians.sh_coeff], "lr": cfg["feature_lr"]},
    ])

    writer = SummaryWriter(cfg["log_dir"])
    data_iter = cycle(train_dataloader)
    grad_accum = torch.zeros(cfg["N_gaussians"], device=device)
    grad_denom = torch.zeros(cfg["N_gaussians"], device=device)
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

    pbar = trange(cfg["position_lr_max_steps"])
    for step in pbar:
        img, cam = next(data_iter)
        img = img.to(device)
        cam = cam.to(device)

        for param_group in optimizer.param_groups:
            if param_group["name"] == "mean":
                param_group["lr"] = get_position_lr(step, cfg) * scene_extent

        active_sh_deg = min(3, step // cfg["sh_schedule_interval"])
        loss = train_step(rasterizer, gaussians, img, cam, optimizer, bg_color,
                          cfg["lambda_dssim"], active_sh_deg, grad_accum, grad_denom, ssim)
        pbar.set_postfix(loss=f"{loss:.4f}", N=gaussians.mean.shape[0])

        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/num_gaussians", gaussians.mean.shape[0], step)
        writer.add_scalar("train/lr_position", optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("train/active_sh_deg", active_sh_deg, step)

        if (step >= cfg["densify_from_iter"] and step < cfg["densify_until_iter"]
                and step % cfg["densification_interval"] == 0):
            grad_accum, grad_denom = adaptive_control(
                optimizer, gaussians, grad_accum, grad_denom,
                cfg["densify_grad_threshold"], scene_extent,
                cfg["percent_dense"], cfg["opacity_threshold"],
                step >= cfg["opacity_reset_interval"],
            )
        if step > 0 and step < cfg["densify_until_iter"] and step % cfg["opacity_reset_interval"] == 0:
            reset_opacity(gaussians, cfg["opacity_reset_value"])

        if step > 0 and step % cfg["eval_interval"] == 0:
            metrics = evaluate(rasterizer, gaussians, val_dataloader, bg_color,
                               active_sh_deg, device, max_frames=cfg["eval_max_frames"])
            writer.add_scalar("val/psnr", metrics["psnr"], step)
            writer.add_scalar("val/ssim", metrics["ssim"], step)
            writer.add_image("val/pred", metrics["sample_pred"], step)
            writer.add_image("val/gt",   metrics["sample_gt"],   step)

    # final test eval
    test_metrics = evaluate(rasterizer, gaussians, test_dataloader, bg_color,
                            active_sh_deg=3, device=device)
    writer.add_scalar("test/psnr", test_metrics["psnr"], cfg["position_lr_max_steps"])
    writer.add_scalar("test/ssim", test_metrics["ssim"], cfg["position_lr_max_steps"])
    print(f"\nTest: PSNR={test_metrics['psnr']:.2f}  "
          f"SSIM={test_metrics['ssim']:.4f}  (n={test_metrics['n']})")
    writer.close()

    # Save final Gaussian state
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "gaussian_state": gaussians.state_dict(),
        "N": gaussians.mean.shape[0],
        "downscale": train_dataset.downscale,
    }, "checkpoints/final.pt")
    print(f"Saved checkpoint with N={gaussians.mean.shape[0]} Gaussians.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3DGS on a Blender scene from a YAML config.")
    parser.add_argument("--config", default="configs/paper.yaml",
                        help="Path to YAML config file (default: configs/paper.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")
    print(f"  resolution: 800/{cfg['downscale']} = {800 // cfg['downscale']}px")
    print(f"  N_gaussians (init): {cfg['N_gaussians']}, steps: {cfg['position_lr_max_steps']}")

    main(cfg)
