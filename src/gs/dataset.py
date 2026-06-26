from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class Camera:
    """Pinhole camera in OpenCV convention: +X right, +Y down, +Z into scene."""
    c2w: torch.Tensor   # (4, 4) float32, camera-to-world
    w2c: torch.Tensor   # (4, 4) float32, world-to-camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to(self, device):
        self.c2w = self.c2w.to(device)
        self.w2c = self.w2c.to(device)
        return self

class BlenderDataset(Dataset):
    """NeRF-synthetic (Blender) dataset.

    Directory layout:
        <scene_dir>/
            transforms_train.json
            transforms_val.json
            transforms_test.json
            train/r_0.png  ...
            val/r_0.png    ...
            test/r_0.png   ...
    """

    def __init__(
        self,
        scene_dir: str | Path,
        split: str = "train",
        downscale: int = 1,
        white_bg: bool = True,
    ):
        self.root = Path(scene_dir)
        self.downscale = downscale
        self.white_bg = white_bg

        with open(self.root / f"transforms_{split}.json") as f:
            meta = json.load(f)

        self.frames = meta["frames"]
        self._camera_angle_x = meta["camera_angle_x"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = self._load_image(frame["file_path"])
        cam = self._make_camera(frame["transform_matrix"], img.shape[-1], img.shape[-2])
        return img, cam

    # ------------------------------------------------------------------

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.root / (rel_path + ".png")
        rgba = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0

        if self.downscale > 1:
            h, w = rgba.shape[:2]
            pil = Image.fromarray((rgba * 255).astype(np.uint8))
            pil = pil.resize((w // self.downscale, h // self.downscale), Image.LANCZOS)
            rgba = np.array(pil, dtype=np.float32) / 255.0

        rgb, alpha = rgba[..., :3], rgba[..., 3:4]

        if self.white_bg:
            rgb = rgb * alpha + (1.0 - alpha)   # composite over white
        else:
            rgb = rgb * alpha

        # (H, W, 3) → (3, H, W)
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _make_camera(self, c2w_list: list, width: int, height: int) -> Camera:
        c2w = torch.tensor(c2w_list, dtype=torch.float32)  # (4, 4)

        # NeRF-synthetic uses OpenGL convention: +Y up, camera looks along -Z.
        # Flip Y and Z columns so the matrix becomes OpenCV convention
        # (+Y down, camera looks along +Z), which our rasterizer will expect.
        c2w[:, 1:3] *= -1

        f = 0.5 * width / np.tan(0.5 * self._camera_angle_x)

        return Camera(
            c2w=c2w,
            w2c=torch.linalg.inv(c2w),
            fx=f, fy=f,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )
    
def compute_scene_extent(dataset:BlenderDataset):
    centers = torch.tensor([f["transform_matrix"][i][3] for f in dataset.frames for i in range(3)], dtype=torch.float32).reshape(-1,3)
    isocenter = centers.mean(0,keepdim=True)
    return torch.linalg.norm(centers-isocenter,dim=-1).max().item() * 1.1
