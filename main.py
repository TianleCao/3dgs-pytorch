import torch
import torch.optim as optim
from dataset import BlenderDataset, compute_scene_extent
from gaussian import GaussianModel
from torch.utils.data import DataLoader
from train import train_step
from rasterizer import Rasterizer
from control import adaptive_control, reset_opacity
from tqdm import trange

lambda_dssim = 0.2 # D-SSIM loss scaling
N_gaussians = 100_000
densification_interval = 100
opacity_reset_interval = 3000
densify_from_iter = 500
densify_until_iter = 15_000
densify_grad_threshold = 0.0002
percent_dense = 0.01
opacity_threshold = 0.005
opacity_reset_value = 0.01
position_lr_init = 0.00016
position_lr_final = 0.0000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000
feature_lr = 0.0025
opacity_lr = 0.025
scaling_lr = 0.005
rotation_lr = 0.001

def collate_fn(batch):
    imgs, cams = zip(*batch)
    assert len(imgs) == 1, "only batch size = 1 is supported"
    return torch.stack(imgs,0), cams[0]
 
def get_position_lr(step:int):
    t = step / position_lr_max_steps
    return position_lr_init * (position_lr_final/position_lr_init)**t

def cycle(dataloader:DataLoader):
    while True:
        for data in dataloader:
            yield data

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BlenderDataset('data', 'train', downscale=1)
    val_dataset = BlenderDataset('data', 'val', downscale=1)
    test_dataset = BlenderDataset('data','test', downscale=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn = collate_fn)
    scene_extent = compute_scene_extent(train_dataset)
    ## random init for now
    gaussians = GaussianModel(N_gaussians).to(device) # 100K gaussians
    
    rasterizer = Rasterizer()

    optimizer = optim.Adam([{'name': 'mean','params': [gaussians.mean], 'lr':position_lr_init*scene_extent}, 
                            {'name': 'scale','params': [gaussians.scale], 'lr':scaling_lr},
                            {'name': 'rotation','params': [gaussians.rotation], 'lr': rotation_lr},
                            {'name': 'opacity','params': [gaussians.opacity], 'lr': opacity_lr},
                            {'name': 'sh_coeff','params': [gaussians.sh_coeff], 'lr': feature_lr}]) # list of dict to specify lr for each group

    ## training
    data_iter = cycle(train_dataloader)
    grad_accum = torch.zeros(N_gaussians, device=device)
    grad_denom = torch.zeros(N_gaussians, device=device)
    bg_color = torch.tensor([1.0,1.0,1.0], device=device)
    for step in trange(position_lr_max_steps):
        img, cam = next(data_iter)
        img = img.to(device)
        cam = cam.to(device)
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'mean':
                param_group['lr'] = get_position_lr(step)*scene_extent
        loss = train_step(rasterizer, gaussians, img, cam, optimizer, bg_color, lambda_dssim, min(3, step//1000), grad_accum, grad_denom)
        if step >= densify_from_iter and step < densify_until_iter and step % densification_interval == 0:
            grad_accum, grad_denom = adaptive_control(optimizer, gaussians, grad_accum, grad_denom, 
                     densify_grad_threshold, scene_extent, percent_dense, opacity_threshold)
        if step >0 and step < densify_until_iter and step % opacity_reset_interval == 0:
            reset_opacity(gaussians, opacity_reset_value)
