import torch
import torch.nn as nn
from .gaussian import GaussianModel
from .utils import inverse_sigmoid
import math

@torch.no_grad()
def reset_opacity(gaussians: GaussianModel, opacity_reset_value:float):
    gaussians.opacity.data.copy_(inverse_sigmoid(torch.ones_like(gaussians.get_opacity)*opacity_reset_value))

@torch.no_grad()
def adaptive_control(optimizer: torch.optim.Optimizer, gaussians: GaussianModel, grad_accum: torch.Tensor, grad_denom: torch.Tensor, 
                     densify_grad_threshold: float, scene_extent: float, percent_dense: float, opacity_threshold: float, prune_big_scale:bool):
    grad = grad_accum / grad_denom
    grad[grad.isnan()] = 0
    candidates = grad>densify_grad_threshold
    scales = torch.max(gaussians.get_scale,dim=-1).values
    opacity_values = gaussians.get_opacity
    split_mask = candidates & (scales>scene_extent*percent_dense)
    clone_mask = candidates & (scales<=scene_extent*percent_dense)
    clone_tensors = clone_gaussian(gaussians, clone_mask)
    split_tensors = split_gaussian(gaussians, split_mask)
    cat_tensors_to_optimizer(optimizer, gaussians, merge(clone_tensors, split_tensors))
    if prune_big_scale:
        # match trainging() function of reference repo, where we only prune based on scale after we perform opacity reset at least once
        keep_mask = (~split_mask) & (opacity_values>=opacity_threshold) & (scales<scene_extent*0.1)
    else:
        keep_mask = (~split_mask) & (opacity_values>=opacity_threshold)
    N_new = clone_mask.sum() + split_mask.sum()*2
    keep_mask = torch.cat([keep_mask, torch.ones(N_new, dtype=bool, device=split_mask.device)],0)
    prune_gaussian(optimizer, gaussians, keep_mask)
    N = len(gaussians.mean)
    return torch.zeros(N, device=grad_accum.device), torch.zeros(N, device=grad_denom.device)

@torch.no_grad()
def cat_tensors_to_optimizer(optimizer: torch.optim.Optimizer, gaussians: GaussianModel, new_tensors: dict):
    if not new_tensors:
        return
    for group in optimizer.param_groups:
        name = group['name']
        old_params = group['params'][0]
        new_vals = new_tensors[name]
        new_params = nn.Parameter(torch.cat([old_params, new_vals], 0))

        old_state = optimizer.state.pop(old_params)
        optimizer.state[new_params] = {'step': old_state['step'], 'exp_avg': torch.cat([old_state['exp_avg'], torch.zeros_like(new_vals)],0), 
                                       'exp_avg_sq': torch.cat([old_state['exp_avg_sq'], torch.zeros_like(new_vals)],0)}
        group['params'][0] = new_params
        setattr(gaussians, name, new_params)

@torch.no_grad()
def clone_gaussian(gaussians: GaussianModel, clone_mask: torch.Tensor):
    if torch.sum(clone_mask) == 0:
        return 
    # as in densify_and_clone() function of reference impl, the authors didn't "move it in the direction of the positional gradient", but just copy
    new_tensors = {}
    new_tensors['mean'] = gaussians.mean[clone_mask]
    new_tensors['scale'] = gaussians.scale[clone_mask]
    new_tensors['rotation'] = gaussians.rotation[clone_mask]
    new_tensors['opacity'] = gaussians.opacity[clone_mask]
    new_tensors['sh_coeff'] = gaussians.sh_coeff[clone_mask]
    return new_tensors

@torch.no_grad()
def split_gaussian(gaussians: GaussianModel, split_mask: torch.Tensor):
    if torch.sum(split_mask) == 0:
        return
    new_tensors = {}
    z = torch.randn((2, torch.sum(split_mask), 3), device = split_mask.device) # random normal gaussian, [2,M,3]
    new_means = gaussians.mean[split_mask] + (gaussians.get_cov_world_square_root()[split_mask] @ z.unsqueeze(-1)).squeeze(-1) # [M,3] + sqz([M,3,3] @ [2,M,3,1]) -> [2,M,3]
    new_tensors['mean'] = new_means.reshape(-1,3)
    new_tensors['scale'] = torch.repeat_interleave(gaussians.scale[split_mask] - math.log(1.6),2,0)
    new_tensors['rotation'] = torch.repeat_interleave(gaussians.rotation[split_mask],2,0)
    new_tensors['opacity'] = torch.repeat_interleave(gaussians.opacity[split_mask],2)
    new_tensors['sh_coeff'] = torch.repeat_interleave(gaussians.sh_coeff[split_mask],2,0)
    return new_tensors

@torch.no_grad()
def prune_gaussian(optimizer: torch.optim.Optimizer, gaussians: GaussianModel, keep_mask:torch.Tensor):
    for group in optimizer.param_groups:
        name = group['name']
        old_params = group['params'][0]
        new_params = nn.Parameter(old_params[keep_mask])

        old_state = optimizer.state.pop(old_params)
        optimizer.state[new_params] = {'step': old_state['step'], 'exp_avg': old_state['exp_avg'][keep_mask], 
                                       'exp_avg_sq': old_state['exp_avg_sq'][keep_mask]}
        group['params'][0] = new_params
        setattr(gaussians, name, new_params)

def merge(*dicts):
    dicts = [d for d in dicts if d is not None]
    if not dicts:
        return None
    merged = {}
    params = dicts[0].keys()
    for param in params:
        merged[param] = torch.cat([d[param] for d in dicts],0)
    return merged
