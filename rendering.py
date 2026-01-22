import torch
import warp as wp
# Assumes uw_renderer_utils exists as per original file, or provides stub
from uw_renderer_utils import UW_render 

def make_underwater_single(rgb, depth):
    """rgb: (H,W,4) uint8, depth: (H,W) float32 -> Returns (H,W,3) float32 [0,1]"""
    H, W, _ = rgb.shape
    device = wp.get_preferred_device()
    
    raw_wp = wp.from_torch(rgb.contiguous(), dtype=wp.uint8)
    depth_wp = wp.from_torch(depth.contiguous(), dtype=wp.float32)
    uw_wp = wp.empty_like(raw_wp)

    wp.launch(
        dim=(H, W), kernel=UW_render,
        inputs=[raw_wp, depth_wp, wp.vec3(0.0, 0.31, 0.24), wp.vec3(0.05, 0.05, 0.05), wp.vec3(0.05, 0.20, 0.05)],
        outputs=[uw_wp], device=device,
    )
    return wp.to_torch(uw_wp)[..., :3].to(torch.float32) / 255.0

def make_underwater_batch_torch(rgb_batch, depth_batch, device=None):
    """Torch-only implementation for batch processing."""
    if device is None: device = rgb_batch.device
    rgb = rgb_batch[..., :3].to(device, dtype=torch.float32)
    d = depth_batch.unsqueeze(-1).to(device, dtype=torch.float32)
    
    # Params
    bs_val = torch.tensor([0.0, 0.31, 0.24], device=device).view(1,1,1,3)
    atten = torch.tensor([0.05]*3, device=device).view(1,1,1,3)
    bs_coeff = torch.tensor([0.05, 0.20, 0.05], device=device).view(1,1,1,3)

    uw_rgb = rgb * torch.exp(-d * atten) + bs_val * 255.0 * (1.0 - torch.exp(-d * bs_coeff))
    return torch.clamp(uw_rgb, 0, 255).to(torch.uint8).permute(0, 3, 1, 2)

def rgb_to_gray_torch(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.size(1) == 4: rgb = rgb[:, :3]
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
    return (rgb.float() * w).sum(dim=1, keepdim=True)
