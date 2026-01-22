import torch
import warp as wp
# Assumes uw_renderer_utils exists in the same directory
from uw_renderer_utils import UW_render

def make_underwater_single(rgb, depth):
    """
    rgb:   torch uint8 tensor, shape (H, W, 4)  (RGBA from Isaac camera)
    depth: torch float32 tensor, shape (H, W)

    Returns:
        uw_rgb: torch float32 tensor, (H, W, 3), values in [0,1]
    """
    # sanity and shape fixes
    if rgb.dim() != 3 or rgb.shape[-1] != 4:
        raise ValueError(f"rgb must be (H, W, 4) uint8, got {rgb.shape}")
    if depth.dim() != 2:
        raise ValueError(f"depth must be (H, W) float32, got {depth.shape}")

    H, W, _ = rgb.shape

    rgb = rgb.contiguous()
    depth = depth.contiguous()

    # choose device for Warp (GPU if available)
    device = wp.get_preferred_device()

    # torch → Warp arrays
    raw_wp   = wp.from_torch(rgb,   dtype=wp.uint8,   )
    depth_wp = wp.from_torch(depth, dtype=wp.float32,)

    # allocate output (same shape as raw image)
    uw_wp = wp.empty_like(raw_wp)

    # underwater params (same defaults as UW_Camera)
    backscatter_value = wp.vec3(0.0, 0.31, 0.24)
    atten_coeff       = wp.vec3(0.05, 0.05, 0.05)   # UW_param[6:9] in original
    backscatter_coeff = wp.vec3(0.05, 0.20, 0.05)   # UW_param[3:6] in original

    # launch kernel; note dim = (H, W) because you index [i, j]
    wp.launch(
        dim=(H, W),
        kernel=UW_render,
        inputs=[raw_wp, depth_wp, backscatter_value, atten_coeff, backscatter_coeff],
        outputs=[uw_wp],
        device=device,
    )

    # Warp → torch
    uw = wp.to_torch(uw_wp)  # (H, W, 4) uint8

    # keep RGB, normalize to [0,1]
    uw_rgb = uw[..., :3].to(torch.float32) / 255.0    # (H, W, 3)

    return uw_rgb

def make_underwater_batch(rgb_batch, depth_batch):
    """
    rgb_batch:   (N, H, W, 4) uint8
    depth_batch: (N, H, W)    float32

    Returns:
        uw_batch: (N, H, W, 3) float32 in [0,1]
    """
    if rgb_batch.dim() != 4 or rgb_batch.shape[-1] != 4:
        raise ValueError(f"rgb_batch must be (N, H, W, 4), got {rgb_batch.shape}")
    if depth_batch.dim() != 3:
        raise ValueError(f"depth_batch must be (N, H, W), got {depth_batch.shape}")

    N, H, W, _ = rgb_batch.shape
    if depth_batch.shape[0] != N or depth_batch.shape[1] != H or depth_batch.shape[2] != W:
        raise ValueError("rgb_batch and depth_batch shapes do not match")

    uw_list = []
    for i in range(N):
        uw = make_underwater_single(rgb_batch[i], depth_batch[i])
        uw_list.append(uw)

    uw_batch = torch.stack(uw_list, dim=0)
    return uw_batch.permute(0, 3, 1, 2)

def make_underwater_batch_torch(
    rgb_batch,         # (N,H,W,4) uint8
    depth_batch,       # (N,H,W) float32
    backscatter_value = (0.0, 0.31, 0.24), # iterable of 3 floats in [0,1]
    atten_coeff = (0.05, 0.05, 0.05),       # iterable of 3 floats
    backscatter_coeff = (0.05, 0.20, 0.05), # iterable of 3 floats
    device=None,
):
    """
    Returns:
        uw_batch: (N,4,H,W) uint8   # RGBA, same scaling as input
    """
    if device is None:
        device = rgb_batch.device

    # ensure correct dtypes/devices
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)

    # split RGB / A
    rgb = rgb_batch[..., :3].to(torch.float32)  # (N,H,W,3) in [0,255]

    # depth to (N,H,W,1) for broadcasting
    d = depth_batch.unsqueeze(-1).to(torch.float32)  # (N,H,W,1)

    # params as (1,1,1,3) for broadcasting
    backscatter_value = torch.tensor(backscatter_value, dtype=torch.float32, device=device).view(1,1,1,3)
    atten_coeff       = torch.tensor(atten_coeff,       dtype=torch.float32, device=device).view(1,1,1,3)
    backscatter_coeff = torch.tensor(backscatter_coeff, dtype=torch.float32, device=device).view(1,1,1,3)

    # exp_atten = exp(- depth * atten_coeff)
    exp_atten = torch.exp(-d * atten_coeff)          # (N,H,W,3)

    # exp_back = exp(- depth * backscatter_coeff)
    exp_back = torch.exp(-d * backscatter_coeff)     # (N,H,W,3)

    # UW_RGB = raw_RGB * exp_atten + backscatter_value*255 * (1 - exp_back)
    uw_rgb = rgb * exp_atten \
           + backscatter_value * 255.0 * (1.0 - exp_back)

    # clamp 0..255 and cast back to uint8
    uw_rgb = torch.clamp(uw_rgb, 0, 255)
    uw_rgb_u8 = uw_rgb.to(torch.uint8)

    return uw_rgb_u8.permute(0, 3, 1, 2)

def rgb_to_gray_torch(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (B,3,H,W) or (B,4,H,W) -> (B,1,H,W)
    if rgb.size(1) == 4:
        rgb = rgb[:, :3]
    w = rgb.new_tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
    gray = (rgb.to(torch.float32) * w).sum(dim=1, keepdim=True)
    return gray
