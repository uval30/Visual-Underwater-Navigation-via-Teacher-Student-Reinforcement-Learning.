import torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, ActionTerm, ActionTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from pxr import UsdLux, UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# ----------------------------
# Math & Helper Functions
# ----------------------------
def _stash_term(env, name: str, tensor: torch.Tensor):
    if not hasattr(env, "extras") or env.extras is None: env.extras = {}
    d = env.extras.setdefault("reward_terms", {})
    d[name] = tensor.detach().cpu().numpy()

def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dim=-1)

def add_quat_noise(q, sigma_rad=0.1):
    N = q.shape[0]
    axis = torch.randn(N, 3, device=q.device)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    theta = torch.randn(N, 1, device=q.device) * sigma_rad
    half = 0.5 * theta
    dq = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=1)
    qn = quat_mul(dq, q)
    return qn / (qn.norm(dim=1, keepdim=True) + 1e-8)

def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

# ----------------------------
# Rewards & Observations
# ----------------------------
def distance_to_target(env, max_dist : float = 4):
    rov, tgt = env.scene["rov"].data.root_state_w[:, :3], env.scene["target"].data.root_state_w[:, :3]
    val = torch.norm(rov - tgt, dim=-1) / max_dist
    _stash_term(env, "distance_to_target", val)
    return val

def progress_to_target_normalized(env, eps: float = 1e-6):
    rov, tgt = env.scene["rov"].data.root_state_w[:, :3], env.scene["target"].data.root_state_w[:, :3]
    dist = torch.norm(rov - tgt, dim=-1)
    
    # Init buffers
    if not hasattr(env, "_buf_prev_dist") or env._buf_prev_dist.shape != dist.shape:
        env._buf_prev_dist, env._buf_prev_rov, env._buf_prev_tgt = dist.clone(), rov.clone(), tgt.clone()

    raw = env._buf_prev_dist - dist
    d_rel = (rov - env._buf_prev_rov) - (tgt - env._buf_prev_tgt)
    path_len = torch.norm(d_rel, dim=-1).clamp_min(eps)
    prog_norm = (raw / path_len).clamp(-1.0, 1.0)

    # Handle resets
    reset_mask = (env.episode_length_buf == 0) if hasattr(env, "episode_length_buf") else torch.zeros_like(prog_norm, dtype=torch.bool)
    if reset_mask.any():
        prog_norm[reset_mask] = 0.0
        env._buf_prev_dist[reset_mask] = dist[reset_mask]
        env._buf_prev_rov[reset_mask] = rov[reset_mask]
        env._buf_prev_tgt[reset_mask] = tgt[reset_mask]

    env._buf_prev_dist, env._buf_prev_rov, env._buf_prev_tgt = dist.detach(), rov.detach(), tgt.detach()
    _stash_term(env, "prog_norm", prog_norm)
    return prog_norm

def goal_reached_bool(env, radius: float = 0.5):
    return torch.norm(env.scene["rov"].data.root_state_w[:, :3] - env.scene["target"].data.root_state_w[:, :3], dim=-1) <= radius

def goal_reached(env, radius: float = 0.5):
    reach = goal_reached_bool(env, radius).float()
    _stash_term(env, "goal_reached", reach)
    return reach

def collision_bool(env, threshold: float = 0.5):
    return env.scene["rov_contact"].data.net_forces_w.norm(dim=-1) > threshold

def pointing_loss(env, axis=(1., 0., 0.), lam_rate=0.5):
    root = env.scene["rov"].data.root_state_w
    pos, q, w_w = root[:, :3], root[:, 3:7], root[:, 10:13]
    tgt = env.scene["target"].data.root_state_w[:, :3]
    
    rel_b = quat_apply_inverse(q, tgt - pos) - torch.tensor([0., 0., 0.055], device=pos.device)
    r_hat = rel_b / rel_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ex = torch.tensor(axis, device=pos.device).expand_as(r_hat)
    
    align = 2.0 * (0.5 * ((ex * r_hat).sum(-1).clamp(-1, 1) + 1.0))**4.0 - 1.0
    w_b = quat_apply_inverse(q, w_w)
    w_perp = (w_b - (w_b * ex).sum(-1, keepdim=True) * ex).norm(dim=-1)
    huber = torch.tanh(torch.where(w_perp <= 0.5, 0.5*w_perp**2, 0.5*(w_perp-0.25)))
    
    reward = align - lam_rate * 0.5 * (1.0 + align) * huber
    _stash_term(env, "pointing_loss", reward)
    return reward

def obs_orientation_bodyframe_noise(env, sigma_rad=0.1):
    q_wb = env.scene["rov"].data.root_state_w[:, 3:7]
    return add_quat_noise(quat_conjugate(q_wb), sigma_rad)

def obs_target_rel_body(env):
    root = env.scene['rov'].data.root_state_w
    rel_w = env.scene["target"].data.root_state_w[:, :3] - root[:, :3]
    return quat_apply_inverse(root[:, 3:7], rel_w)

def obs_target_rel_body_noise(env):
    rel_b = obs_target_rel_body(env)
    return rel_b + torch.randn_like(rel_b) * 0.1

# ----------------------------
# Physics & Control Classes
# ----------------------------
class ROVWrenchController(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init
