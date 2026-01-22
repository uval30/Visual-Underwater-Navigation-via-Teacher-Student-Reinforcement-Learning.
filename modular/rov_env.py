import argparse
import math
import torch
import torch.nn.functional as F
from pxr import UsdLux, UsdGeom, Gf

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, quat_apply_inverse

# Import argparse args from a central config or handle them here
# For the sake of this file, we assume args_cli is available or we define defaults
# If running standalone, you might need to re-parse args or pass them in.
# Here we just use a placeholder for num_envs if not provided externally.
try:
    from __main__ import args_cli
except ImportError:
    class Args:
        num_envs = 16
        device = "cuda"
    args_cli = Args()

# ----------------------------
# Physics & Control Classes
# ----------------------------

class ROVWrenchController(ActionTerm):
    """
    RL -> direct body-frame wrench [Fx,Fy,Fz, Tx,Ty,Tz].
    Just applies limits + first-order actuator lag.
    """
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._rov = env.scene[cfg.asset_name]
        self._env = env
        self._device   = getattr(self._rov, "device", getattr(env, "device", "cpu"))
        self._num_envs = int(env.num_envs)
        self.T_act = 0.12
        self._tau_cmd = torch.zeros((self._num_envs, 6), device=self._device)
        self._raw     = torch.zeros((self._num_envs, 6), device=self._device)
        self.dt        = float(getattr(env.cfg.sim, "dt", None))
        self.zero = torch.zeros((self._num_envs, 1), device=self._device)
    @property
    def action_dim(self): return 6
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._tau_cmd

    def process_actions(self, actions: torch.Tensor):
        # env-rate: just store setpoints (body-frame desired)
        self._raw = actions

    def apply_actions(self):
        
        F = self._raw[:, :3]
        T = self._raw[:, 3: ]
        tau_ref = torch.cat([F, T], dim=-1)

        noise_scale = 0.10
        noise = 1.0 + noise_scale * (2.0 * torch.rand_like(tau_ref) - 1.0)
        tau_ref = tau_ref * noise
        tau_ref = torch.clamp(tau_ref, -100.0, 100.0)

        self._tau_cmd = tau_ref 


def coriolis_added_mass_matrix(v_b, w_b, A_lin, A_rot):
    """
    Compute added-mass Coriolis matrix C_A(nu) for diagonal M_A.
    """
    u, v, w = v_b[:, 0], v_b[:, 1], v_b[:, 2]
    p, q, r = w_b[:, 0], w_b[:, 1], w_b[:, 2]
    Xu, Yv, Zw = A_lin
    Kp, Mq, Nr = A_rot

    N = v_b.shape[0]
    C_A = torch.zeros((N, 6, 6), device=v_b.device)

    # Fill elements according to Fossen/MDPI structure
    C_A[:, 0, 4] = -Zw * w
    C_A[:, 0, 5] =  Yv * v
    C_A[:, 1, 3] =  Zw * w
    C_A[:, 1, 5] = -Xu * u
    C_A[:, 2, 3] = -Yv * v
    C_A[:, 2, 4] =  Xu * u

    C_A[:, 3, 1] = -Zw * w
    C_A[:, 3, 2] =  Yv * v
    C_A[:, 3, 4] = -Nr * r
    C_A[:, 3, 5] =  Mq * q

    C_A[:, 4, 0] =  Zw * w
    C_A[:, 4, 2] = -Xu * u
    C_A[:, 4, 3] =  Nr * r
    C_A[:, 4, 5] = -Kp * p

    C_A[:, 5, 0] = -Yv * v
    C_A[:, 5, 1] =  Xu * u
    C_A[:, 5, 3] = -Mq * q
    C_A[:, 5, 4] =  Kp * p

    return C_A

class ROVHydroApplier(ActionTerm):
    """
    Computes drag + buoyancy + restoring (body frame), adds controller wrench,
    writes net wrench to PhysX each physics step.
    """
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._rov = env.scene[cfg.asset_name]
        self._env = env
        self._device   = getattr(self._rov, "device", getattr(env, "device", "cpu"))
        self._num_envs = int(env.num_envs)
        self.dt        = float(getattr(env.cfg.sim, "dt", 0.01))
        
        # damping (diag linear + quadratic)
        D1 = torch.tensor([13.7, 0.0, 33.0, 0.0, 0.8, 0.0], device=self._device)
        D2 = torch.tensor([141.0, 217.0, 190.0, 1.19, 0.47, 1.50], device=self._device)
        self.D1_lin = D1[:3];  self.D1_rot = D1[3:]
        self.D2_lin = D2[:3];  self.D2_rot = D2[3:]
        
        # hydrostatics
        self.B = 1000.0 * 9.81 * 0.0134
        self.r_cb_b  = torch.tensor([0.0, 0.0, 0.01], device=self._device)
        self.k_phi_theta = torch.tensor([1, 1], device=self._device)
        self._ctrl = None
        self._raw  = torch.zeros((self._num_envs, 0), device=self._device)
        self._proc = self._raw


        self.A_lin = torch.tensor([6.36, 7.12, 18.68], device=self._device)   # [X_u̇, Y_v̇, Z_ẇ]
        self.A_rot = torch.tensor([0.189, 0.135, 0.222], device=self._device)  # [K_ṗ, M_q̇, N_ṙ]

        # buffers for finite-difference accelerations in body frame
        self._prev_vb = torch.zeros((self._num_envs, 3), device=self._device)
        self._prev_wb = torch.zeros((self._num_envs, 3), device=self._device)
        self._a_lp    = torch.zeros((self._num_envs, 3), device=self._device)  # optional low-pass
        self._alp_lp  = torch.zeros((self._num_envs, 3), device=self._device)


        self.tether_anchor_offset = torch.tensor([-4.0, 0.0, 1.26], device=self._device)   # env-local
        self.tether_attach_b      = torch.tensor([-0.20, 0.0, 0.056], device=self._device) # body-frame attach
        self.tether_rho = 997.0
        self.tether_Cd  = 1.2
        self.tether_d   = 0.0076  # Fathom tether diameter (m)
        self.tether_L_min = 0.1
        self.tether_L_max = 30.0
        self.tether_force_clip = None  # e.g. 150.0 if you need stability


        self.T_tether_b = None
        self.F_tether_b = None


        self.err_A_lin = torch.tensor([1.2, 1.2, 1.2], device=self._device, dtype=torch.float32)  # [u,v,w]
        self.err_A_rot = torch.tensor([2.0, 2.0, 2.0], device=self._device, dtype=torch.float32)  # [p,q,r]
        self.err_D     = torch.tensor([1.10, 1.10, 1.15, 1.10, 1.10, 1.12], device=self._device, dtype=torch.float32)  # [u,v,w,p,q,r]

        # make sure A tensors are float and on correct device (usually already true)
        self.A_lin = self.A_lin.to(device=self._device, dtype=torch.float32)
        self.A_rot = self.A_rot.to(device=self._device, dtype=torch.float32)

        # OVERWRITE IN PLACE
        self.A_lin.mul_(self.err_A_lin)   # <-- was using err_A_lin (bug). Must be self.err_A_lin
        self.A_rot.mul_(self.err_A_rot)

        # D1, D2 are local tensors here; scale and then split into the fields used later
        D1 = D1.to(device=self._device, dtype=torch.float32)
        D2 = D2.to(device=self._device, dtype=torch.float32)

        D1.mul_(self.err_D)
        D2.mul_(self.err_D)

        self.D1_lin = D1[:3]
        self.D1_rot = D1[3:]
        self.D2_lin = D2[:3]
        self.D2_rot = D2[3:]

    @property
    def action_dim(self): return 0
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._proc

    def process_actions(self, actions: torch.Tensor):  # required by abstract base
        pass

    def apply_actions(self):
        if self._ctrl is None:
            self._ctrl = self._env.action_manager.get_term("lagg")

        root   = self._rov.data.root_state_w
        quat_w = root[:, 3:7]
        v_w    = root[:, 7:10]
        w_w    = root[:,10:13]


        v_b = quat_apply_inverse(quat_w, v_w)
        w_b = quat_apply_inverse(quat_w, w_w)

        abs_v = torch.abs(v_b); abs_w = torch.abs(w_b)
        Fd_b = -(self.D1_lin[None,:] + self.D2_lin[None,:]*abs_v) * v_b
        Td_b = -(self.D1_rot[None,:] + self.D2_rot[None,:]*abs_w) * w_b

        a_b    = (v_b - self._prev_vb) / self.dt
        alp_b  = (w_b - self._prev_wb) / self.dt
        alpha = 0.2
        self._a_lp   = (1-alpha)*self._a_lp   + alpha*a_b
        self._alp_lp = (1-alpha)*self._alp_lp + alpha*alp_b
        a_b   = self._a_lp
        alp_b = self._alp_lp

        self._prev_vb = v_b.detach()
        self._prev_wb = w_b.detach()

        F_am_b = -(self.A_lin[None, :] * a_b)     # N
        T_am_b = -(self.A_rot[None, :] * alp_b)   # N·m

    
        # buoyancy (world up -> body) + CB lever torque
        F_buoy_w = torch.zeros_like(v_w)
        F_buoy_w[:,2] = self.B
        F_buoy_b = quat_apply_inverse(quat_w, F_buoy_w)
        T_buoy_b = torch.cross(self.r_cb_b.expand_as(F_buoy_b), F_buoy_b, dim=-1)

     
        """nu = torch.cat([v_b, w_b], dim=-1)  # (N,6)
        C_A = coriolis_added_mass_matrix(v_b, w_b, self.A_lin, self.A_rot)
        tau_ca = -torch.bmm(C_A, nu.unsqueeze(-1)).squeeze(-1)  # (N,6)
        F_ca_b = tau_ca[:, :3]
        T_ca_b = tau_ca[:, 3:]"""

        # ------------------------------------THEATER--------------------------------------

        p_a_w = self._env.scene.env_origins + self.tether_anchor_offset.view(1, 3)

        # COM position in world
        p_c_w = root[:, :3]

        # attach point in world: p_r = p_c + R(q)*r_attach_body
        attach = self.tether_attach_b.view(1, 3).repeat(self._num_envs, 1)
        p_r_w = p_c_w + quat_apply(quat_w, attach)

        # tether direction
        r = p_r_w - p_a_w
        L = torch.norm(r, dim=1, keepdim=True).clamp_min(1e-6)
        r_hat = r / L

        # perpendicular velocity (world)
        v_par  = torch.sum(v_w * r_hat, dim=1, keepdim=True) * r_hat
        v_perp = v_w - v_par
        speed_perp = torch.norm(v_perp, dim=1, keepdim=True).clamp_min(1e-6)

        # effective length proxy
        L_eff = L.clamp(self.tether_L_min, self.tether_L_max)

        # quadratic cross-flow drag lumped over length
        K = 0.5 * self.tether_rho * self.tether_Cd * (self.tether_d * L_eff)  # (N,1)
        F_tether_w = -(K * speed_perp) * v_perp  # (N,3)

        # optional clip (helps RL stability)
        if self.tether_force_clip is not None:
            Fmag = torch.norm(F_tether_w, dim=1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(self.tether_force_clip / Fmag, max=1.0)
            F_tether_w = F_tether_w * scale

        # torque about COM from off-center attach point (world)
        lever_w = (p_r_w - p_c_w)
        T_tether_w = torch.cross(lever_w, F_tether_w, dim=1)

        # convert tether wrench to body frame (because you apply is_global=False)
        F_tether_b = quat_apply_inverse(quat_w, F_tether_w)
        T_tether_b = quat_apply_inverse(quat_w, T_tether_w)


        self.F_tether_b = F_tether_b
        self.T_tether_b = T_tether_b

        # thrust from controller (body-frame)
        tau_thr_b = self._ctrl.processed_actions
        F_thr_b, T_thr_b = tau_thr_b[:, :3], tau_thr_b[:, 3:]
        # total wrench (body frame)
        F_tot_b = F_thr_b + Fd_b + F_buoy_b + F_am_b + F_tether_b
        T_tot_b = T_thr_b + Td_b + T_buoy_b + T_am_b + T_tether_b


        # apply in body frame; write to sim
        self._rov.set_external_force_and_torque(    
            forces=F_tot_b.unsqueeze(1),    # (N,1,3)
            torques=T_tot_b.unsqueeze(1),   # (N,1,3)
            is_global=False
        )
        self._rov.write_data_to_sim()

    def reset(self, env_ids=None):
        zeros = torch.zeros((self._num_envs, 1, 3), device=self._device)
        self._rov.set_external_force_and_torque(zeros, zeros, is_global=False)
        self._rov.write_data_to_sim()


class ROVWrenchActuatorLag(ActionTerm):
    """
    RL → desired body-frame wrench [Fx, Fy, Fz, Tx, Ty, Tz] via raw action.
    Implements actuator dynamics (3rd-order transfer function) as a discrete
    state-space filter so that the actual applied wrench lags/settles realistically.
    """
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._rov       = env.scene[cfg.asset_name]
        self._env       = env
        self._device    = getattr(self._rov, "device", getattr(env, "device", "cpu"))
        self._num_envs  = int(env.num_envs)
        self._action_dim = 6   # assuming 6 channels (3 forces + 3 torques)
        
        # Sample time
        self.dt = env.cfg.decimation * env.cfg.sim.dt
        if self.dt is None:
            raise ValueError("Simulation dt must be defined in cfg.sim.dt")
        
        # Discrete state-space matrices (from your discretised 3rd-order model)
        A_s = torch.tensor([[2.4518, -2.1034, 0.6408],
                            [1.0000,  0.0000,  0.0000],
                            [0.0000,  1.0000,  0.0000]], dtype=torch.float32, device=self._device)
        B_s = torch.tensor([[0.25],
                            [0.0000],
                            [0.0000]], dtype=torch.float32, device=self._device)
        C_s = torch.tensor([[0.269, -0.0075, -0.2184]], dtype=torch.float32, device=self._device)
        D_s = torch.tensor([[0]], dtype=torch.float32, device=self._device)

        # --- Replicate same dynamics for all 6 wrench channels ---
        m = self._action_dim                # 6
        I = torch.eye(m, dtype=torch.float32, device=self._device)

        # Block-diagonal (Kronecker) replication
        self.A = torch.kron(I, A_s)         # (18 x 18)
        self.B = torch.kron(I, B_s)         # (18 x 6)
        self.C = torch.kron(I, C_s)         # ( 6 x 18)
        self.D = torch.kron(I, D_s)         # ( 6 x 6)

        # State buffer per-env
        self.n_states = self.A.shape[0]     # 18
        self._x = torch.zeros((self._num_envs, self.n_states),
                              dtype=torch.float32, device=self._device)
        
        self._raw = torch.zeros((self._num_envs, self._action_dim), dtype=torch.float32, device=self._device)
        self._tau_cmd = torch.zeros((self._num_envs, self._action_dim), dtype=torch.float32, device=self._device)
        self._ctrl = None
    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._tau_cmd

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        if self._ctrl is None:
            self._ctrl = self._env.action_manager.get_term("vel_controller")
        # Input u = self._raw → state update → output tau_cmd
        # x[k+1] = A x[k] + B u[k]
        # y[k]   = C x[k] + D u[k]
        u = self._ctrl.processed_actions  # shape: (num_envs, action_dim)
        
        self._x = torch.matmul(self._x, self.A.T) + torch.matmul(u, self.B.T)

        # Compute output:
        self._tau_cmd = (torch.matmul(self._x, self.C.T) + torch.matmul(u, self.D.T)).clip(-100,100)

# ----------------------------
# Helper Functions & Rewards
# ----------------------------

def quat_mul(a, b):
    # a,b: (...,4) in [x,y,z,w]
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dim=-1)

def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def add_quat_noise(q, sigma_rad=0.1):
    N = q.shape[0]
    device, dtype = q.device, q.dtype
    axis = torch.randn(N, 3, device=device, dtype=dtype)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    theta = torch.randn(N, 1, device=device, dtype=dtype) * sigma_rad
    half = 0.5 * theta
    dq = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=1)
    qn = quat_mul(dq, q)
    return qn / (qn.norm(dim=1, keepdim=True) + 1e-8)

def _stash_term(env, name: str, tensor: torch.Tensor):
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    d = env.extras.get("reward_terms")
    if d is None:
        d = {}
        env.extras["reward_terms"] = d
    d[name] = tensor.detach().cpu().numpy()

def apply_random_wrench_to_target(env, env_ids, force_mean: float, force_std: float):
    tgt = env.scene["target"]
    num_envs = int(env.num_envs)
    device = tgt.device if hasattr(tgt, "device") else env.device
    forces = (torch.randn((num_envs, 1, 3), device=device) * force_std + force_mean)
    torques = torch.zeros((num_envs, 1, 3), device=device)
    tgt.set_external_force_and_torque(forces=forces, torques=torques, is_global=False)
    tgt.write_data_to_sim()

def distance_to_target(env, max_dist : float = 4):
    rov = env.scene["rov"].data.root_state_w[:, :3]
    tgt = env.scene["target"].data.root_state_w[:, :3]
    dist = torch.norm(rov - tgt, dim=-1)
    val = dist / max_dist
    env._rt_distance = val.detach()
    _stash_term(env, "distance_to_target", val)
    return val

def progress_to_target_normalized(env, eps: float = 1e-6):
    rov = env.scene["rov"].data.root_state_w[:, :3]
    tgt = env.scene["target"].data.root_state_w[:, :3]
    dist = torch.norm(rov - tgt, dim=-1)

    prev_dist = getattr(env, "_buf_prev_dist", None)
    prev_rov  = getattr(env, "_buf_prev_rov",  None)
    prev_tgt  = getattr(env, "_buf_prev_tgt",  None)
    if (prev_dist is None or prev_rov is None or prev_tgt is None or prev_dist.shape != dist.shape):
        env._buf_prev_dist = dist.detach().clone()
        env._buf_prev_rov  = rov.detach().clone()
        env._buf_prev_tgt  = tgt.detach().clone()
        prev_dist = env._buf_prev_dist
        prev_rov  = env._buf_prev_rov
        prev_tgt  = env._buf_prev_tgt

    raw = prev_dist - dist
    d_rov = rov - prev_rov
    d_tgt = tgt - prev_tgt
    d_rel = d_rov - d_tgt
    path_len = torch.norm(d_rel, dim=-1).clamp_min(eps)
    prog_norm = (raw / path_len).clamp(-1.0, 1.0)

    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.bool()
    elif hasattr(env, "episode_length_buf"):
        reset_mask = (env.episode_length_buf == 0)
    else:
        reset_mask = torch.zeros_like(prog_norm, dtype=torch.bool)

    if reset_mask.any():
        prog_norm[reset_mask] = 0.0
        env._buf_prev_dist[reset_mask] = dist[reset_mask]
        env._buf_prev_rov[reset_mask]  = rov[reset_mask]
        env._buf_prev_tgt[reset_mask]  = tgt[reset_mask]

    env._buf_prev_dist = dist.detach()
    env._buf_prev_rov  = rov.detach()
    env._buf_prev_tgt  = tgt.detach()
    env._rt_progress_raw   = raw.detach()
    env._rt_progress_norm  = prog_norm.detach()
    env._rt_path_len_rel   = path_len.detach()

    _stash_term(env, "prog_norm", prog_norm)
    return prog_norm

def goal_reached_bool(env, radius: float = 0.5):
    val = (torch.norm(env.scene["rov"].data.root_state_w[:, :3] - env.scene["target"].data.root_state_w[:, :3], dim=-1) <= radius)
    return val

def goal_reached(env, radius: float = 0.5):
    reach = goal_reached_bool(env, radius).float()
    env._rt_success = reach.detach()
    _stash_term(env, "goal_reached", reach)
    return reach

def collision_bool(env, threshold: float = 0.0) -> torch.Tensor:
    cs = env.scene["rov_contact"]
    forces_mag = cs.data.net_forces_w.norm(dim=-1)
    collided = forces_mag > threshold
    return collided.to(dtype=torch.bool).squeeze(-1)

def collision(env, threshold: float = 0.5) -> torch.Tensor:
    val = collision_bool(env, threshold).float()
    env._rt_collision = val.detach()
    return val

def obs_orientation_bodyframe_noise(env, sigma_rad=0.1):
    rov = env.scene["rov"]
    quat_wb = rov.data.root_state_w[:, 3:7]
    quat_bw = quat_conjugate(quat_wb)
    quat_bw = add_quat_noise(quat_bw, sigma_rad)
    return quat_bw

def obs_orientation_bodyframe(env):
    rov = env.scene["rov"]
    quat_wb = rov.data.root_state_w[:, 3:7]
    quat_bw = quat_conjugate(quat_wb)
    return quat_bw

def obs_target_rel_body_noise(env):
    root = env.scene['rov'].data.root_state_w
    pos_w = root[:, :3]
    quat_w = root[:, 3:7]
    tgt_w  = env.scene["target"].data.root_state_w[:, :3]
    rel_w = tgt_w - pos_w
    rel_b = quat_apply_inverse(quat_w, rel_w)
    rel_b = rel_b + torch.randn_like(rel_b) * 0.1
    return rel_b 

def obs_target_rel_body(env):
    root = env.scene['rov'].data.root_state_w
    pos_w = root[:, :3]
    quat_w = root[:, 3:7]
    tgt_w  = env.scene["target"].data.root_state_w[:, :3]
    rel_w = tgt_w - pos_w
    rel_b = quat_apply_inverse(quat_w, rel_w)
    return rel_b 

def effor_x_axis(env):
    w_w = env.scene["rov"].data.root_state_w[:, 10:13]
    quat_wb = env.scene["rov"].data.root_state_w[:, 3:7]
    w_b = quat_apply_inverse(quat_wb, w_w)
    val = w_b[:, 0] ** 2 / 100
    _stash_term(env, "effor_x_axis", val)
    return val

def ang_rate_cost(env):
    w_w = env.scene["rov"].data.root_state_w[:, 10:13]
    quat_wb = env.scene["rov"].data.root_state_w[:, 3:7]
    w_b = quat_apply_inverse(quat_wb, w_w)
    val = (w_b ** 2).sum(-1) / 100
    _stash_term(env, "ang_rate_cost", val)
    return val

def pointing_loss(env, axis=(1., 0., 0.), lam_rate=0.5, huber_delta=0.5, eps=1e-6):
    root = env.scene["rov"].data.root_state_w
    pos  = root[:, :3]
    q    = root[:, 3:7]
    w_w  = root[:, 10:13]
    tgt  = env.scene["target"].data.root_state_w[:, :3]

    cam_off_b = torch.tensor([0.0, 0.0, 0.05506], device=pos.device, dtype=pos.dtype)
    rel_b = quat_apply_inverse(q, tgt - pos) - cam_off_b
    r_hat = rel_b / rel_b.norm(dim=-1, keepdim=True).clamp_min(eps)
    ex  = torch.as_tensor(axis, device=r_hat.device, dtype=r_hat.dtype)
    ex  = ex / ex.norm().clamp_min(eps)
    exN = ex.expand_as(r_hat)

    cos_th = (exN * r_hat).sum(-1).clamp(-1.0, 1.0)
    u = 0.5 * (cos_th + 1.0)
    gamma = 4.0
    align = 2.0 * (u ** gamma) - 1.0

    w_b    = quat_apply_inverse(q, w_w)
    w_perp = w_b - (w_b * exN).sum(-1, keepdim=True) * exN
    vmag   = w_perp.norm(dim=-1)

    d     = torch.as_tensor(huber_delta, device=vmag.device, dtype=vmag.dtype)
    huber = torch.where(vmag <= d, 0.5 * vmag * vmag, d * (vmag - 0.5 * d))
    huber_norm = torch.tanh(huber)
    w_brake = 0.5 * (1.0 + align)
    reward = align - lam_rate * w_brake * huber_norm

    near_target = (rel_b.norm(dim=-1) < 1e-3)
    if near_target.any():
        reward[near_target] = align[near_target]

    _stash_term(env, "pointing_loss", reward)
    return reward

def randomize_global_dome_light(env, env_ids, prim_path="/World/Lighting/DomeLight", intensity_range=(500.0, 5000.0), color_jitter=0.2, temp_range=(2500.0, 8000.0), yaw_range=(-180.0, 180.0), pitch_range=(-45.0, 45.0)):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return
    device = env.device if hasattr(env, "device") else "cpu"
    dome = UsdLux.DomeLight(prim)
    
    intensity = float(torch.empty((), device=device).uniform_(*intensity_range).item())
    dome.CreateIntensityAttr(intensity)

    base_color = torch.ones(3, device=device)
    noise = (torch.rand(3, device=device) - 0.5) * 2.0 * color_jitter
    color = torch.clamp(base_color + noise, 0.0, 2.0).tolist()
    dome.CreateColorAttr(Gf.Vec3f(*color))

    temp = float(torch.empty((), device=device).uniform_(*temp_range).item())
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(temp)

    yaw = float(torch.empty((), device=device).uniform_(*yaw_range).item())
    pitch = float(torch.empty((), device=device).uniform_(*pitch_range).item())
    roll = float(torch.empty((), device=device).uniform_(-10.0, 10.0).item())

    xformable = UsdGeom.Xformable(prim)
    rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_attr = rotate_op.GetAttr()
    rotate_attr.Set(Gf.Vec3f(pitch, yaw, roll))

def randomize_global_sun_light(env, env_ids, prim_path="/World/Lighting/SunLight", intensity_range=(1000.0, 10000.0), color_jitter=0.2, temp_range=(2500.0, 8000.0), yaw_range=(-180.0, 180.0), pitch_range=(10.0, 80.0), move_origin=False, origin_range=((-5.0, 5.0), (-5.0, 5.0), (3.0, 8.0))):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return
    device = env.device if hasattr(env, "device") else "cpu"
    sun = UsdLux.DistantLight(prim)

    intensity = float(torch.empty((), device=device).uniform_(*intensity_range).item())
    sun.CreateIntensityAttr(intensity)

    base_color = torch.ones(3, device=device)
    noise = (torch.rand(3, device=device) - 0.5) * 2.0 * color_jitter
    color = torch.clamp(base_color + noise, 0.0, 2.0).tolist()
    sun.CreateColorAttr(Gf.Vec3f(*color))

    temp = float(torch.empty((), device=device).uniform_(*temp_range).item())
    sun.CreateEnableColorTemperatureAttr(True)
    sun.CreateColorTemperatureAttr(temp)

    yaw = float(torch.empty((), device=device).uniform_(*yaw_range).item())
    pitch = float(torch.empty((), device=device).uniform_(*pitch_range).item())
    roll = float(torch.empty((), device=device).uniform_(-10.0, 10.0).item())

    xformable = UsdGeom.Xformable(prim)
    rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_attr = rotate_op.GetAttr()
    rotate_attr.Set(Gf.Vec3f(pitch, yaw, roll))

    if move_origin:
        tx_attr = prim.GetAttribute("xformOp:translate")
        if not tx_attr:
            tx_op = xformable.AddTranslateOp()
            tx_attr = tx_op.GetAttr()
        (xr, yr, zr) = origin_range
        x = float(torch.empty((), device=device).uniform_(*xr).item())
        y = float(torch.empty((), device=device).uniform_(*yr).item())
        z = float(torch.empty((), device=device).uniform_(*zr).item())
        tx_attr.Set(Gf.Vec3d(x, y, z))

# ----------------------------
# Configurations
# ----------------------------

@configclass
class ROVSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Lighting/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )
    sun_light = AssetBaseCfg(
        prim_path="/World/Lighting/SunLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.98, 0.95), intensity=40000.0, angle=0.53),
    )
    pool = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pool",
        spawn=sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/ROV/pool_water.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0))
    )
    rov = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ROV",
        spawn=sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/ROV/BROV_low.usd", activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0.0)),
    )
    rov_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ROV",
        update_period=0.0, history_length=1, debug_vis=True,
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/ROV/Camera/UW_camera",
        spawn=None,
        data_types=["rgb"],
        width=720, height=1280, update_period=0.005*7,
    )

@configclass
class ActionsCfg:
    vel_controller  = ActionTermCfg(class_type=ROVWrenchController, asset_name="rov")
    lagg = ActionTermCfg(class_type=ROVWrenchActuatorLag, asset_name="rov")
    hydro      = ActionTermCfg(class_type=ROVHydroApplier,     asset_name="rov")

@configclass
class EventCfg:
    reset_pose = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("rov"), "pose_range": {"x": (-1.3,-1), "y": (-1,1), "z": (0.4,0.6)}, "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)}},
    )
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("target"), "pose_range": {"x": (0.8,3), "y": (-0.6,0.6), "z": (0.4,0.8)}, "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)}},
    )
    pool = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("pool"), "pose_range": {"x": (0,0), "y": (0,0), "z": (0,0)}, "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)}},
    )
    move_target_rw = EventTerm(
        func=apply_random_wrench_to_target, mode="interval", interval_range_s=(0.1, 0.1),
        params={"force_mean": 0, "force_std": 0.5},
    )   
    randomize_dome = EventTerm(
        func=randomize_global_dome_light, mode="interval", interval_range_s=(1, 2),
        params={"prim_path": "/World/Lighting/DomeLight", "intensity_range": (500.0, 1500.0), "color_jitter": 0.2},
    )
    randomize_sun = EventTerm(
        func=randomize_global_sun_light, mode="interval", interval_range_s=(1, 2),
        params={"prim_path": "/World/Lighting/SunLight", "intensity_range": (1000.0, 3000.0), "color_jitter": 0.2},
    )
    randomize_color_target = EventTerm(
        func=mdp.randomize_visual_color, mode="interval", interval_range_s=(0.0, 2),
        params={"colors": {"r": (0, 0.078), "g": (0, 0.078), "b": (0, 0.078)}, "asset_cfg": SceneEntityCfg("target"), "event_name": "rep_cube_randomize_color_target"},
    )

@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        ori = ObsTerm(func=obs_orientation_bodyframe_noise)
        dist = ObsTerm(func=obs_target_rel_body_noise)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: Policy = Policy()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=-2)
    progress = RewTerm(func=progress_to_target_normalized, weight=2.5)
    distance = RewTerm(func=distance_to_target, weight=-1.2)
    termination = RewTerm(func=mdp.is_terminated, weight=-20)  
    success  = RewTerm(func=goal_reached, weight=180)
    line_of_sight = RewTerm(func=pointing_loss, weight=5)
    stability = RewTerm(func=ang_rate_cost, weight=-0.5)
    stability_x = RewTerm(func=effor_x_axis, weight=-6)

@configclass
class TerminationsCfg:
    time_out       = DoneTerm(func=mdp.time_out, time_out=True)
    reached_target = DoneTerm(func=goal_reached_bool)
    collision      = DoneTerm(func=collision_bool)

@configclass
class ROVEnvCfg(ManagerBasedRLEnvCfg):
    scene = ROVSceneCfg(num_envs=args_cli.num_envs, env_spacing=10, replicate_physics=False)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    def __post_init__(self):
        self.decimation = 7
        self.sim.dt = 0.005 
        self.viewer.eye = [4.5, 0.0, 3.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.episode_length_s = 10.0
        physx = self.sim.physx
        physx.use_gpu = True
        physx.use_gpu_pipeline = True
        physx.gpu_max_rigid_contact_count   = 6_300_000
        physx.gpu_max_rigid_patch_count     = 1_600_000
        physx.gpu_found_lost_pairs_capacity = 5_000_000
        physx.gpu_heap_capacity             = 512 * 1024 * 1024
        physx.gpu_temp_buffer_capacity      = 256 * 1024 * 1024
        physx.gpu_persistent_contact_stream_capacity = 128 * 1024 * 1024
        physx.gpu_max_soft_body_contacts    = 0
        physx.gpu_max_particle_contacts     = 0
        self.sim.render = sim_utils.RenderCfg(enable_translucency=True)
