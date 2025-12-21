# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import time

import math, torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sim import schemas
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, CameraCfg, TiledCameraCfg
from stable_baselines3 import PPO, SAC


import numpy as np
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback


import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan


from uw_renderer_utils import UW_render  # this is the file you just copied

import warp as wp
from pxr import UsdLux, UsdGeom, Gf
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
import torch.nn.functional as F

from isaaclab.utils import configclass  # if you use @configclass
# ----------------------------
# Scene: single-body ROV asset
# ----------------------------
@configclass
class ROVSceneCfg(InteractiveSceneCfg):

    



    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )



    dome_light = AssetBaseCfg(
        prim_path="/World/Lighting/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=3000.0,
        ),
    )

    # single global distant "sun" light
    sun_light = AssetBaseCfg(
        prim_path="/World/Lighting/SunLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.98, 0.95),
            intensity=40000.0,
            angle=0.53,
        ),
    )


    # --- NEW: pool (static geometry) ---
    pool = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pool",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/ROV/pool_water.usd",  # <- your pool file
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 0),
        )
    )


    rov = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ROV",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/ROV/BROV_low.usd",
            activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),rot=(1, 0, 0, 0.0)),
    )

    rov_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ROV",  # attaches to this prim and its rigid bodies
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )

    target = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Target",
    spawn=sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),  # 20 cm cube
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),
            metallic=0.2,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
)


   
    front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/ROV/Camera/UW_camera",
        spawn=None,          # again, use existing USD camera
        data_types=["rgb"],
        width=720,
        height=1280,
        update_period=0.005*7,
    )
    
# ----------------------------
# Observation callables (no joints)
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
    Args:
        v_b: (N,3) linear velocity [u,v,w]
        w_b: (N,3) angular velocity [p,q,r]
        A_lin: (3,) [X_udot, Y_vdot, Z_wdot]
        A_rot: (3,) [K_pdot, M_qdot, N_rdot]
    Returns:
        C_A: (N,6,6) skew-symmetric matrix
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
        #force to pwn
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
        # You must fill these in (size matches action dimension)
        # Example assuming each channel uses same 3-state model:
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






@configclass
class ActionsCfg:
    #model = ActionTermCfg(class_type=NoMode, asset_name='rov')
    vel_controller  = ActionTermCfg(class_type=ROVWrenchController, asset_name="rov")
    lagg = ActionTermCfg(class_type=ROVWrenchActuatorLag, asset_name="rov")
    hydro      = ActionTermCfg(class_type=ROVHydroApplier,     asset_name="rov")




def apply_random_wrench_to_target(env,
    env_ids,
    force_mean: float,
    force_std: float,

    ):
    tgt      = env.scene["target"]
    num_envs = int(env.num_envs)
    device   = tgt.device if hasattr(tgt, "device") else env.device

    # Sample forces (mean zero)
    forces = (torch.randn((num_envs, 1, 3), device=device) * force_std + force_mean)

    # Sample torques
    torques = torch.zeros((num_envs, 1, 3), device=device)

    # Apply wrench in world‐frame
    tgt.set_external_force_and_torque(
        forces   = forces,
        torques  = torques,
        is_global= False,
    )

    # Commit to sim
    tgt.write_data_to_sim()


def distance_to_target(env, max_dist : float = 4):
    rov = env.scene["rov"].data.root_state_w[:, :3]
    tgt = env.scene["target"].data.root_state_w[:, :3]
    dist = torch.norm(rov - tgt, dim=-1)
    val = dist / max_dist                      # raw term (before weighting)
    env._rt_distance = val.detach()        # <-- tap the exact value used
    _stash_term(env, "distance_to_target", val)
    return val

def progress_to_target_normalized(env, eps: float = 1e-6):
    """
    Normalized progress in [-1, 1] using RELATIVE displacement:
      +1  -> moved straight toward target
       0  -> no net closure (sideways / noise)
      -1  -> moved straight away
    Robust to target motion, substeps, and dt/velocity mismatches.
    """
    # Current positions
    rov = env.scene["rov"].data.root_state_w[:, :3]
    tgt = env.scene["target"].data.root_state_w[:, :3]

    # Current distance
    dist = torch.norm(rov - tgt, dim=-1)

    # Init buffers on first call or shape change
    prev_dist = getattr(env, "_buf_prev_dist", None)
    prev_rov  = getattr(env, "_buf_prev_rov",  None)
    prev_tgt  = getattr(env, "_buf_prev_tgt",  None)
    if (
        prev_dist is None or prev_rov is None or prev_tgt is None
        or prev_dist.shape != dist.shape
    ):
        env._buf_prev_dist = dist.detach().clone()
        env._buf_prev_rov  = rov.detach().clone()
        env._buf_prev_tgt  = tgt.detach().clone()
        prev_dist = env._buf_prev_dist
        prev_rov  = env._buf_prev_rov
        prev_tgt  = env._buf_prev_tgt

    # Raw progress (meters): reduction in distance
    raw = prev_dist - dist

    # Relative displacement over the sampling interval
    d_rov = rov - prev_rov         # how much the ROV moved
    d_tgt = tgt - prev_tgt         # how much the target moved
    d_rel = d_rov - d_tgt          # relative motion ROV vs target
    path_len = torch.norm(d_rel, dim=-1).clamp_min(eps)

    # Normalized progress in [-1, 1] up to numerical noise
    prog_norm = (raw / path_len).clamp(-1.0, 1.0)

    # Handle resets (zero out and re-seed buffers)
    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.bool()
    elif hasattr(env, "episode_length_buf"):
        reset_mask = (env.episode_length_buf == 0)
    else:
        reset_mask = torch.zeros_like(prog_norm, dtype=torch.bool)

    if reset_mask.any():
        prog_norm[reset_mask] = 0.0
        dist_r = dist[reset_mask]
        env._buf_prev_dist[reset_mask] = dist_r
        env._buf_prev_rov[reset_mask]  = rov[reset_mask]
        env._buf_prev_tgt[reset_mask]  = tgt[reset_mask]

    # Update buffers
    env._buf_prev_dist = dist.detach()
    env._buf_prev_rov  = rov.detach()
    env._buf_prev_tgt  = tgt.detach()

    # Tap for logging
    env._rt_progress_raw   = raw.detach()
    env._rt_progress_norm  = prog_norm.detach()
    env._rt_path_len_rel   = path_len.detach()

    _stash_term(env, "prog_norm", prog_norm)
    return prog_norm


def _stash_term(env, name: str, tensor: torch.Tensor):
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    d = env.extras.get("reward_terms")
    if d is None:
        d = {}
        env.extras["reward_terms"] = d
    d[name] = tensor.detach().cpu().numpy()

def goal_reached_bool(env, radius: float = 0.5):
    val = (torch.norm(env.scene["rov"].data.root_state_w[:, :3] - env.scene["target"].data.root_state_w[:, :3], dim=-1) <= radius)
           # <-- tap success term
    return val


def goal_reached(env, radius: float = 0.5):
    reach = goal_reached_bool(env, radius).float()
    env._rt_success = reach.detach()
    _stash_term(env, "goal_reached", reach)
    return reach


def collision_bool(env, threshold: float = 0.0) -> torch.Tensor:
    """
    Boolean per-env: True if any contact force magnitude > threshold.
    Assumes contact sensor key 'rov_contact'.
    """
    cs = env.scene["rov_contact"]
    # net_forces_w: (N,3)
    forces_mag = cs.data.net_forces_w.norm(dim=-1)    # (N,)
    collided = forces_mag > threshold
    return collided.to(dtype=torch.bool).squeeze(-1)


def collision(env, threshold: float = 0.5) -> torch.Tensor:
    """
    Boolean per-env: True if any contact force magnitude > threshold.
    Assumes contact sensor key 'rov_contact'.
    """
    val = collision_bool(env, threshold).float()
    env._rt_collision = val.detach()
    return val


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

def add_quat_noise(q, sigma_rad=0.1):
    """
    q: (N,4) unit quaternion [x,y,z,w]
    sigma_rad: std-dev of noise angle in radians (e.g. 0.05 ~ 3 deg)
    """
    N = q.shape[0]
    device, dtype = q.device, q.dtype

    # random axis (unit)
    axis = torch.randn(N, 3, device=device, dtype=dtype)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)

    # small random angle
    theta = torch.randn(N, 1, device=device, dtype=dtype) * sigma_rad

    # axis-angle -> quaternion noise dq = [axis*sin(theta/2), cos(theta/2)]
    half = 0.5 * theta
    dq = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=1)

    # compose rotations: dq ⊗ q  (global/world-frame perturbation)
    qn = quat_mul(dq, q)
    return qn / (qn.norm(dim=1, keepdim=True) + 1e-8)




def obs_orientation_bodyframe_noise(env, sigma_rad=0.1):
    rov = env.scene["rov"]
    quat_wb = rov.data.root_state_w[:, 3:7]   # world->body
    quat_bw = quat_conjugate(quat_wb)         # body->world

    quat_bw = add_quat_noise(quat_bw, sigma_rad)
    return quat_bw


def obs_orientation_bodyframe(env):
    """
    Orientation of world expressed in the ROV's body frame (body→world inverse).
    """
    rov = env.scene["rov"]
    quat_wb = rov.data.root_state_w[:, 3:7]      # world→body
    quat_bw = quat_conjugate(quat_wb)            # body→world

    noise = torch.randn_like(quat_bw) * 0.1

    return quat_bw



def obs_target_rel_body_noise(env):
    root = env.scene['rov'].data.root_state_w
    pos_w = root[:, :3]
    quat_w = root[:, 3:7]
    tgt_w  = env.scene["target"].data.root_state_w[:, :3]

    rel_w = tgt_w - pos_w
    rel_b = quat_apply_inverse(quat_w, rel_w)   # world → body

    rel_b = rel_b + torch.randn_like(rel_b) * 0.1
    return rel_b 


def obs_target_rel_body(env):
    root = env.scene['rov'].data.root_state_w
    pos_w = root[:, :3]
    quat_w = root[:, 3:7]
    tgt_w  = env.scene["target"].data.root_state_w[:, :3]

    rel_w = tgt_w - pos_w
    rel_b = quat_apply_inverse(quat_w, rel_w)   # world → body
    return rel_b 


def effor_x_axis(env):
    w_w = env.scene["rov"].data.root_state_w[:, 10:13]   # angular velocity (world)
    quat_wb = env.scene["rov"].data.root_state_w[:, 3:7] # orientation (world→body)
    
    # rotate angular velocity to body frame
    w_b = quat_apply_inverse(quat_wb, w_w)
    
    # penalize only rotation around body X-axis (roll)
    val = w_b[:, 0] ** 2 / 100
    
    _stash_term(env, "effor_x_axis", val)
    return val


def ang_rate_cost(env):
    w_w = env.scene["rov"].data.root_state_w[:, 10:13]  # ω in world frame
    quat_wb = env.scene["rov"].data.root_state_w[:, 3:7]  # world->body rotation (xyzw)
    
    # rotate angular velocity to body frame
    w_b = quat_apply_inverse(quat_wb, w_w)  # depends on your framework; usually does q⁻¹ * w * q
    val = (w_b ** 2).sum(-1) / 100
    _stash_term(env, "ang_rate_cost", val)
    return val


def pointing_loss(env, axis=(1., 0., 0.), lam_rate=0.5, huber_delta=0.5, eps=1e-6):
    root = env.scene["rov"].data.root_state_w
    pos  = root[:, :3]
    q    = root[:, 3:7]          # xyzw
    w_w  = root[:, 10:13]
    tgt  = env.scene["target"].data.root_state_w[:, :3]

    cam_off_b = torch.tensor([0.0, 0.0, 0.05506],
                             device=pos.device, dtype=pos.dtype)

    # body-frame LOS
    rel_b = quat_apply_inverse(q, tgt - pos) - cam_off_b
    r_hat = rel_b / rel_b.norm(dim=-1, keepdim=True).clamp_min(eps)

    # chosen body axis
    ex  = torch.as_tensor(axis, device=r_hat.device, dtype=r_hat.dtype)
    ex  = ex / ex.norm().clamp_min(eps)
    exN = ex.expand_as(r_hat)

    # raw cosine alignment [-1, 1]
    cos_th = (exN * r_hat).sum(-1).clamp(-1.0, 1.0)

    # --- Option 1: power-sharpened alignment ---
    # map to [0, 1], sharpen, then map back to [-1, 1]
    u = 0.5 * (cos_th + 1.0)        # [-1,1] -> [0,1]
    gamma = 4.0                     # tune: 3–8; higher = narrower peak
    align = 2.0 * (u ** gamma) - 1.0  # back to [-1,1], but much sharper near 1

    # angular rate that changes pointing (remove roll about ex)
    w_b    = quat_apply_inverse(q, w_w)
    w_perp = w_b - (w_b * exN).sum(-1, keepdim=True) * exN
    vmag   = w_perp.norm(dim=-1)

    # Huber on ||w_perp||, then squash so it’s bounded
    d     = torch.as_tensor(huber_delta, device=vmag.device, dtype=vmag.dtype)
    huber = torch.where(vmag <= d, 0.5 * vmag * vmag, d * (vmag - 0.5 * d))
    huber_norm = torch.tanh(huber)                     # (0,1)

    # brake weight: stronger near (sharpened) alignment
    w_brake = 0.5 * (1.0 + align)                      # [0,1]

    # FINAL: reward = alignment - angle-weighted rate penalty
    reward = align - lam_rate * w_brake * huber_norm

    # near-target: keep alignment reward; zero only the rate penalty
    near_target = (rel_b.norm(dim=-1) < 1e-3)
    if near_target.any():
        reward[near_target] = align[near_target]

    _stash_term(env, "pointing_loss", reward)
    return reward


def randomize_global_dome_light(
    env,
    env_ids: torch.Tensor,
    prim_path: str = "/World/Lighting/DomeLight",
    intensity_range=(500.0, 5000.0),
    color_jitter: float = 0.2,
    temp_range=(2500.0, 8000.0),
    yaw_range=(-180.0, 180.0),
    pitch_range=(-45.0, 45.0),
):
    """Randomize a SINGLE global DomeLight.

    - intensity
    - color + color temperature
    - orientation (yaw/pitch/roll) via xformOp:rotateXYZ
    Position is irrelevant for dome, so I leave it alone.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Dome light prim not found at '{prim_path}'")

    device = env.device if hasattr(env, "device") else "cpu"

    dome = UsdLux.DomeLight(prim)

    # --- Photometric: intensity ---
    intensity = float(
        torch.empty((), device=device).uniform_(*intensity_range).item()
    )
    dome.CreateIntensityAttr(intensity)

    # --- Photometric: color + temperature ---
    base_color = torch.ones(3, device=device)
    noise = (torch.rand(3, device=device) - 0.5) * 2.0 * color_jitter
    color = torch.clamp(base_color + noise, 0.0, 2.0).tolist()
    dome.CreateColorAttr(Gf.Vec3f(*color))

    # Enable color temperature and randomize it
    temp = float(
        torch.empty((), device=device).uniform_(*temp_range).item()
    )
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(temp)

    # --- Geometric: orientation via rotateXYZ ---
    # We treat yaw (around Z) and pitch (around X) in degrees
    yaw = float(
        torch.empty((), device=device).uniform_(*yaw_range).item()
    )
    pitch = float(
        torch.empty((), device=device).uniform_(*pitch_range).item()
    )
    roll = float(
        torch.empty((), device=device).uniform_(-10.0, 10.0).item()
    )

    xformable = UsdGeom.Xformable(prim)
    # Try to reuse existing rotateXYZ op, otherwise create once
    rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_attr = rotate_op.GetAttr()
    rotate_attr.Set(Gf.Vec3f(pitch, yaw, roll))



def randomize_global_sun_light(
    env,
    env_ids: torch.Tensor,
    prim_path: str = "/World/Lighting/SunLight",
    intensity_range=(1000.0, 10000.0),
    color_jitter: float = 0.2,
    temp_range=(2500.0, 8000.0),
    yaw_range=(-180.0, 180.0),
    pitch_range=(10.0, 80.0),
    move_origin=False,
    origin_range=((-5.0, 5.0), (-5.0, 5.0), (3.0, 8.0)),
):
    """Randomize a SINGLE global DistantLight (sun).

    - intensity
    - color + color temperature
    - orientation (yaw/pitch/roll)
    - optional translation (physically irrelevant for distant light, but you can set it)
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Sun light prim not found at '{prim_path}'")

    device = env.device if hasattr(env, "device") else "cpu"

    sun = UsdLux.DistantLight(prim)

    # --- Photometric: intensity ---
    intensity = float(
        torch.empty((), device=device).uniform_(*intensity_range).item()
    )
    sun.CreateIntensityAttr(intensity)

    # --- Photometric: color + temperature ---
    base_color = torch.ones(3, device=device)
    noise = (torch.rand(3, device=device) - 0.5) * 2.0 * color_jitter
    color = torch.clamp(base_color + noise, 0.0, 2.0).tolist()
    sun.CreateColorAttr(Gf.Vec3f(*color))

    temp = float(
        torch.empty((), device=device).uniform_(*temp_range).item()
    )
    sun.CreateEnableColorTemperatureAttr(True)
    sun.CreateColorTemperatureAttr(temp)

    # --- Geometric: orientation (direction) ---
    yaw = float(
        torch.empty((), device=device).uniform_(*yaw_range).item()
    )
    pitch = float(
        torch.empty((), device=device).uniform_(*pitch_range).item()
    )
    roll = float(
        torch.empty((), device=device).uniform_(-10.0, 10.0).item()
    )

    xformable = UsdGeom.Xformable(prim)

    rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_attr = rotate_op.GetAttr()
    rotate_attr.Set(Gf.Vec3f(pitch, yaw, roll))

    # --- Optional: move origin (only cosmetic for distant light) ---
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








@configclass
class EventCfg:
    reset_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("rov"),
            "pose_range": {"x": (-1.3,-1), "y": (-1,1), "z": (0.4,0.6)},
            "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)},
        },
    )

    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {"x": (0.8,3), "y": (-0.6,0.6), "z": (0.4,0.8)},
            "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)},
        },
    )

    pool = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={  
            "asset_cfg": SceneEntityCfg("pool"),
            "pose_range": {"x": (0,0), "y": (0,0), "z": (0,0)},
            "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0)},
        },
    )

    move_target_rw = EventTerm(
    func           = apply_random_wrench_to_target,
    mode           = "interval",
    interval_range_s= (0.1, 0.1),
    params         = {
        "force_mean": 0,
        "force_std": 0.5,
    }
    )   

    randomize_dome = EventTerm(
        func=randomize_global_dome_light,
        mode="interval",
        interval_range_s= (1, 2),
        params={
            "prim_path": "/World/Lighting/DomeLight",
            "intensity_range": (500.0, 1500.0),
            "color_jitter": 0.2,
        },
    )


    randomize_sun = EventTerm(
        func=randomize_global_sun_light,
        mode="interval",
        interval_range_s= (1, 2),
        params={
            "prim_path": "/World/Lighting/SunLight",
            "intensity_range": (1000.0, 3000.0),
            "color_jitter": 0.2,
        },
    )


    randomize_color_target = EventTerm(
    func=mdp.randomize_visual_color,
    mode="interval",
    interval_range_s= (0.0, 2),
    params={
            "colors": {
            "r": (0, 0.078),
            "g": (0, 0.078),
            "b": (0, 0.078),
        },
        "asset_cfg": SceneEntityCfg("target"),
        "event_name": "rep_cube_randomize_color_target",
    },
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
    alive = RewTerm(func=mdp.is_alive,                          weight=-2)
    progress = RewTerm(func=progress_to_target_normalized,      weight=2.5)
    distance = RewTerm(func=distance_to_target,                 weight=-1.2)
    termination = RewTerm(func=mdp.is_terminated,               weight=-20)  
    success  = RewTerm(func=goal_reached,                       weight=180)
    line_of_sight = RewTerm(func=pointing_loss,                     weight=5)
    stability = RewTerm(func=ang_rate_cost,                         weight=-0.5)
    stability_x = RewTerm(func=effor_x_axis,                        weight=-6)


@configclass
class TerminationsCfg:
    # Always good: end after max steps
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
        self.decimation = 7    # 200 Hz sim / 4 = 50 Hz RL
        self.sim.dt = 0.005 
        self.viewer.eye = [4.5, 0.0, 3.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.episode_length_s = 10.0
        physx = self.sim.physx  # isaaclab.sim.schemas.PhysxCfg
        physx.use_gpu = True
        physx.use_gpu_pipeline = True

        # From your earlier floors (~10k env): contacts ~5.67M, patches ~200–210k
        # Scale + headroom, but stay sane:
        physx.gpu_max_rigid_contact_count   = 6_300_000    # ~64 B each ≈ 2.56 GB
        physx.gpu_max_rigid_patch_count     = 1_600_000     # ~32 B each ≈ 45 MB

        # Pairs: keep modest; your 30M attempt cost ~457 MB and helped trigger OOM.
        # Start lower and let logs tell you if to bump:
        physx.gpu_found_lost_pairs_capacity = 5_000_000    # ~16 B each ≈ 305 MB

        # Big pools: avoid 1 GB chunks; keep them moderate
        physx.gpu_heap_capacity             = 512 * 1024 * 1024   # 512 MB
        physx.gpu_temp_buffer_capacity      = 256 * 1024 * 1024   # 256 MB
        physx.gpu_persistent_contact_stream_capacity = 128 * 1024 * 1024

        # If you don't use these features:
        physx.gpu_max_soft_body_contacts    = 0
        physx.gpu_max_particle_contacts     = 0

        self.sim.render = sim_utils.RenderCfg(
            # optional: choose mode; or leave None and use CLI --rendering_mode
            enable_translucency=True,        # <- glass / water etc. actually translucent
      
        )

class RewardTermsRunningAvgPrinter(BaseCallback):
    """
    Keeps a running (since training start) average of selected reward_terms across ALL envs.
    Prints/logs every `every_n` steps. Ignores episode terminations completely.
    """
    def __init__(self, every_n=500, keys=None):
        super().__init__()
        self.every_n = every_n
        self.keys = set(keys) if keys else None
        self._sum = {}    # dict: term -> scalar running sum
        self._count = {}  # dict: term -> integer count of samples accumulated

    def _accumulate(self, term, val):
        # val: float
        if term not in self._sum:
            self._sum[term] = 0.0
            self._count[term] = 0
        self._sum[term] += float(val)
        self._count[term] += 1

    def _on_step(self) -> bool:
        # Pull per-env extras and accumulate running stats
        if self.num_timesteps % self.every_n != 0:
            return True
        extras_list = self.training_env.get_attr("extras", indices=None)

        if extras_list is not None:
            for extras in extras_list:
                if not extras:
                    continue
                rt = extras.get("reward_terms", {})
                for k, v in rt.items():
                    if (self.keys is None) or (k in self.keys):
                        # v could be a scalar np array or vector per env; take mean to get one number
                        self._accumulate(k, np.mean(v))

        # Print / log every_n steps
        if self.num_timesteps % self.every_n == 0 and self._sum:
            parts = []
            for k in sorted(self._sum.keys()):
                if self._count[k] > 0:
                    avg = self._sum[k] / self._count[k]
                    parts.append(f"{k}: {avg:.3f}")
                    self.logger.record(f"reward_terms_running_mean/{k}", float(avg))
            if parts:
                print(f"[t={self.num_timesteps}] " + " | ".join(parts))

        return True

def play_ppo_policy(env_cfg, model_path, n_episodes=5, render=False):
    # build the Isaac environment + wrapper
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    vec_env = Sb3VecEnvWrapper(isaac_env)
    # optionally consistency check wrapper if you used it
    # vec_env = CheckRewardConsistency(vec_env, isaac_env, env_idx=0)

    # Load the PPO model
    model = PPO.load(model_path, env=vec_env, device="cuda")

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while True:
            # get action (deterministic for inference)
            action, _states = model.predict(obs, deterministic=True)
            # step the environment
            obs, reward, dones, infos = vec_env.step(action)
            
            ep_reward += np.sum(reward)  # or handle vector env sum
            ep_length += 1

            if render:
                # If your wrapper / env supports rendering
                vec_env.render()

            # In vectorized envs, “dones” is an array; break when all are done or some logic
            if isinstance(dones, (list, tuple, np.ndarray)):
                if all(dones):
                    break
            else:
                if dones:
                    break

        print(f"Episode {ep}: length = {ep_length}, reward = {ep_reward}")

    # Close envs
    vec_env.close()
    isaac_env.close()

def main_sac():
    from stable_baselines3 import PPO, SAC
    # --- build Isaac RL env ---
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)   # num_envs > 1 is fine (SAC supports VecEnv)
    vec_env = Sb3VecEnvWrapper(isaac_env)
    vec_env = VecCheckNan(vec_env, raise_exception=True)
    # vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    # === SAC (for continuous actions) ===
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        device="cuda",
        # Replay + updates
        buffer_size=5_000_000,
        batch_size=8048,
        n_steps=10,
        ent_coef='auto',
        learning_rate=3e-3,
        learning_starts=500_000,
        train_freq=1,
        gradient_steps=20,
        tau=0.001,
        gamma=0.99,
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),
            log_std_init=-1.0,
        ),
        verbose=0,
    )

    # === Callbacks ===
    # 1) Your running-average printer
    log_cb = RewardTermsRunningAvgPrinter(
        every_n=1_000_000,
        keys={"pointing_loss", "effor_x_axis", "ang_rate_cost",
              "goal_reached", "prog_norm", "distance_to_target"},
    )

    # 2) Periodic checkpoint saving
    # save_freq is in *env steps* (across all parallel envs)
    ckpt_dir = "./checkpoints_sac_rov"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000_000,          # save every 1M steps
        save_path=ckpt_dir,
        name_prefix="sac_rov",        # files like sac_rov_1000000_steps.zip
        save_replay_buffer=False,     # set True if you want the buffer too (big!)
        save_vecnormalize=False,      # set True if you use VecNormalize above
    )

    # Combine both callbacks
    callback = CallbackList([log_cb, checkpoint_cb])

    start = time.time()
    try:
        model.learn(
            total_timesteps=500_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        # If you kill it manually, at least save the last state
        print("Interrupted, saving last checkpoint to sac_rov_policy_interrupt.zip")
        model.save("sac_rov_policy_interrupt")

    end = time.time()
    elapsed = end - start
    print(f"Training finished in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds).")

    # Final "best guess" model at the end
    model.save("sac_rov_policy_28hz_3")
    vec_env.close()





def play_sac_policy():
    # --- build Isaac RL env (same config as training) ---
    from stable_baselines3 import PPO, SAC

    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    vec_env = Sb3VecEnvWrapper(isaac_env)

    # --- load trained model ---
    model = SAC.load("sac_rov_policy_28hz_2.zip", env=vec_env, device="cuda")

    obs = vec_env.reset()
    episode_rewards = [0.0 for _ in range(vec_env.num_envs)]

    print("Playing policy... Press Ctrl+C to stop.")

    while True:
        # deterministic=True → no exploration noise
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        # accumulate reward for logging
        for i in range(vec_env.num_envs):
            episode_rewards[i] += rewards[i]
            if dones[i]:
                print(f"Env {i} episode reward: {episode_rewards[i]:.3f}")
                episode_rewards[i] = 0.0

        # optional: slow down visualization if rendering in real time
        # time.sleep(0.02)

    vec_env.close()

import torch as th
from torch import nn

class ImageEncoder(nn.Module):
    """
    Simple CNN encoder for Isaac camera images.
    Input:  (N, C, H, W)
    Output: (N, latent_dim)
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (N, 64, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                   # -> (N, 64*4*4 = 1024)
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        z = self.fc(x)
        return z


class StudentGaussianPolicy(nn.Module):
    """
    Image encoder + orientation -> Gaussian action policy.
    Outputs:
      - mu, log_std     : action distribution (what SAC cares about)
      - teacher_mu, teacher_log_std : predicted state distribution (only you use)
    """

    LOG_STD_MIN_ACTION = -4.0   # std ≈ 0.018
    LOG_STD_MAX_ACTION =  1.0   # std ≈ 7.39  (or even 3.0 → std ≈ 20)

    LOG_STD_MIN_STATE  = -4.0
    LOG_STD_MAX_STATE  =  -1   # std ≈ 1.0 is safely above your 0.3

    def __init__(self, action_dim, env, in_channels=1,
                 latent_dim_img=128, ori_dim=4):
        super().__init__()

        self.encoder = ImageEncoder(in_channels=in_channels,
                                    latent_dim=latent_dim_img)

        feat_dim = latent_dim_img + ori_dim   # this must match SB3 features_dim

        # == this is what SB3's actor.latent_pi mimics ==
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        state_dim = env.observation_space.shape[0]

        # extra heads for teacher-state prediction (BC loss only)
        self.teacher_state_mu  = nn.Linear(feat_dim, state_dim)
        self.teacher_state_log_std = nn.Linear(feat_dim, state_dim)

        # == these are what SB3's actor.mu / actor.log_std mimic ==
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, images, ori):
        """
        images: (N, C, H, W)
        ori:    (N, 4)
        Returns:
            mu:              (N, act_dim)
            log_std:         (N, act_dim)
            teacher_mu:      (N, state_dim)
            teacher_log_std: (N, state_dim)
        """
        z_img = self.encoder(images)          # (N, latent_dim_img)
        z = th.cat([z_img, ori], dim=1)       # (N, feat_dim)

        h = self.backbone(z)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = th.clamp(log_std, self.LOG_STD_MIN_ACTION, self.LOG_STD_MAX_ACTION)

        teacher_mu = self.teacher_state_mu(z)
        teacher_log_std = th.clamp(
            self.teacher_state_log_std(z),
            self.LOG_STD_MIN_STATE,
            self.LOG_STD_MAX_STATE,
        )

        return mu, log_std, teacher_mu, teacher_log_std

    def act(self, images, ori, deterministic=False):
        mu, log_std, _, _ = self(images, ori)
        if deterministic:
            return mu

        std = log_std.exp()
        eps = th.randn_like(mu)
        return mu + eps * std



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
    uw_rgb = uw[..., :3].to(th.float32) / 255.0    # (H, W, 3)

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

    uw_batch = th.stack(uw_list, dim=0)
    return uw_batch.permute(0, 3, 1, 2)



import torch as th

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
    rgb = rgb_batch[..., :3].to(th.float32)  # (N,H,W,3) in [0,255]

    # depth to (N,H,W,1) for broadcasting
    d = depth_batch.unsqueeze(-1).to(th.float32)  # (N,H,W,1)

    # params as (1,1,1,3) for broadcasting
    backscatter_value = th.tensor(backscatter_value, dtype=th.float32, device=device).view(1,1,1,3)
    atten_coeff       = th.tensor(atten_coeff,       dtype=th.float32, device=device).view(1,1,1,3)
    backscatter_coeff = th.tensor(backscatter_coeff, dtype=th.float32, device=device).view(1,1,1,3)

    # exp_atten = exp(- depth * atten_coeff)
    exp_atten = th.exp(-d * atten_coeff)          # (N,H,W,3)

    # exp_back = exp(- depth * backscatter_coeff)
    exp_back = th.exp(-d * backscatter_coeff)     # (N,H,W,3)

    # UW_RGB = raw_RGB * exp_atten + backscatter_value*255 * (1 - exp_back)
    uw_rgb = rgb * exp_atten \
           + backscatter_value * 255.0 * (1.0 - exp_back)

    # clamp 0..255 and cast back to uint8
    uw_rgb = th.clamp(uw_rgb, 0, 255)
    uw_rgb_u8 = uw_rgb.to(th.uint8)

    return uw_rgb_u8.permute(0, 3, 1, 2)




def rgb_to_gray_torch(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (B,3,H,W) or (B,4,H,W) -> (B,1,H,W)
    if rgb.size(1) == 4:
        rgb = rgb[:, :3]
    w = rgb.new_tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
    gray = (rgb.to(torch.float32) * w).sum(dim=1, keepdim=True)
    return gray


def train_student():
    from stable_baselines3 import SAC
    import torch as th
    from collections import deque
    import random
    import math                                        # NEW
    device = th.device("cuda")

    # 1) Env + expert
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    venv = Sb3VecEnvWrapper(isaac_env)
 
    expert = SAC.load(
        "sac_rov_policy_28hz_2.zip",
        env=venv,
        device="cuda",
    )
    encoder = ImageEncoder()
    # camera handle (attribute, not dict)
    cam = isaac_env.scene['front_cam']

    # action dimensions
    act_space = venv.action_space
    action_dim = act_space.shape[0]

    # 2) Student network (encoder + head)
    student = StudentGaussianPolicy(
        action_dim=action_dim,
        env=venv,
    ).to(device)

    optimizer = th.optim.Adam(student.parameters(), lr=3e-3)
    scheduler = th.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 25],
        gamma=0.01
    )
    num_rounds = 100
    steps_per_round = 500
    MAX_LEN = 800
    ROT_LOSS_WEIGHT = 70
    STATE_LOSS_WEIGHT = 10  # tune this
    print('ROT_LOSS_WEIGHT', ROT_LOSS_WEIGHT)
    print('STATE_LOSS_WEIGHT', STATE_LOSS_WEIGHT)
    print(optimizer)

    all_imgs = deque(maxlen=MAX_LEN)
    all_acts = deque(maxlen=MAX_LEN)
    all_ori  = deque(maxlen=MAX_LEN)
    all_states = deque(maxlen=MAX_LEN)

    debug_saved = 0
    for round_idx in range(num_rounds):
        obs = venv.reset()
        steps = 0

        while steps < steps_per_round:
            # -------- TEACHER OBS (STATE, from env) --------
            teacher_state = obs  # whatever venv returns; SB3 expects numpy

            # -------- STUDENT OBS (CAMERA → TORCH) --------
            rgb = cam.data.output["rgb"]
            rgb = rgb.permute(0, 3, 1, 2).float()                 # -> (B,C,H,W), float
            rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)

            gray = rgb_to_gray_torch(rgb)

            if gray.max() > 1.0:
                gray = gray / 255.0

            # store for training later
            all_imgs.append(gray.detach().cpu().clone())


            expert_action, _ = expert.predict(
                teacher_state,
                deterministic=True,
            )

            teacher_ori = obs_orientation_bodyframe(isaac_env)
            techer_dist = obs_target_rel_body(isaac_env)
            state_t = th.cat([techer_dist, teacher_ori], dim=1)
            all_states.append(state_t.detach().cpu().clone())

            act_t = th.as_tensor(expert_action, device=device, dtype=th.float32)
            all_acts.append(act_t.detach().cpu().clone())

            # -------- STUDENT ACTS IN ENV --------
            with th.no_grad():
                ori = obs_orientation_bodyframe_noise(isaac_env)
                student_action = student.act(gray, ori, deterministic=False)  # (N_envs, act_dim)

            # SB3 env expects numpy actions
            if (round_idx - 60) >= 0 and (0.5 > random.random()):
                next_obs, rewards, dones, infos = venv.step(student_action)
            else:
                next_obs, rewards, dones, infos = venv.step(expert_action)

            all_ori.append(ori.detach().cpu().clone())
            obs = next_obs
            steps += student_action.shape[0]

        # -------- AFTER EACH ROUND: TRAIN STUDENT (BC / DAGGER STEP) --------
        imgs_t = th.cat(list(all_imgs), dim=0).to(device)  
        acts_t = th.cat(list(all_acts), dim=0).to(device)  
        ori_t = th.cat(list(all_ori), dim=0).to(device)
        states_t = th.cat(list(all_states), dim=0).to(device)  

        dataset_size = imgs_t.shape[0]
        batch_size = 6000
        num_epochs = 200

        loss_sum = 0.0
        n_batches = 0

        diff_l1_sum = 0.0          # all action dims
        n_action_elems = 0

        l1_trans_sum = 0.0         # only translation dims
        l1_rot_sum   = 0.0         # only rotation dims
        n_trans_elems = 0
        n_rot_elems   = 0

        # stats for teacher-state prediction
        state_l1_sum = 0.0
        n_state_elems = 0

        # NEW: accumulators for empirical and predicted variances
        action_sq_err_sum = 0.0        # sum (diff^2) over all action dims  # NEW
        action_pred_var_sum = 0.0      # sum (var) over all action dims     # NEW

        state_sq_err_sum = 0.0         # sum (state_diff^2) over all dims   # NEW
        state_pred_var_sum = 0.0       # sum (state_var) over all dims      # NEW

        for epoch in range(num_epochs):
            perm = th.randperm(dataset_size, device=device)

            for start in range(0, dataset_size, batch_size):
                idx = perm[start:start + batch_size]
                batch_imgs   = imgs_t[idx]
                batch_acts   = acts_t[idx]
                batch_ori    = ori_t[idx]
                batch_states = states_t[idx]   # (B, state_dim)

                # student outputs Gaussian params for actions + teacher state
                mu, log_std, teacher_mu, teacher_logstd = student(batch_imgs, batch_ori)

                # ---------- ACTION LOSS (same as before) ----------
                std = log_std.exp()
                var = std * std

                diff = batch_acts - mu                         # (B, act_dim)

                # per-dim Gaussian NLL (up to constant)
                nll = 0.5 * (diff * diff / var + 2.0 * log_std)

                # split translation / rotation: first 3, last 3
                nll_trans = nll[:, :3]
                nll_rot   = nll[:, 3:]

                loss_trans = nll_trans.mean()
                loss_rot   = nll_rot.mean()

                L_action = loss_trans + ROT_LOSS_WEIGHT * loss_rot

                # ---------- TEACHER-STATE LOSS (Gaussian NLL) ----------
                state_std = teacher_logstd.exp()
                state_var = state_std * state_std

                state_diff = batch_states - teacher_mu          # (B, state_dim)

                # per-dim Gaussian NLL for state (up to constant)
                state_nll = 0.5 * (state_diff * state_diff / state_var + 2.0 * teacher_logstd)

                L_state = state_nll.mean()

                # total loss
                loss = L_action + STATE_LOSS_WEIGHT * L_state

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --------- stats for actions ---------
                loss_sum += loss.item()
                n_batches += 1

                trans = diff[:, :3]
                rot   = diff[:, 3:]

                diff_l1_sum += diff.abs().sum().item()
                n_action_elems += diff.numel()

                l1_trans_sum += trans.abs().sum().item()
                l1_rot_sum   += rot.abs().sum().item()

                n_trans_elems += trans.numel()
                n_rot_elems   += rot.numel()

                # NEW: accumulate empirical and predicted variances for actions
                action_sq_err_sum   += (diff * diff).sum().item()   # NEW
                action_pred_var_sum += var.sum().item()             # NEW

                # --------- stats for teacher state prediction (L1) ---------
                state_l1_sum += state_diff.abs().sum().item()
                n_state_elems += state_diff.numel()

                # NEW: accumulate empirical and predicted variances for state
                state_sq_err_sum   += (state_diff * state_diff).sum().item()  # NEW
                state_pred_var_sum += state_var.sum().item()                  # NEW

        avg_loss    = loss_sum / max(1, n_batches)
        avg_l1      = diff_l1_sum / max(1, n_action_elems)
        avg_l1trans = l1_trans_sum / max(1, n_trans_elems)
        avg_l1rot   = l1_rot_sum   / max(1, n_rot_elems)
        avg_state_l1 = state_l1_sum / max(1, n_state_elems)

        # NEW: compute global empirical and predicted stds
        emp_action_std = math.sqrt(action_sq_err_sum / max(1, n_action_elems))   # NEW
        pred_action_std = math.sqrt(action_pred_var_sum / max(1, n_action_elems))# NEW

        emp_state_std = math.sqrt(state_sq_err_sum / max(1, n_state_elems))      # NEW
        pred_state_std = math.sqrt(state_pred_var_sum / max(1, n_state_elems))   # NEW

        print(
            f"[DAgger] Round {round_idx} done; "
            f"dataset size = {dataset_size} "
            f"loss = {avg_loss:.6f} "
            f"avg L1 = {avg_l1:.3f} "
            f"avg L1_trans = {avg_l1trans:.3f} "
            f"avg L1_rot = {avg_l1rot:.3f} "
            f"avg teacher-state L1 = {avg_state_l1:.3f} "
            f"emp_action_std = {emp_action_std:.4f} "
            f"pred_action_std = {pred_action_std:.4f} "
            f"emp_state_std = {emp_state_std:.4f} "
            f"pred_state_std = {pred_state_std:.4f}"
        )

    th.save(student.state_dict(), './student_acting_29hz')
    print(f"Saved student policy")
    isaac_env.close()
    return student




def load_student(action_dim, env, device="cuda"):
    student = StudentGaussianPolicy(
        action_dim=action_dim,
        env=env,
    ).to(device)

    state_dict = th.load("./student_acting_29hz", map_location=device)
    student.load_state_dict(state_dict)
    student.eval()
    return student


def run_student_sim(num_steps=500):
    import torch as th
    from stable_baselines3 import SAC
    import torch as th
    from collections import deque
    import random
    device = th.device("cuda")

    # --- recreate env exactly as in training ---
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device  # e.g. "cuda:0"
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    venv = Sb3VecEnvWrapper(isaac_env)

    obs = venv.reset()
    cam = isaac_env.scene["front_cam"]

    action_dim = venv.action_space.shape[0]
    student = load_student(action_dim=action_dim, env=venv, device=device)
    rewards_sum = 0
    obs = venv.reset()
    events_t = []
    avg_action = []
    avg_speed= []
    quat_wb = isaac_env.scene["rov"].data.root_state_w[:, 3:7] # orientation (world→body)
    
    
    for t in range(num_steps):
        # -------- CAMERA → underwater tensor (same as training) --------
        rgb = cam.data.output["rgb"]              # (N,H,W,4) or (N,H,W,4)
        rgb = rgb.permute(0, 3, 1, 2).float()                 # -> (B,C,H,W), float
        rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)
        
        rgbw = rgb_to_gray_torch(rgb).to(device)  # (N,4,H,W)
        
        # -------- ORIENTATION FEATURE (same as training) --------
        ori = obs_orientation_bodyframe_noise(isaac_env).to(device)      # (N,4)
        if rgbw.max() > 1.0:
            rgbw = rgbw / 255.0
        # -------- STUDENT ACTION ON GPU --------
        with th.no_grad():
            student_action, _, teacher_state, _ = student.forward(rgbw, ori)

        # -------- STEP ENV WITH STUDENT --------
        # safest: convert to numpy for SB3 VecEnv
        obs, rewards, dones, infos = venv.step(
            student_action
        )
        hit = rewards >= 1.0
        avg_action.append(student_action)
        w_w = isaac_env.scene["rov"].data.root_state_w[:, 10:13]   # angular velocity (world)
    
        w_b = quat_apply_inverse(quat_wb, w_w)
        avg_speed.append(w_b)


        if hit.any():
            dt = isaac_env.sim.get_physics_dt()
            decim = isaac_env.cfg.decimation  # physics steps per RL step
            t_env = isaac_env.episode_length_buf.float() * (dt * decim)  # (num_envs,)

            # append only the times for envs that hit
            events_t.extend(t_env[hit].tolist())
            

        rewards_sum += sum(rewards)
        
        # optional: handle dones / logging here
        # e.g. print some episode rewards if you track them

    all_abs_values = torch.stack(avg_action, dim=0) 
    print(all_abs_values.mean(dim=(0, 1)))
    print(rewards_sum)
    avg_speed = torch.stack(avg_speed, dim=0) 
    print(avg_speed.mean(dim=(0, 1)))

    isaac_env.close()
    print("Student rollout finished.")







# Call this instead of your old main()
if __name__ == "__main__":
    #play_sac_policy()
    #main_sac()
    #train_student()
    run_student_sim()

