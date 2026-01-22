import argparse
import random
import math
import torch
import torch.nn.functional as F
from collections import deque
from stable_baselines3 import SAC
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.utils.math import quat_apply_inverse

# Local imports
from rov_env import ROVEnvCfg, obs_orientation_bodyframe, obs_target_rel_body, obs_orientation_bodyframe_noise
from models import StudentGaussianPolicy, ImageEncoder
from rendering import rgb_to_gray_torch

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
args_cli, _ = parser.parse_known_args()

def train_student():
    device = torch.device("cuda")

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

    optimizer = torch.optim.Adam(student.parameters(), lr=3e-3)
    # scheduler unused but defined in original
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 25],
        gamma=0.01
    )
    num_rounds = 100
    steps_per_round = 500
    MAX_LEN = 800
    ROT_LOSS_WEIGHT = 70
    STATE_LOSS_WEIGHT = 10 
    print('ROT_LOSS_WEIGHT', ROT_LOSS_WEIGHT)
    print('STATE_LOSS_WEIGHT', STATE_LOSS_WEIGHT)
    print(optimizer)

    all_imgs = deque(maxlen=MAX_LEN)
    all_acts = deque(maxlen=MAX_LEN)
    all_ori  = deque(maxlen=MAX_LEN)
    all_states = deque(maxlen=MAX_LEN)

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
            state_t = torch.cat([techer_dist, teacher_ori], dim=1)
            all_states.append(state_t.detach().cpu().clone())

            act_t = torch.as_tensor(expert_action, device=device, dtype=torch.float32)
            all_acts.append(act_t.detach().cpu().clone())

            # -------- STUDENT ACTS IN ENV --------
            with torch.no_grad():
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
        imgs_t = torch.cat(list(all_imgs), dim=0).to(device)  
        acts_t = torch.cat(list(all_acts), dim=0).to(device)  
        ori_t = torch.cat(list(all_ori), dim=0).to(device)
        states_t = torch.cat(list(all_states), dim=0).to(device)  

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
            perm = torch.randperm(dataset_size, device=device)

            for start in range(0, dataset_size, batch_size):
                idx = perm[start:start + batch_size]
                batch_imgs   = imgs_t[idx]
                batch_acts   = acts_t[idx]
                batch_ori    = ori_t[idx]
                batch_states = states_t[idx]   # (B, state_dim)

                # student outputs Gaussian params for actions + teacher state
                mu, log_std, teacher_mu, teacher_logstd = student(batch_imgs, batch_ori)

                # ---------- ACTION LOSS ----------
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

    torch.save(student.state_dict(), './student_acting_29hz')
    print(f"Saved student policy")
    isaac_env.close()
    return student

def load_student(action_dim, env, device="cuda"):
    student = StudentGaussianPolicy(
        action_dim=action_dim,
        env=env,
    ).to(device)

    state_dict = torch.load("./student_acting_29hz", map_location=device)
    student.load_state_dict(state_dict)
    student.eval()
    return student

def run_student_sim(num_steps=500):
    device = torch.device("cuda")

    # --- recreate env exactly as in training ---
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    venv = Sb3VecEnvWrapper(isaac_env)

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
        rgb = cam.data.output["rgb"]              # (N,H,W,4)
        rgb = rgb.permute(0, 3, 1, 2).float()     # -> (B,C,H,W), float
        rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)
        
        rgbw = rgb_to_gray_torch(rgb).to(device)  # (N,4,H,W)
        
        # -------- ORIENTATION FEATURE (same as training) --------
        ori = obs_orientation_bodyframe_noise(isaac_env).to(device)      # (N,4)
        if rgbw.max() > 1.0:
            rgbw = rgbw / 255.0
        
        # -------- STUDENT ACTION ON GPU --------
        with torch.no_grad():
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

    all_abs_values = torch.stack(avg_action, dim=0) 
    print(all_abs_values.mean(dim=(0, 1)))
    print(rewards_sum)
    avg_speed = torch.stack(avg_speed, dim=0) 
    print(avg_speed.mean(dim=(0, 1)))

    isaac_env.close()
    print("Student rollout finished.")

if __name__ == "__main__":
    # train_student()
    run_student_sim()
