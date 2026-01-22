import argparse, torch, random
import torch.nn.functional as F
from collections import deque
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=["train", "play"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import SAC
from rov_env import ROVEnvCfg, obs_orientation_bodyframe_noise, obs_target_rel_body, obs_orientation_bodyframe_noise
from models import StudentGaussianPolicy
from rendering import rgb_to_gray_torch

def train_student():
    device = torch.device("cuda")
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    venv = Sb3VecEnvWrapper(isaac_env)
    
    # Load Expert
    expert = SAC.load("sac_rov_final.zip", env=venv, device="cuda")
    cam = isaac_env.scene['front_cam']
    
    # Init Student
    student = StudentGaussianPolicy(action_dim=venv.action_space.shape[0], env=venv).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-3)
    
    # Buffers
    all_imgs, all_acts, all_ori, all_states = deque(maxlen=800), deque(maxlen=800), deque(maxlen=800), deque(maxlen=800)
    
    for round_idx in range(100):
        obs = venv.reset()
        steps = 0
        while steps < 500:
            # Process Images
            rgb = cam.data.output["rgb"].permute(0, 3, 1, 2).float()
            rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)
            gray = rgb_to_gray_torch(rgb) / 255.0
            
            # Get Expert Action (Teacher)
            expert_action, _ = expert.predict(obs, deterministic=True)
            
            # Save Data
            all_imgs.append(gray.detach().cpu())
            all_acts.append(torch.tensor(expert_action).cpu())
            all_states.append(torch.cat([obs_target_rel_body(isaac_env), obs_orientation_bodyframe_noise(isaac_env)], dim=1).detach().cpu())

            # Student Act (for Rollout)
            with torch.no_grad():
                ori = obs_orientation_bodyframe_noise(isaac_env)
                all_ori.append(ori.detach().cpu())
                student_act = student.act(gray, ori)
            
            # Step (Mix Expert/Student)
            act_to_use = student_act.cpu().numpy() if (round_idx > 60 and random.random() < 0.5) else expert_action
            obs, _, _, _ = venv.step(act_to_use)
            steps += venv.num_envs

        # Train Loop (BC)
        print(f"Training Round {round_idx}...")
        imgs_t = torch.cat(list(all_imgs)).to(device)
        acts_t = torch.cat(list(all_acts)).to(device)
        ori_t  = torch.cat(list(all_ori)).to(device)
        
        for _ in range(200): # Epochs
            mu, log_std, _, _ = student(imgs_t, ori_t)
            loss = F.mse_loss(mu, acts_t) # Simple MSE for brevity, add Gaussian NLL if needed
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    torch.save(student.state_dict(), "student_policy.pt")
    isaac_env.close()

def play_student():
    device = torch.device("cuda")
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    venv = Sb3VecEnvWrapper(isaac_env)
    
    student = StudentGaussianPolicy(action_dim=venv.action_space.shape[0], env=venv).to(device)
    student.load_state_dict(torch.load("student_policy.pt"))
    student.eval()
    
    cam = isaac_env.scene['front_cam']
    venv.reset()
    
    while True:
        rgb = cam.data.output["rgb"].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)
        gray = rgb_to_gray_torch(rgb).to(device) / 255.0
        ori = obs_orientation_bodyframe_noise(isaac_env).to(device)
        
        with torch.no_grad():
            action = student.act(gray, ori, deterministic=True)
            
        venv.step(action.cpu().numpy())

if __name__ == "__main__":
    if args_cli.mode == "train": train_student()
    else: play_student()
