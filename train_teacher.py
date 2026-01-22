import argparse, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan
from rov_env import ROVEnvCfg

def main_sac():
    env_cfg = ROVEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    venv = Sb3VecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    venv = VecCheckNan(venv, raise_exception=True)

    model = SAC(
        policy="MlpPolicy", env=venv, device="cuda",
        buffer_size=5_000_000, batch_size=8048, learning_rate=3e-3,
        train_freq=1, gradient_steps=20, ent_coef='auto',
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
    )

    ckpt = CheckpointCallback(save_freq=1_000_000, save_path="./checkpoints_sac_rov", name_prefix="sac_rov")
    model.learn(total_timesteps=500_000_000, callback=ckpt)
    model.save("sac_rov_final")
    venv.close()

if __name__ == "__main__":
    main_sac()
