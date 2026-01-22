import time
import os
import argparse
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecCheckNan
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnv

from rov_env import ROVEnvCfg

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--device", type=str, default="cuda")
# Add other args if needed
args_cli, _ = parser.parse_known_args()

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

def main_sac():
    # --- build Isaac RL env ---
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)   # num_envs > 1 is fine (SAC supports VecEnv)
    vec_env = Sb3VecEnvWrapper(isaac_env)
    vec_env = VecCheckNan(vec_env, raise_exception=True)

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
    log_cb = RewardTermsRunningAvgPrinter(
        every_n=1_000_000,
        keys={"pointing_loss", "effor_x_axis", "ang_rate_cost",
              "goal_reached", "prog_norm", "distance_to_target"},
    )

    ckpt_dir = "./checkpoints_sac_rov"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000_000,
        save_path=ckpt_dir,
        name_prefix="sac_rov",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList([log_cb, checkpoint_cb])

    start = time.time()
    try:
        model.learn(
            total_timesteps=500_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("Interrupted, saving last checkpoint to sac_rov_policy_interrupt.zip")
        model.save("sac_rov_policy_interrupt")

    end = time.time()
    elapsed = end - start
    print(f"Training finished in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds).")

    model.save("sac_rov_policy_28hz_3")
    vec_env.close()

def play_sac_policy():
    env_cfg = ROVEnvCfg()
    env_cfg.sim.device = args_cli.device
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    vec_env = Sb3VecEnvWrapper(isaac_env)

    model = SAC.load("sac_rov_policy_28hz_2.zip", env=vec_env, device="cuda")

    obs = vec_env.reset()
    episode_rewards = [0.0 for _ in range(vec_env.num_envs)]

    print("Playing policy... Press Ctrl+C to stop.")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        for i in range(vec_env.num_envs):
            episode_rewards[i] += rewards[i]
            if dones[i]:
                print(f"Env {i} episode reward: {episode_rewards[i]:.3f}")
                episode_rewards[i] = 0.0
    vec_env.close()

if __name__ == "__main__":
    main_sac()
