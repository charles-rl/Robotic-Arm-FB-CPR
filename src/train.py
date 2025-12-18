import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import imageio
import wandb
import os

# --- ADDED TQC HERE ---
from sb3_contrib import TQC, CrossQ
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

from environment import RobotArmEnv

# Configuration
DEBUG = False
EVAL = False
TASK = "lift"
ALGO = "TQC"  # <--- CHANGE THIS: "SAC", "TQC", or "PPO" or "CrossQ"
CONTROL_MODE = "delta_end_effector"
RUN_NAME = f"{ALGO}_{CONTROL_MODE}"


class RawRewardCallback(BaseCallback):
    """Logs raw reward/length to WandB."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                wandb.log({
                    "rollout/raw_episode_reward": info['episode']['r'],
                    "rollout/raw_episode_length": info['episode']['l'],
                    "env_steps": self.num_timesteps
                })
        return True


class VideoRecorderCallback(BaseCallback):
    """
    Records a video of the agent's performance every `render_freq` steps.
    """

    def __init__(self, eval_env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.video_path = "latest_training_run.mp4"

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            print(f"🎥 Recording video at step {self.num_timesteps}...")
            if isinstance(self._eval_env, VecNormalize):
                self._eval_env.training = False
                self._eval_env.norm_reward = False
                self._eval_env.obs_rms = self.training_env.obs_rms
                self._eval_env.ret_rms = self.training_env.ret_rms

            frames = []
            for _ in range(self._n_eval_episodes):
                obs = self._eval_env.reset()
                done = False
                while not done:
                    frame = self._eval_env.envs[0].render()
                    frames.append(frame)
                    action, _ = self.model.predict(obs, deterministic=self._deterministic)
                    obs, _, dones, _ = self._eval_env.step(action)
                    done = dones[0]

            try:
                imageio.mimsave(self.video_path, frames, fps=30)
                print(f"✅ Video saved to {self.video_path}")
            except Exception as e:
                print(f"❌ Failed to save video: {e}")
        return True


def test_environment_structure():
    print("--- 1. Checking Environment Compliance ---")
    env = RobotArmEnv(reward_type="dense", task=TASK, control_mode=CONTROL_MODE)
    try:
        check_env(env)
        print("✅ Environment passed Gymnasium checks!")
    except Exception as e:
        print(f"❌ Environment failed checks: {e}")
    env.close()


def train_agent():
    print(f"\n--- 2. Setting up Training ({ALGO}) for task: {TASK} ---")

    if DEBUG:
        log_interval = 1
        batch_size = 64
        buffer_size = 50_000
        total_timesteps = 10_000
        video_freq = 200
    else:
        log_interval = 10
        batch_size = 256
        buffer_size = 1_000_000
        total_timesteps = 300_000
        video_freq = 10_000

    # 1. Initialize WandB
    run = wandb.init(
        project=f"so101-{TASK}-fb",
        name=RUN_NAME,
        config={"algo": ALGO, "task": TASK},
        sync_tensorboard=True,
    )

    # 2. Create Training Env
    env = DummyVecEnv(
        [lambda: Monitor(RobotArmEnv(render_mode=None, reward_type="dense", task=TASK, control_mode=CONTROL_MODE))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., clip_reward=100.)

    # 3. Create Eval Env (Optional, currently unused for speed)
    # eval_env = DummyVecEnv([lambda: RobotArmEnv(render_mode="rgb_array", reward_type="dense", task=TASK)])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # eval_env.training = False

    # 4. Define Algorithm based on ALGO selection
    common_params = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}"
    )

    if ALGO == "TQC":
        model = TQC(
            **common_params,
            top_quantiles_to_drop_per_net=2,  # <--- The TQC Magic Parameter
            learning_rate=3e-4,
            batch_size=batch_size,
            buffer_size=buffer_size,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
    elif ALGO == "CrossQ":
        model = CrossQ(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,  # CrossQ likes higher learning rates
            batch_size=256,
            buffer_size=1_000_000,
            ent_coef="auto",
        )
    elif ALGO == "SAC":
        model = SAC(
            **common_params,
            learning_rate=5e-5,
            batch_size=batch_size,
            buffer_size=buffer_size,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
    elif ALGO == "PPO":
        # PPO is On-Policy: No buffer_size, different batch logic
        model = PPO(
            **common_params,
            learning_rate=3e-4,
            n_steps=2048,  # Steps to run before update
            batch_size=64,  # Minibatch size
            ent_coef=0.01,
        )
    else:
        raise ValueError(f"Unknown Algorithm: {ALGO}")

    # 5. Callbacks
    raw_reward_callback = RawRewardCallback()
    wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=f"./models/{run.id}")

    callback_group = CallbackList([
        raw_reward_callback,
        # video_callback,
        wandb_callback,
        checkpoint_callback
    ])

    print("--- 3. Starting Training ---")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=log_interval,
        callback=callback_group,
    )

    # 6. Save
    model.save(f"../models/{ALGO.lower()}_so101_{TASK}")
    env.save(f"../models/vec_normalize_{TASK}.pkl")
    print("✅ Model saved.")

    run.finish()
    env.close()


def evaluate(episodes=3, find_best=False):
    print(f"\n--- 4. Evaluating Agent: {TASK} ({ALGO}) ---")

    # 1. Load Env
    env_ = DummyVecEnv(
        [lambda: RobotArmEnv(render_mode="rgb_array", reward_type="dense", task=TASK, control_mode=CONTROL_MODE)])
    max_steps = env_.get_attr("max_episode_steps")[0]

    # 2. Load Normalization Stats
    path = f"../models/vec_normalize_{TASK}.pkl"
    if os.path.exists(path):
        env = VecNormalize.load(path, env_)
        env.training = False
        env.norm_reward = False
        print("Loaded Normalization Stats.")
    else:
        env = env_

    # 3. Load Correct Model Class
    if ALGO == "TQC":
        ModelClass = TQC
    elif ALGO == "SAC":
        ModelClass = SAC
    elif ALGO == "PPO":
        ModelClass = PPO

    model = ModelClass.load(f"../models/{ALGO.lower()}_so101_{TASK}")

    # 4. Loop
    if not find_best:
        for ep in range(episodes):
            frames = []
            obs = env.reset()
            video_path = f"eval_ep_{ep}.mp4"
            print(f"Simulating Episode {ep + 1}/{episodes}...")

            for i in range(max_steps + 10):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                frame = env.envs[0].render()
                frames.append(frame)
                if dones[0]: break

            print(f"Saving {video_path}...")
            imageio.mimsave(video_path, frames, fps=30)
    else:
        num_good_eps = 0
        while num_good_eps < episodes:
            frames = []
            obs = env.reset()
            video_path = f"eval_ep_{num_good_eps}.mp4"
            print(f"Simulating Episode {num_good_eps + 1}/{episodes}...")
            total_reward = 0.0
            for i in range(max_steps + 10):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                total_reward += rewards[0]
                frame = env.envs[0].render()
                frames.append(frame)
                if dones[0]: break

            if total_reward > 550.0:
                print(f"Saving {video_path}...")
                imageio.mimsave(video_path, frames, fps=30)
                num_good_eps += 1
            else:
                print("Failed episode")


if __name__ == "__main__":
    test_environment_structure()

    if not EVAL:
        train_agent()
    else:
        evaluate(find_best=True)