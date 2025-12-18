import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import imageio
import wandb
import os

from stable_baselines3 import SAC
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
RUN_NAME = f"SAC"


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
    Overwrites the previous video to save space.
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

            # Sync normalization stats if the env is normalized
            # This ensures the eval env "sees" the world the same way the training env does
            if isinstance(self._eval_env, VecNormalize):
                # self.training_env is the env passed to model.learn()
                self._eval_env.training = False
                self._eval_env.norm_reward = False
                # Copy statistics (mean/var) from training env to eval env
                self._eval_env.obs_rms = self.training_env.obs_rms
                self._eval_env.ret_rms = self.training_env.ret_rms

            frames = []

            for _ in range(self._n_eval_episodes):
                obs = self._eval_env.reset()
                done = False
                while not done:
                    # Get render frame
                    # Note: We need to access the inner env to call render() if it's wrapped in VecEnv
                    # But VecEnv's render() usually delegates correctly if setup right.
                    # For custom implementation:
                    frame = self._eval_env.envs[0].render()
                    frames.append(frame)

                    action, _ = self.model.predict(obs, deterministic=self._deterministic)
                    obs, _, dones, _ = self._eval_env.step(action)
                    done = dones[0]

            # Save and Overwrite
            try:
                imageio.mimsave(self.video_path, frames, fps=30)
                print(f"✅ Video saved to {self.video_path}")

                # Optional: Log to WandB (Uncomment if you want history)
                # wandb.log({"video": wandb.Video(self.video_path, fps=30, format="mp4")}, step=self.num_timesteps)
            except Exception as e:
                print(f"❌ Failed to save video: {e}")

        return True


def test_environment_structure():
    print("--- 1. Checking Environment Compliance ---")
    env = RobotArmEnv(reward_type="dense", task=TASK)
    try:
        check_env(env)
        print("✅ Environment passed Gymnasium checks!")
    except Exception as e:
        print(f"❌ Environment failed checks: {e}")
        # Continue anyway for debugging
    env.close()


def train_agent():
    print(f"\n--- 2. Setting up Training (SAC) for task: {TASK} ---")

    if DEBUG:
        log_interval = 1
        batch_size = 64
        buffer_size = 50_000
        total_timesteps = 10_000
        video_freq = 200
    else:
        # reach
        # 256 batch size
        # total_timesteps = 200_000
        log_interval = 10
        batch_size = 256
        buffer_size = 1_000_000
        total_timesteps = 300_000
        video_freq = 10_000  # Save video every 10k steps

    # 1. Initialize WandB
    run = wandb.init(
        project=f"so101-{TASK}-fb",
        name=RUN_NAME,
        config={"algo": "SAC", "task": TASK},
        sync_tensorboard=True,
    )

    # 2. Create Training Env (No Render, optimized for speed)
    env = DummyVecEnv([lambda: Monitor(RobotArmEnv(render_mode=None, reward_type="dense", task=TASK))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., clip_reward=100.)

    # 3. Create Evaluation/Video Env (With Render)
    # We wrap this in VecNormalize too so it understands the normalized inputs
    # eval_env = DummyVecEnv([lambda: RobotArmEnv(render_mode="rgb_array", reward_type="dense", task=TASK)])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # We turn off training on this one, stats will be synced in the callback
    # eval_env.training = False

    # not use for reach task

    # 4. Define Algorithm
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=batch_size,
        buffer_size=buffer_size,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        tensorboard_log=f"runs/{run.id}"
    )

    # 5. Callbacks
    # video_callback = VideoRecorderCallback(eval_env, render_freq=video_freq)
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
    model.save(f"../models/sac_so101_{TASK}")
    env.save(f"../models/vec_normalize_{TASK}.pkl")
    print("✅ Model saved.")

    run.finish()
    env.close()
    # eval_env.close()


def evaluate(episodes=3, find_best=False):
    print(f"\n--- 4. Evaluating Agent: {TASK} ---")

    # 1. Load Env
    env_ = DummyVecEnv([lambda: RobotArmEnv(render_mode="rgb_array", reward_type="dense", task=TASK)])

    # 2. Access inner max_episode_steps correctly
    # DummyVecEnv wraps the env in a list. We access attributes via get_attr
    max_steps = env_.get_attr("max_episode_steps")[0]

    # 3. Load Normalization Stats
    path = f"../models/vec_normalize_{TASK}.pkl"
    if os.path.exists(path):
        env = VecNormalize.load(path, env_)
        env.training = False
        env.norm_reward = False
        print("Loaded Normalization Stats.")
    else:
        print("Warning: No normalization stats found. Using raw env.")
        env = env_

    # 4. Load Model
    model = SAC.load(f"../models/sac_so101_{TASK}")

    # 5. Loop
    if not find_best:
        for ep in range(episodes):
            frames = []
            obs = env.reset()
            video_path = f"eval_ep_{ep}.mp4"
            print(f"Simulating Episode {ep + 1}/{episodes}...")

            # We assume the env has a max_steps limit, but we use a safety break just in case
            for i in range(max_steps + 10):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)

                # Use the inner env's render to get the pixel array
                frame = env.envs[0].render()
                frames.append(frame)

                if dones[0]:
                    break

            print(f"Saving {video_path}...")
            imageio.mimsave(video_path, frames, fps=30)
    else:
        num_good_eps = 0
        while num_good_eps < episodes:
            frames = []
            obs = env.reset()
            video_path = f"eval_ep_{num_good_eps}.mp4"
            print(f"Simulating Episode {num_good_eps + 1}/{episodes}...")

            # We assume the env has a max_steps limit, but we use a safety break just in case
            total_reward = 0.0
            for i in range(max_steps + 10):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                total_reward += rewards[0]

                # Use the inner env's render to get the pixel array
                frame = env.envs[0].render()
                frames.append(frame)

                if dones[0]:
                    break

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