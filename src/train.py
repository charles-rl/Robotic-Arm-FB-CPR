import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import imageio
import wandb
import os

from sb3_contrib import TQC, CrossQ
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

from environment import RobotArmEnv

# Configuration
DEBUG = False
EVAL = False
TASK = "lift"
ALGO = "TQC"  # <--- CHANGE THIS: "SAC", "TQC", or "PPO" or "CrossQ"
CONTROL_MODE = "delta_end_effector"  # delta_end_effector delta_joint_position
REWARD_THRESHOLD = 100.0
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


# --- NEW CALLBACK TO FIX SAVING ---
class CheckpointWithStatsCallback(BaseCallback):
    """
    Saves the model AND the VecNormalize statistics every `save_freq` steps.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 1. Save Model
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)

            # 2. Save Normalization Stats
            stats_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
            # Get the env from the model safely
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(stats_path)
                if self.verbose > 1:
                    print(f"Saved model and stats to {self.save_path}")
            else:
                if self.verbose > 0:
                    print("⚠️ Warning: VecNormalize not found, only saved model.")
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
    else:
        log_interval = 10
        batch_size = 512
        buffer_size = 1_000_000
        total_timesteps = 300_000

    # 1. Initialize WandB
    run = wandb.init(
        project=f"so101-{TASK}-fb",
        name=RUN_NAME,
        config={"algo": ALGO, "task": TASK, "control": CONTROL_MODE},
        sync_tensorboard=True,
    )

    # 2. Create Training Env
    env = DummyVecEnv(
        [lambda: Monitor(RobotArmEnv(render_mode=None, reward_type="dense", task=TASK, control_mode=CONTROL_MODE))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    policy_kwargs = dict(net_arch=[256, 256])
    if ALGO == "TQC":
        policy_kwargs = dict(
            use_sde=True,  # <--- 1. ENABLE gSDE (Smooth Exploration)
            log_std_init=-2,  # Start with smaller, more precise exploration
            net_arch=[256, 256],  # Standard architecture
        )

    # 3. Define Algorithm
    common_params = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
    )

    if ALGO == "TQC":
        model = TQC(
            **common_params,
            top_quantiles_to_drop_per_net=2,
            learning_rate=3e-4,
            batch_size=batch_size,
            buffer_size=buffer_size,
            train_freq=(100, "step"),
            gradient_steps=200,
            ent_coef="auto",
        )
    elif ALGO == "CrossQ":
        model = CrossQ(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            batch_size=256,
            buffer_size=1_000_000,
            ent_coef="auto",
        )
    elif ALGO == "SAC":
        model = SAC(
            **common_params,
            learning_rate=3e-4,  # Standard SAC LR
            batch_size=batch_size,
            buffer_size=buffer_size,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
    elif ALGO == "PPO":
        model = PPO(
            **common_params,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
        )
    else:
        raise ValueError(f"Unknown Algorithm: {ALGO}")

    # 4. Callbacks
    raw_reward_callback = RawRewardCallback()
    wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)

    # --- REPLACED CheckpointCallback WITH CheckpointWithStatsCallback ---
    checkpoint_callback = CheckpointWithStatsCallback(
        save_freq=20000,
        save_path=f"./models/{run.id}",
        name_prefix="rl_model"
    )

    callback_group = CallbackList([
        raw_reward_callback,
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

    # 5. Final Save
    model.save(f"../models/{ALGO.lower()}_so101_{TASK}")
    env.save(f"../models/vec_normalize_{TASK}.pkl")
    print("✅ Final Model & Stats saved.")

    run.finish()
    env.close()


def evaluate(best_reward, episodes=3, find_best=False, only_visualize=False):
    print(f"\n--- 4. Evaluating Agent: {TASK} ({ALGO}) ---")

    # 1. Load Env
    if only_visualize:
        env_ = DummyVecEnv(
            [lambda: RobotArmEnv(render_mode="human", reward_type="dense", task=TASK, control_mode=CONTROL_MODE)])
    else:
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
        print("⚠️ Warning: No normalization stats found. Evaluation might be broken.")
        env = env_

    # 3. Load Correct Model Class
    if ALGO == "TQC":
        ModelClass = TQC
    elif ALGO == "CrossQ":
        ModelClass = CrossQ
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
                if only_visualize:
                    env.envs[0].render()
                else:
                    frame = env.envs[0].render()
                    frames.append(frame)
                if dones[0]: break

            if not only_visualize:
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

            if total_reward > best_reward:
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
        evaluate(best_reward=REWARD_THRESHOLD, find_best=False)
