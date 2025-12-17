import argparse
import os
import pickle
import numpy as np
import torch
import gymnasium as gym
import imageio

# Import the Agent and Env from your training script
# Assumes train_ppo_rnd.py and so101_env.py are in the same folder
from train_rnd import Agent, RunningMeanStd
from environment import RobotArmEnv


class SingleEnvWrapper:
    """
    The Agent class in train_ppo_rnd.py expects a Vectorized Environment
    (envs.single_observation_space). This wrapper makes a standard Env
    look like a Vector Env so the Agent initializes correctly.
    """

    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


def evaluate(
        run_folder: str,
        model_name: str = "latest_model.pt",
        rms_name: str = "latest_obs_rms.pkl",
        episodes: int = 3,
        save_video: bool = False
):
    device = torch.device("cpu")  # Inference is fast, CPU is fine

    # 1. Load Environment
    # We use rgb_array to capture frames for video, but we can also render human if needed
    render_mode = "rgb_array" if save_video else "human"
    env = RobotArmEnv(render_mode=render_mode, task="reach", reward_type="sparse")

    # 2. Load Agent
    # Wrap env so Agent can read shapes
    dummy_env = SingleEnvWrapper(env)
    agent = Agent(dummy_env).to(device)

    model_path = os.path.join(run_folder, model_name)
    print(f"Loading Model from: {model_path}")
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # 3. Load Normalization Statistics
    # CRITICAL: If you don't do this, the agent will see garbage values
    rms_path = os.path.join(run_folder, rms_name)
    print(f"Loading Obs Normalization from: {rms_path}")
    with open(rms_path, "rb") as f:
        obs_rms = pickle.load(f)

    # 4. Evaluation Loop
    for episode in range(episodes):
        obs, _ = env.reset()
        frames = []
        done = False
        step = 0
        total_reward = 0

        print(f"--- Starting Episode {episode + 1} ---")

        while not done:
            # A. Normalize Observation manually
            # (obs - mean) / sqrt(var)
            # We clip to [-10, 10] just like in training
            obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            obs_norm = np.clip(obs_norm, -10, 10).astype(np.float32)

            # B. Get Action
            with torch.no_grad():
                tensor_obs = torch.FloatTensor(obs_norm).unsqueeze(0).to(device)

                # OPTION 1: Deterministic (Best for Eval)
                # Use the mean of the Gaussian distribution
                action_mean = agent.actor_mean(tensor_obs)
                action = action_mean.cpu().numpy()[0]

                # OPTION 2: Stochastic (If you want to see variance)
                # action, _, _, _, _ = agent.get_action_and_value(tensor_obs)
                # action = action.cpu().numpy()[0]

            # C. Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # D. Capture Video
            if save_video:
                frame = env.render()  # Returns RGB array
                # Draw the target text or info on frame if desired
                frames.append(frame)
            else:
                env.render()  # Just show window

            if done:
                print(f"Episode finished. Steps: {step}, Reward: {total_reward:.4f}")

        # Save Video
        if save_video:
            video_path = f"video_ep_{episode + 1}.mp4"
            print(f"Saving video to {video_path}...")
            # 30 FPS matches your 33ms frame skip roughly
            imageio.mimsave(video_path, frames, fps=30)

    env.close()


if __name__ == "__main__":
    # CHANGE THIS to the specific timestamp folder created in /runs/
    # Example: "runs/SO101-Reach-v0__train_ppo_rnd__1__1702721234"
    RUN_FOLDER = "runs/a"

    # Check if folder exists before running
    if not os.path.exists(RUN_FOLDER):
        print(f"Error: Folder {RUN_FOLDER} not found.")
        print("Please check your 'runs/' directory and paste the correct folder name in the script.")
    else:
        evaluate(RUN_FOLDER, episodes=3, save_video=True)