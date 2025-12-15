from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

from environment import RobotArmEnv

DEBUG = False


def test_environment_structure():
    print("--- 1. Checking Environment Compliance ---")
    env = RobotArmEnv(reward_type="dense", task="reach")  # Dense reward is better for quick testing

    # SB3 comes with a checker that runs random actions and checks shapes/types
    try:
        check_env(env)
        print("✅ Environment passed Gymnasium checks!")
    except Exception as e:
        print(f"❌ Environment failed checks: {e}")
        return
    env.close()

def train_agent():
    print("\n--- 2. Setting up Training (SAC) ---")

    if DEBUG:
        log_interval = 1
        batch_size = 64
        buffer_size = 50_000
        total_timesteps = 10_000
    else:
        log_interval = 10
        batch_size = 256
        buffer_size = 1_000_000
        total_timesteps = 200_000

    # 1. Initialize WandB
    run = wandb.init(
        project="so101-reach-fb",  # Your project name
        name="SO101-DeltaJoint-Run1",
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": total_timesteps,
            "env_name": "SO101-Reach",
            "algo": "SAC"
        },
        sync_tensorboard=True,  # <--- CRITICAL: Syncs SB3 logs to WandB
    )

    # 2. Create Env
    env = DummyVecEnv([lambda: Monitor(RobotArmEnv(render_mode=None, reward_type="dense", task="reach"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 3. Define the Algorithm
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
        tensorboard_log=f"runs/{run.id}"  # <--- Point this to a local folder
    )

    print("--- 3. Starting Training ---")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=log_interval,
        callback=WandbCallback(
            gradient_save_freq=100,  # Logs gradients every 100 steps
            verbose=2
        )
    )

    # 4. Save locally and Finish
    model.save("../models/sac_so101_reach")
    env.save("../models/vec_normalize.pkl")
    print("✅ Model saved.")

    run.finish()  # Cleanly close the WandB run
    env.close()


def visualize_agent():
    print("\n--- 4. Visualizing Result ---")

    # 1. Load Env (Must use 'human' render mode now)
    env = DummyVecEnv([lambda: RobotArmEnv(render_mode="human", reward_type="dense", task="reach")])

    # 2. Load Normalization Stats
    # We must normalize the test inputs exactly how we normalized training inputs
    env = VecNormalize.load("./models/vec_normalize.pkl", env)
    env.training = False  # Disable updating stats during test
    env.norm_reward = False

    # 3. Load Model
    model = SAC.load("./models/sac_so101_reach")

    obs = env.reset()
    for i in range(1000):
        # Predict action
        action, _states = model.predict(obs, deterministic=True)

        # Step
        obs, rewards, dones, info = env.step(action)

        env.render()

        if dones[0]:
            print("Target Reached or Timeout!")
            obs = env.reset()


if __name__ == "__main__":
    # Uncomment these one by one
    test_environment_structure()
    train_agent()
    if DEBUG:
        visualize_agent()