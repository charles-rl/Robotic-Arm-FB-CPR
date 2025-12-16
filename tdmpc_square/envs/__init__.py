import warnings
import gymnasium as gym
from tdmpc_square.envs.wrappers.tensor import TensorWrapper

# --- SURGERY: Import ONLY your environment ---
try:
    from tdmpc_square.envs.so101 import make_env as make_so101_env
except ImportError as e:
    print(f"Error importing SO101: {e}")
    make_so101_env = None

warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    # gym.logger.set_level(40)

    # 1. Check if the task is your custom robot
    if "so101" in cfg.task:
        if make_so101_env is None:
            raise ValueError("Could not import SO101 environment wrapper.")

        # Instantiate
        env = make_so101_env(cfg)

        # Wrap in TensorWrapper (Handles extra buffer logic if needed, 
        # though our custom wrapper does most of it)
        env = TensorWrapper(env)
    else:
        raise ValueError(f"Task '{cfg.task}' is not supported in this stripped version.")

    # 2. Populate Config Shapes (Critical for TD-MPC2 to build networks)
    try:
        # Dict spaces
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:
        # Box spaces (Your case)
        # cfg.get("obs", "state") checks if we are using 'state' or 'rgb'
        obs_key = cfg.get("obs", "state")
        cfg.obs_shape = {obs_key: env.observation_space.shape}

    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.unwrapped.max_episode_steps

    # Random seed steps buffer
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    return env