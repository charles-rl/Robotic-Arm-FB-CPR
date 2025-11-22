from generate_dataset import Dataset
from environment import RobotArmEnv
import torch as th

if __name__ == "__main__":
    save_location = "../data/raw_1000eps.data"
    new_save_location = "../data/processed_1000eps.data"
    env = RobotArmEnv(False)
    dataset_size = th.load(save_location)["counter"]
    episode_data = Dataset(dataset_size, env.observation_space[0], env.action_space[0])
    del env
    episode_data.load(save_location)


