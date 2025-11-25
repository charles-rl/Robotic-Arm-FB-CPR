from generate_dataset import Dataset
from environment import RobotArmEnv
import torch as th
import json

def normalize_array(arr, minimum, maximum):
    return 2 * ((arr - minimum) / (maximum - minimum)) - 1

def standardize_array(arr, mean, std):
    return (arr - mean) / std

if __name__ == "__main__":
    save_location = "../data/raw_1000eps.data"
    new_save_location = "../data/processed_1000eps.data"
    env = RobotArmEnv(False)
    dataset_size = th.load(save_location)["counter"]
    episode_data = Dataset(dataset_size, env.observation_space[0], env.action_space[0])
    episode_data.load(save_location)

    """
    actions min and max is [-20.0, 20.0] according to the data
    observations 0 min and max is [-2.5, 2.5] makes sense for arm length
    observations 1 min and max is [-2.5, 2.5]
    
    Standardize motor forces and velocity because it handles outliers better
    """
    episode_data.actions = normalize_array(episode_data.actions, -20, 20)
    env_max_force = float(env.MAX_FORCE)
    motor_force_mean, motor_force_std = episode_data.motor_forces.mean(), episode_data.motor_forces.std()
    episode_data.motor_forces = standardize_array(episode_data.motor_forces, motor_force_mean, motor_force_std)
    # For current observations
    episode_data.observations[:, 0] = normalize_array(episode_data.observations[:, 0], -2.5, 2.5)
    episode_data.observations[:, 1] = normalize_array(episode_data.observations[:, 1], -2.5, 2.5)
    # Append last observation to get complete observations
    all_observations1 = th.concat((episode_data.observations[:, -1], episode_data.observations_[-1, -1].unsqueeze(dim=-1)))
    observation1_mean, observation1_std = all_observations1.mean(), all_observations1.std()
    episode_data.observations[:, -1] = standardize_array(episode_data.observations[:, -1], observation1_mean, observation1_std)

    all_observations2 = th.concat((episode_data.observations[:, -2], episode_data.observations_[-1, -2].unsqueeze(dim=-1)))
    observation2_mean, observation2_std = all_observations2.mean(), all_observations2.std()
    episode_data.observations[:, -2] = standardize_array(episode_data.observations[:, -2], observation2_mean, observation2_std)

    all_observations3 = th.concat((episode_data.observations[:, -3], episode_data.observations_[-1, -3].unsqueeze(dim=-1)))
    observation3_mean, observation3_std = all_observations3.mean(), all_observations3.std()
    episode_data.observations[:, -3] = standardize_array(episode_data.observations[:, -3], observation3_mean, observation3_std)

    # For future observations
    episode_data.observations_[:, 0] = normalize_array(episode_data.observations_[:, 0], -2.5, 2.5)
    episode_data.observations_[:, 1] = normalize_array(episode_data.observations_[:, 1], -2.5, 2.5)
    episode_data.observations_[:, -1] = standardize_array(episode_data.observations_[:, -1], observation1_mean, observation1_std)
    episode_data.observations_[:, -2] = standardize_array(episode_data.observations_[:, -2], observation2_mean, observation2_std)
    episode_data.observations_[:, -3] = standardize_array(episode_data.observations_[:, -3], observation3_mean, observation3_std)
    episode_data.save(new_save_location)

    del env

    means_and_stds = {
        "observation -1 mean": observation1_mean.item(),
        "observation -1 std": observation1_std.item(),
        "observation -2 mean": observation2_mean.item(),
        "observation -2 std": observation2_std.item(),
        "observation -3 mean": observation3_mean.item(),
        "observation -3 std": observation3_std.item(),
        "motor force mean": motor_force_mean.item(),
        "motor force std": motor_force_std.item()
    }
    with open("../data/means_and_stds.json", "w") as f:
        json.dump(means_and_stds, f)

    print("Saved means and stds to means_and_stds.json")
