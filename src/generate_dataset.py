from environment import RobotArmEnv, simp_angle
import numpy as np
import time
import torch as th

from src.environment import theoretical2simulation_angles

class Dataset:
    def __init__(self, dataset_size, n_observations, n_actions):
        self.observations = th.zeros((dataset_size, n_observations), dtype=th.float32)
        self.observations_ = th.zeros((dataset_size, n_observations), dtype=th.float32)
        self.actions = th.zeros((dataset_size, n_actions), dtype=th.float32)
        self.arm_raw_angles = th.zeros((dataset_size, n_actions), dtype=th.float32)
        self.arm_raw_velocities = th.zeros((dataset_size, n_actions), dtype=th.float32)
        self.motor_rates = th.zeros((dataset_size, n_actions), dtype=th.float32)
        self.motor_forces = th.zeros((dataset_size, n_actions), dtype=th.float32)
        self.dones = th.zeros((dataset_size, 1), dtype=th.bool)
        self.counter = 0

    def update(self, obs, action, obs_, done, info):
        # assert(self.counter < self.observations.shape[0])
        self.observations[self.counter] = th.tensor(obs, dtype=th.float32)
        self.observations_[self.counter] = th.tensor(obs_, dtype=th.float32)
        self.actions[self.counter] = th.tensor(action, dtype=th.float32)
        self.dones[self.counter] = th.tensor(done, dtype=th.bool)
        self.arm_raw_angles[self.counter] = th.tensor(info["arm_raw_angles"], dtype=th.float32)
        self.arm_raw_velocities[self.counter] = th.tensor(info["arm_raw_velocities"], dtype=th.float32)
        self.motor_rates[self.counter] = th.tensor(info["motor_rates"], dtype=th.float32)
        self.motor_forces[self.counter] = th.tensor(info["motor_forces"], dtype=th.float32)
        self.counter += 1

    def save(self, filepath):
        """Saves the dataset to a file."""
        # We create a dictionary containing all the tensors and the counter
        payload = {
            'observations': self.observations,
            'actions': self.actions,
            'arm_raw_angles': self.arm_raw_angles,
            'arm_raw_velocities': self.arm_raw_velocities,
            'motor_rates': self.motor_rates,
            'motor_forces': self.motor_forces,
            'counter': self.counter
        }
        th.save(payload, filepath)
        print(f"Dataset saved to {filepath} ({self.counter} transitions)")

    def load(self, filepath):
        """Loads data from a file into this dataset instance."""
        print(f"Loading dataset from {filepath}...")
        payload = th.load(filepath)

        # Restore the data
        # Note: This assumes the loaded file fits into the size of this dataset
        self.observations = payload['observations']
        self.actions = payload['actions']
        self.arm_raw_angles = payload['arm_raw_angles']
        self.arm_raw_velocities = payload['arm_raw_velocities']
        self.motor_rates = payload['motor_rates']
        self.motor_forces = payload['motor_forces']
        self.counter = payload['counter']

        print(f"Successfully loaded {self.counter} transitions.")


def validate_point(x, y, arm_lengths, angles_theo):
    total_length = arm_lengths[0] + arm_lengths[1] + arm_lengths[2]
    if np.sqrt(x*x + y*y) > total_length:
        return False
    if y < -0.9:  # 90% of arm length
        return False
    l1, l2, l3 = arm_lengths
    t1, t2, t3 = angles_theo
    if l1 * np.sin(t1) < -0.9:
        return False
    if l1 * np.sin(t1) + l2 * np.sin(t1 + t2) < -0.9:
        return False
    return True


def generate_target_angles(robot_arm_lengths_normalized, info, env):
    while True:
        x = int(np.random.uniform(low=0, high=env.WIDTH))
        y = int(np.random.uniform(low=0, high=env.HEIGHT))
        target_pos_world = np.array([x, y])
        target_pos_relative = (target_pos_world - np.array([env.BASE_X, env.BASE_Y - env.ARM_LENGTH])) / env.ARM_LENGTH
        target_pos_relative[1] = -target_pos_relative[1]

        target_angles_theoretical = env.controller.position_to_angles(
            target_pos_relative[0],
            target_pos_relative[1],
            robot_arm_lengths_normalized,
            info["arm_raw_angles"]
        )

        if validate_point(target_pos_relative[0], target_pos_relative[1], robot_arm_lengths_normalized, target_angles_theoretical):
            break
    env.target_pos_world = target_pos_world

    target_angles = theoretical2simulation_angles(target_angles_theoretical)
    target_angles_ = []
    for curr_angle, target_angle in zip(info["arm_raw_angles"], target_angles):
        if abs(target_angle - curr_angle) > np.pi:
            target_angles_.append(simp_angle(target_angle))
        else:
            target_angles_.append(target_angle)
    target_angles = np.array(target_angles_)

    return target_angles

def do_episode(episode_data_):
    # TODO: Put this render into the environment
    is_render = False
    env = RobotArmEnv(is_render)
    robot_arm_lengths_normalized = [1.0, 1.0, 0.5]

    done = False
    observation, info = env.reset(render=is_render)
    target_angles = generate_target_angles(robot_arm_lengths_normalized, info, env)

    while not done:
        # action = np.array([0.0, 0.0, 0.0])

        if env.timesteps == 500:
            target_angles = generate_target_angles(robot_arm_lengths_normalized, info, env)

        values, is_stop_motors = env.controller.do_angles(
            target_angles,
            info["arm_raw_angles"],
            info["arm_raw_velocities"]
        )

        if not is_stop_motors:
            action = values

        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_data_.update(observation, action, observation_, done, info)

        observation = observation_

        if is_render:
            env.render()
    env.close()
    del env
    print(episode_data_.counter)


if __name__ == "__main__":
    save_location = "../data/episodes.data"
    env_ = RobotArmEnv(False)
    n_episodes = 3
    dataset_size_ = env_.MAX_TIMESTEPS * n_episodes
    episode_data = Dataset(dataset_size_, env_.observation_space[0], env_.action_space[0])
    del env_
    
    for i in range(n_episodes):
        do_episode(episode_data)
        # if i % 10 == 0:
        print(f"Episode {i} out of {n_episodes}")

    episode_data.save(save_location)
