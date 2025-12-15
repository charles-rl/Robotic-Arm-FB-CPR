import numpy as np
import gymnasium
import mujoco
import mujoco.viewer

class RobotArmEnv(gymnasium.Env):
    MAX_TIMESTEPS = 1000
    FRAME_SKIP = 17
    def __init__(self, render_mode=None, reward_type="sparse", task="reach", control_mode="delta_joint_position"):
        """
        ## Robot Specification
        - **Robot:** SO-101 (Standard Open Manipulator).
        - **Dimensions:** 6 Total Degrees of Freedom (DOF).
            - **Arm:** 6 Joints (Base, Shoulder, Upper_Arm, Lower_Arm, Wrist, Gripper).
        - Simulation time step: 0.002 seconds
        - Robot Control Frequency: 30 Hz (0.034 seconds, 29.41 FPS)

        ## State Space (45 dims)
        - **Robot Proprioception (15 dims):**
            - **Joint Positions (5 dims):** Normalized to based on physical joint limits. `[-1, 1]`
            - **Joint Velocities (5 dims):** For the FB representation to understand momentum and dynamics.
            - **Gripper State (2 dims):** Joint Position and Velocity.
            - **End-Effector (EE) Position (3 dims):** Cartesian coordinates relative to the base. $(x, y, z)$        ```
                - *Note:* Most VLA models (including ACT, Diffusion Policy, and SmolVLA) include EE position in the state vector. It helps the network learn faster because it doesn't have to internally calculate Forward Kinematics (converting angles to XYZ) from scratch.
        - **Object States (30 dims):**
            - **Cube A & Cube B Position (3 dims each):** $(x, y, z)$
            - **Cube A & Cube B Orientation (6 dims each):** **Rotate6D** representation. This is continuous and easier for NNs to learn than Quaternions.
                - Idea from this paper: https://arxiv.org/pdf/2510.10274
                - **Rotate6D** reference paper: https://arxiv.org/pdf/1812.07035
            - **Cube A & Cube B Velocities (6 dims each):** Linear/Angular velocities

        ## Action Space
        *Design Choice: We will code the environment to support switching between Option B and C to test which yields better FB representations.*
        - **Option A: Absolute Joint Position (SmolVLA)**
            - **Action:** Direct target joint angles `[q1, q2, q3, q4, q5, gripper]`.
            - **Context:** This is what SmolVLA uses.
            - SmolVLA only makes this work because they use **Action Chunking** (predicting 50 steps at a time) and Temporal Ensembling to smooth the path. Without chunking, a single-step policy using Absolute Position will cause the robot to jitter and vibrate violently.
            - Documented for comparison, but will not be used
        - **Option B: Delta End-Effector (Gymnasium)**
            - **Action:** `[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]`.
            - **Logic:** The network commands the hand to move in 3D space. We use an Inverse Kinematics (IK) solver to figure out the joint angles.
            - **Pros:** Very easy for the policy to learn "Pick" (just move Z down).
            - **Cons:** Harder to implement (needs stable IK).
            - **Will be used in the beginning to test policy training**
        - **Option C: Delta Joint Position (Changed SmolVLA)**
            - **Action:** `[dq1, dq2, dq3, dq4, dq5, gripper]`.
            - **Logic:** `next_pos = current_pos + (action * scale)`.
            - **Pros:** No IK needed. Smoother than Option A.
            - **Primary candidate for experimentation.**
        - **Binary Gripper**
            - Simulated grippers are finicky.
            - Logic:
                - `If action > 0:` Command max width (Open).
                - `If action < 0:` Command min width (Closed).
            - *Delta control for grippers usually results in the object slipping because the network "forgets" to keep applying closing pressure.*

        Joint Names:
            shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        Link/Body Names:
            base, shoulder, upper_arm, lower_arm, wrist, gripper, moving_jaw_so101_v1
        Control Modes:
            Option B: delta_end_effector
            Option C: delta_joint_position
        Tasks:
            base - (45 dims) default for FB training
            reach - for testing, reach random target position, observation dim has relative target position
            pick - for testing, pick cube and reach target position, observation dim has relative target position
        reward types:
            sparse
            dense
        """
        self.model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.task = task
        self.reward_type = reward_type

        self.joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        self.joint_min = self.model.jnt_range[:6, 0]
        self.joint_max = self.model.jnt_range[:6, 1]
        self.cube_a_id = None
        self.cube_b_id = None
        self.target_pos = np.zeros(3)
        control_modes = ("delta_joint_position", "delta_end_effector")
        self.control_mode = control_modes.index(control_mode)
        self.render_mode = render_mode

        # Initialize positions - will be set randomly in reset()
        self._arm_start_pos = np.array([
            0.0,
            self.joint_min[1],
            self.joint_max[2],
            np.random.uniform(0.4, 0.6),
            0.0,
            self.joint_min[-1],  # Close gripper
        ], dtype=np.float32)

        self.cube_start_positions = [
            [-0.25, 0.00, 0.43],  # 1. Center
            [-0.25, 0.15, 0.43],  # 2. Left
            [-0.25, -0.15, 0.43],  # 3. Right
            [-0.10, 0.10, 0.43],  # 4. Far Left
            [-0.10, -0.10, 0.43],  # 5. Far Right
        ]
        self.rot6d_mat = np.zeros(9)
        self.rot6d_idxs = np.array([0, 3, 6, 1, 4, 7], dtype=np.int32)
        self.base_pos_world = None
        self.timesteps = 0

        obs_dims = 45
        actions_dims = 6
        if self.task == "reach":
            obs_dims += 3
            actions_dims -= 1

        # 64 bit chosen for 64 bit computers
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dims,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(actions_dims,), dtype=np.float32)

    def _get_cube_data(self, body_id):
        rel_pos = self.data.xpos[body_id] - self.base_pos_world

        # Orientation -> Rotate6D (6 dims)
        quat = self.data.xquat[body_id]
        mujoco.mju_quat2Mat(self.rot6d_mat, quat)
        rot6d = self.rot6d_mat[self.rot6d_idxs].copy()

        # Velocity (6 dims)
        # data.cvel returns 6D spatial velocity: [Angular(3), Linear(3)]
        spatial_vel = self.data.cvel[body_id]
        return np.concatenate([rel_pos, rot6d, spatial_vel])

    def _get_obs(self):
        """
        :return: Unnormalized qpos and qvel, must be normalized outside the environment
        """
        qpos = self.data.qpos[:6]
        qvel = self.data.qvel[:6]
        # Tip of the jaw position
        ee_pos_world = self.data.site("gripperframe").xpos
        ee_pos_rel = ee_pos_world - self.base_pos_world

        robot_state = np.concatenate([qpos, qvel, ee_pos_rel])

        cube_a_state = self._get_cube_data(self.cube_a_id)
        cube_b_state = self._get_cube_data(self.cube_b_id)

        extra_state = np.array([])
        if self.task == "reach":
            target_pos_rel = self.target_pos - self.base_pos_world
            extra_state = target_pos_rel

        obs = np.concatenate([robot_state, cube_a_state, cube_b_state, extra_state]).astype(self.observation_space.dtype)
        return obs

    def _sample_target(self):
        """
        Generates a random target position using spherical coordinates.
        Adjust the limits based on the SO-101 dimensions.
        """
        # 1. Define Limits (SO-101 Specifics)
        # Rho (Radius): Min to avoid hitting self, Max to stay within reach
        min_rho, max_rho = 0.15, 0.35

        # Theta (Azimuth/Yaw): Left-to-Right reach
        # 0 is forward. -pi/2 is right, pi/2 is left.
        min_theta, max_theta = -4 * np.pi / 9, 4 * np.pi / 9

        # Phi (Elevation/Pitch): Up-and-Down reach
        # 0 is horizontal table level, pi/2 is straight up.
        min_phi, max_phi = np.pi / 36, 4 * np.pi / 9

        # 2. Random Sampling
        rho = np.random.uniform(min_rho, max_rho)
        theta = np.random.uniform(min_theta, max_theta)
        phi = np.random.uniform(min_phi, max_phi)

        # 3. Spherical to Cartesian Conversion (Z-up)
        # x = forward (depth), y = side (width), z = up (height)
        self.target_pos[0] = rho * np.cos(phi) * np.cos(theta)
        self.target_pos[1] = rho * np.cos(phi) * np.sin(theta)
        self.target_pos[2] = rho * np.sin(phi)

        # Offset: Add to robot base position if base is not at (0,0,0)
        self.target_pos += self.base_pos_world

    def _compute_reward(self):
        """
        Calculates reward based on the current task.
        SmolVLA/LeRobot Style: 0.5 for partial success, 1.0 for full success.
        """

        ee_pos = self.data.site("gripperframe").xpos

        if self.task == "reach":
            distance = np.linalg.norm(ee_pos - self.target_pos)

            if self.reward_type == "sparse":
                return 1.0 if distance < 0.05 else 0.0
            elif self.reward_type == "dense":
                return -distance

        elif self.task == "pick_place":
            pass
            # cube_pos = self.get_body_pos("cube_a")
            # box_pos = self.get_body_pos("box_1")
            #
            # # Stage 1: Grasping (Is the cube lifted off the table?)
            # # Assume table height is 0.0
            # is_lifted = cube_pos[2] > 0.05
            #
            # # Stage 2: Placing (Is cube inside the box radius?)
            # # We ignore Z for distance check to make it easier
            # dist_to_box = np.linalg.norm(cube_pos[:2] - box_pos[:2])
            # is_in_box = dist_to_box < 0.08 and is_lifted
            #
            # reward = 0.0
            # if is_lifted:
            #     reward += 0.5
            # if is_in_box:
            #     reward += 0.5  # Total 1.0
            #
            # return reward

        return 0.0

    def step(self, action: np.ndarray):
        if self.control_mode == 0:
            scale = np.pi / 100.0
            current_arm_ctrl = self.data.ctrl[:5].copy()
            target_arm_ctrl = current_arm_ctrl + (action[:5] * scale)

            target_arm_ctrl = np.clip(target_arm_ctrl, self.joint_min[:5], self.joint_max[:5])

            self.data.ctrl[:5] = target_arm_ctrl

            if self.task == "reach":
                self.data.ctrl[-1] = self.joint_min[-1] # permanently closed
            else:
                gripper_action = action[5]
                if gripper_action > 0:
                    self.data.ctrl[5] = self.joint_max[5]  # Open
                else:
                    self.data.ctrl[5] = self.joint_min[5]  # Closed




        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.timesteps += 1

        terminated = False
        truncated = bool(self.timesteps >= self.MAX_TIMESTEPS)

        reward = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        n_joints = len(self.joint_names)
        self.data.qpos[:n_joints] = self._arm_start_pos
        self.data.qvel[:n_joints] = 0.0  # no arm movement

        # Set the actuators to the same position as well
        self.data.ctrl[:6] = self.data.qpos[:6].copy()

        # Choose cube start position
        chosen_indices = np.random.choice(len(self.cube_start_positions), size=2, replace=False)
        pos_a = self.cube_start_positions[chosen_indices[0]]
        pos_b = self.cube_start_positions[chosen_indices[1]]

        # Set Cube A
        cube_a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_joint")
        adr_a = self.model.jnt_qposadr[cube_a_id]
        self.data.qpos[adr_a: adr_a + 3] = pos_a
        # Set Orientation [w, x, y, z] - Identity Quaternion (No rotation)
        # self.data.qpos[adr_a + 3: adr_a + 7] = [1, 0, 0, 0]
        # Randomize Orientation
        theta = np.random.uniform(-np.pi, np.pi)
        self.data.qpos[adr_a + 3: adr_a + 7] = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

        # Set Cube B2
        cube_b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_joint")
        adr_b = self.model.jnt_qposadr[cube_b_id]
        self.data.qpos[adr_b: adr_b + 3] = pos_b
        # self.data.qpos[adr_b + 3: adr_b + 7] = [1, 0, 0, 0]
        theta = np.random.uniform(-np.pi, np.pi)
        self.data.qpos[adr_b + 3: adr_b + 7] = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

        # Update Kinematics
        mujoco.mj_forward(self.model, self.data)

        # Settle Physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.cube_a_id = self.model.body("cube_a").id
        self.cube_b_id = self.model.body("cube_b").id

        self.base_pos_world = self.data.body("base").xpos

        self._sample_target()
        self.timesteps = 0
        return self._get_obs(), self._get_info()

    def render(self):
        if self.render_mode == "human":
            # 1. Launch Viewer if it doesn't exist yet
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False)

            if self.viewer.is_running():
                self.viewer.user_scn.ngeom = 1
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],  # Radius 2cm
                    pos=self.target_pos,  # <--- Use the class variable
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.5]  # Red, semi-transparent ghost
                )

                self.viewer.sync()

    def _get_info(self):
        """
        TODO: Not tested for now
        Returns dictionary with 'physics' key containing full qpos/qvel state.
        This allows the FB algorithm to reset the sim to this exact moment.
        """
        # Concatenate full qpos and qvel (flattens them into 1D array)
        physics_state = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return {"physics": physics_state}

    def set_physics_state(self, state_vector):
        """
        TODO: Not tested for now
        Sets the MuJoCo simulation state from a flattened qpos+qvel vector.
        Used by FB algorithms to reset context for reward inference.
        """
        # 1. Determine split point
        nq = self.model.nq  # Number of position coordinates
        nv = self.model.nv  # Number of velocity coordinates

        # 2. Split the vector
        new_qpos = state_vector[:nq]
        new_qvel = state_vector[nq: nq + nv]

        # 3. Write to MuJoCo data
        self.data.qpos[:] = new_qpos
        self.data.qvel[:] = new_qvel

        # 4. Forward propagate to update kinematics (xpos, xquat, etc.)
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

if __name__ == "__main__":
    env = RobotArmEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()
