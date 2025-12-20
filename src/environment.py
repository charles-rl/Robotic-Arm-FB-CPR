import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import mink
from scipy.spatial.transform import Rotation as R

POSITIONS = {
    "Center": {
        "air_pos": [-0.25, 0.0, 0.63],
        "air_qpos": np.array([-0.0256, -1.6549, 1.0216, 0.5479, -1.5147, 0.4505]),
        "table_pos": [-0.25, 0.0, 0.43],
        "table_qpos": np.array([-0.1105, -0.6478, 1.2145, 0.6488, -1.5149, 0.7772])
    },
    "Left": {
        "air_pos": [-0.25, 0.15, 0.63],
        "air_qpos": np.array([-0.7104, -0.9144, 0.7525, 0.2296, 1.6324, 0.4714]),
        "table_pos": [-0.25, 0.15, 0.43],
        "table_qpos": np.array([-0.8048, -0.1693, 0.9395, 0.2268, -1.6803, 0.4066])
    },
    "Right": {
        "air_pos": [-0.25, -0.15, 0.63],
        "air_qpos": np.array([0.7957, -0.9497, 0.8520, 0.0732, 1.5838, 0.4697]),
        "table_pos": [-0.25, -0.15, 0.43],
        "table_qpos": np.array([0.8181, -0.1458, 0.9392, 0.1610, 1.6450, 0.5912])
    },
    "Far_Left": {
        "air_pos": [-0.1, 0.1, 0.63],
        "air_qpos": np.array([-0.2480, 0.0105, 0.0497, -0.0548, 1.6172, 0.6541]),
        "table_pos": [-0.1, 0.1, 0.43],
        "table_qpos": np.array([-0.3462, 0.7594, 0.3037, -0.7716, -1.6587, 0.5156])
    },
    "Far_Right": {
        "air_pos": [-0.1, -0.1, 0.63],
        "air_qpos": np.array([0.2271, -0.0655, 0.1389, -0.0051, -1.5699, 0.6396]),
        "table_pos": [-0.1, -0.1, 0.43],
        "table_qpos": np.array([0.2556, 0.8585, 0.2434, -0.8558, -1.5711, 0.7101])
    },
}

class RobotArmEnv(gymnasium.Env):
    max_iters = 2 # tiny changes per timestep
    FRAME_SKIP = 17
    def __init__(self, render_mode=None, reward_type=None, task="base", control_mode="delta_joint_position"):
        """
        ## Robot Specification
        - **Robot:** SO-101 (Standard Open Manipulator).
        - **Dimensions:** 6 Total Degrees of Freedom (DOF).
            - **Arm:** 6 Joints (Base, Shoulder, Upper_Arm, Lower_Arm, Wrist, Gripper).
        - Simulation time step: 0.002 seconds
        - Robot Control Frequency: 30 Hz (0.034 seconds, 29.41 FPS)

        ## FB State Space (59 dims)
        - **Robot Proprioception (29 dims):**
            - **Joint Positions (5 dims):**`
            - **Joint Velocities (5 dims):** For the FB representation to understand momentum and dynamics.
            - **Gripper State (2 dims):** Joint Position and Velocity.
            - **End-Effector (EE) Position (3 dims):** Cartesian coordinates relative to the base. $(x, y, z)$        ```
                - *Note:* Most VLA models (including ACT, Diffusion Policy, and SmolVLA) include EE position in the state vector. It helps the network learn faster because it doesn't have to internally calculate Forward Kinematics (converting angles to XYZ) from scratch.
            - **End-Effector (EE) Orientation (6 dims):** **Rotate6D** representation. Recent change.
            - **End-Effector (EE) Velocities (6 dims):** Angular/Linear velocities. Recent change.
            - **Continuous Contact Sensors (2 dims):** Continuous value indicating if one of the finger pads is touching anything. So one for the moving jaw and the other would be for the not moving jaw.
                - Without this, the FB representation has to infer contact purely from the visual overlap of the Gripper XYZ and Cube XYZ, which is chemically noisy in simulation. Explicit sensors make the transition dynamics much sharper.
        - **Object States (30 dims):**
            - **Cube A & Cube B Position (3 dims each):** $(x, y, z)$
            - **Cube A & Cube B Relative Position to End-Effector (3 dims each):** ee_pos - cube_pos
            - **Cube A & Cube B Orientation (6 dims each):** **Rotate6D** representation. This is continuous and easier for NNs to learn than Quaternions.
                - Idea from this paper: https://arxiv.org/pdf/2510.10274
                - **Rotate6D** reference paper: https://arxiv.org/pdf/1812.07035
            - **Cube A & Cube B Velocities (6 dims each):** Angular/Linear velocities.
        ## Task Specific States (8 dims)
        - **Dynamic Goal Position (3 dims):**
            - **Description:** The absolute coordinates of where the active object (or end-effector) should end up.
            - **Logic per Task:**
                - **Reach:** Random point in space (spherical sampling).
                - **Lift:** Current_Cube_Pos + [0, 0, 0.2]. (Dynamic targets help learning).
                - **Place:** Fixed_Box_Pos.
                - **Stack:** Current_Bottom_Cube_Pos + [0, 0, Cube_Height].
            - **Why:** This allows a single policy network to handle all spatial tasks. The network learns to minimize the distance between the Target Object and this Goal Vector.
        - **Target Object Indicator (2 dims):**
            - **Description:** A One-Hot vector indicating which object is currently relevant.
                - `[1, 0]` = Focus on Cube A.
                - `[0, 1]` = Focus on Cube B.
                - `[0, 0]` = Focus on End-Effector (for Reach tasks).
            - **Why:** Since your Base State includes both cubes, the agent needs to know which one to pick up. Without this, if you say "Place at Box 1," the agent won't know which cube to put there.
        - **Relative Distance: Target Object -> Goal (3 dims)**
            - **Math:** Goal_Pos - Current_Target_Object_Pos.
            - **Why?** Just like you added EE - Cube to the Base State to help the robot find the cube, adding Cube - Box to the Task State helps the robot find the box.
            - Without this: The network has to internally subtract Object_Pos (from dim 30) and Goal_Pos (from dim 60). It can do this, but giving it explicitly speeds up training significantly.

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
            base - (65 dims) default for FB training (600 timesteps)
            reach - reach random target position (100 timesteps)
            lift - pick cube and hover (200 timesteps)
            pick - pick cube and reach target position or box (300 timesteps)
            stack - stack cubes on top of each other (500 timesteps)
        reward types:
            sparse
            dense
        """
        print(f"Environment Initialized with {render_mode}, {reward_type}, {task}, {control_mode}")
        self.renderer = None
        self.model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.05,
            lm_damping=1e-6,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
        self.mink_tasks = [self.end_effector_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model=self.configuration.model)]
        self.mocap_id = self.model.body("target").mocapid[0]
        self.viewer = None
        self.task = task
        self.reward_type = reward_type

        self.joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        self.joint_min = self.model.jnt_range[:6, 0]
        self.joint_max = self.model.jnt_range[:6, 1]
        self.cube_a_id = self.model.body("cube_a").id
        self.cube_b_id = self.model.body("cube_b").id
        self.gripper_body_id = self.data.body("gripper").id
        self.left_jaw_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_sensor")
        self.right_jaw_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_sensor")
        self.dynamic_goal_pos = np.zeros(3)
        self.object_focus_one_hot = np.zeros(2)
        self.object_focus_idx = None
        self.object_start_height = None
        control_modes = ("delta_joint_position", "delta_end_effector")
        self.control_mode = control_modes.index(control_mode)
        self.render_mode = render_mode
        self.total_episodes = 0

        # Initialize positions - will be set randomly in reset()
        self._arm_start_pos = np.array([
            0.0,
            self.joint_min[1],
            self.joint_max[2],
            np.random.uniform(0.4, 0.6),
            np.pi / 2,
            self.joint_min[-1],  # Close gripper
        ], dtype=np.float32)
        # TODO: Double check rewards if it is true
        # TODO: Rescale the positions here to fit according to base pos
        self.cube_start_positions = [
            [-0.25, 0.00, 0.43],  # 1. Center
            [-0.25, 0.15, 0.43],  # 2. Left
            [-0.25, -0.15, 0.43],  # 3. Right
            [-0.10, 0.10, 0.43],  # 4. Far Left
            [-0.10, -0.10, 0.43],  # 5. Far Right
        ]
        self.cube_neutral_start_position = [-0.1, 0.00, 0.43]
        self.action_scale = np.pi / 20.0
        self.ee_pos_scale = 0.001
        self.ee_rot_scale = np.pi / 150.0
        self.delta_quat = np.zeros(4)
        self.rot6d_mat_obs = np.zeros(9)
        self.rot6d_idxs = np.array([0, 3, 6, 1, 4, 7], dtype=np.int32)
        self.base_pos_world = None
        self.timesteps = 0

        self.max_episode_steps = 600
        obs_dims = 65
        actions_dims = 6
        # Task specific states included
        if self.task != "base":
            obs_dims += 6
        if self.control_mode == 1:
            # 3 pos + 3d rotation + 1 gripper
            actions_dims = 7

        if self.task == "reach":
            actions_dims -= 1
            self.max_episode_steps = 100
        elif self.task == "lift":
            self.max_episode_steps = 200
        elif self.task == "pick":
            self.max_episode_steps = 300
        elif self.task == "stack":
            self.max_episode_steps = 500

        # 64 bit chosen for 64 bit computers
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dims,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(actions_dims,), dtype=np.float32)

    def _get_cube_data(self, body_id, ee_pos_world):
        rel_pos = self.data.xpos[body_id] - self.base_pos_world
        ee_cube_rel = ee_pos_world - self.data.xpos[body_id]

        # Orientation -> Rotate6D (6 dims)
        quat = self.data.xquat[body_id]
        mujoco.mju_quat2Mat(self.rot6d_mat_obs, quat)
        rot6d = self.rot6d_mat_obs[self.rot6d_idxs].copy()

        # Velocity (6 dims)
        # data.cvel returns 6D spatial velocity: [Angular(3), Linear(3)]
        spatial_vel = self.data.cvel[body_id]
        return np.concatenate([rel_pos, ee_cube_rel, rot6d, spatial_vel])

    def _get_obs(self):
        """
        :return: Unnormalized qpos and qvel, must be normalized outside the environment
        """
        # TODO: Maybe apply sin and cos to the 5th joint because the wrist joint is kinda like a circle anyways
        qpos = self.data.qpos[:6]
        qvel = self.data.qvel[:6]
        # Tip of the jaw position
        ee_pos_world = self.data.site("gripperframe").xpos
        ee_pos_rel = ee_pos_world - self.base_pos_world
        ee_xmat = self.data.site("gripperframe").xmat
        ee_orientation = ee_xmat[self.rot6d_idxs].copy()
        # Close enough to end effector
        ee_cvel = self.data.cvel[self.gripper_body_id].copy()

        left_jaw_force = self.data.sensordata[self.left_jaw_sensor_id]
        right_jaw_force = self.data.sensordata[self.right_jaw_sensor_id]
        left_force_norm = np.tanh(left_jaw_force / 5.0)
        right_force_norm = np.tanh(right_jaw_force / 5.0)
        jaw_forces = np.array([left_force_norm, right_force_norm])

        robot_state = np.concatenate([qpos, qvel, ee_pos_rel, ee_orientation, ee_cvel, jaw_forces])

        cube_a_state = self._get_cube_data(self.cube_a_id, ee_pos_world)
        cube_b_state = self._get_cube_data(self.cube_b_id, ee_pos_world)

        fb_obs = np.concatenate([robot_state, cube_a_state, cube_b_state]).astype(self.observation_space.dtype)
        if self.task != "base":
            if self.object_focus_idx == 0:
                target_cube_state, distractor_cube_state = cube_a_state, cube_b_state
            else:
                target_cube_state, distractor_cube_state = cube_b_state, cube_a_state
            task_obs = np.concatenate([robot_state, target_cube_state, distractor_cube_state, self._get_task_states(ee_pos_world)]).astype(self.observation_space.dtype)
            return task_obs, fb_obs
        else:
            return fb_obs, fb_obs

    def _get_task_states(self, ee_pos_world):
        task_specific_states = np.zeros(6)
        if self.task == "reach":
            dynamic_goal_rel = self.dynamic_goal_pos - self.base_pos_world
            task_specific_states[0:3] = dynamic_goal_rel
            # end effector relative to goal position
            task_specific_states[3:6] = ee_pos_world - self.dynamic_goal_pos
        elif self.task == "lift":
            dynamic_goal_rel = self.dynamic_goal_pos - self.base_pos_world
            task_specific_states[0:3] = dynamic_goal_rel
            target_cube_id = self.cube_a_id if self.object_focus_idx == 0 else self.cube_b_id
            target_cube_pos = self.data.xpos[target_cube_id].copy()
            obj_to_goal = self.dynamic_goal_pos - target_cube_pos
            task_specific_states[3:6] = obj_to_goal
        return task_specific_states

    def _sample_target(self):
        """
        Generates a random target position using spherical coordinates.
        Adjust the limits based on the SO-101 dimensions.
        """
        if self.task == "reach":
            # Rho (Radius): Min to avoid hitting self, Max to stay within reach
            min_rho, max_rho = 0.2, 0.4

            # Theta (Azimuth/Yaw): Left-to-Right reach
            # 0 is forward. -pi/2 is right, pi/2 is left.
            min_theta, max_theta = -4 * np.pi / 9, 4 * np.pi / 9

            # Phi (Elevation/Pitch): Up-and-Down reach
            # 0 is horizontal table level, pi/2 is straight up.
            min_phi, max_phi = np.pi / 36, 4 * np.pi / 9

            rho = np.random.uniform(min_rho, max_rho)
            theta = np.random.uniform(min_theta, max_theta)
            phi = np.random.uniform(min_phi, max_phi)

            self.dynamic_goal_pos[0] = rho * np.cos(phi) * np.cos(theta)
            self.dynamic_goal_pos[1] = rho * np.cos(phi) * np.sin(theta)
            self.dynamic_goal_pos[2] = rho * np.sin(phi)

            self.dynamic_goal_pos += self.base_pos_world

        elif self.task == "lift":
            target_cube_id = self.cube_a_id if self.object_focus_idx == 0 else self.cube_b_id
            cube_pos_world = self.data.xpos[target_cube_id].copy()

            lift_height = np.random.uniform(0.15, 0.25)

            self.dynamic_goal_pos = cube_pos_world.copy()
            self.dynamic_goal_pos[2] += lift_height

    def _compute_reward(self, has_fallen=False):
        """
        Calculates reward based on the current task.
        SmolVLA/LeRobot Style: 0.5 for partial success, 1.0 for full success.
        """

        ee_pos = self.data.site("gripperframe").xpos

        if self.task == "reach":
            distance = np.linalg.norm(ee_pos - self.dynamic_goal_pos)

            if self.reward_type == "sparse":
                return 1.0 if distance < 0.05 else 0.0
            elif self.reward_type == "dense":
                return 1.0 - np.tanh(3.0 * distance)

        elif self.task == "lift":
            if has_fallen:
                return -10.0

            target_cube_id = self.cube_a_id if self.object_focus_idx == 0 else self.cube_b_id
            cube_pos = self.data.xpos[target_cube_id].copy()

            dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)

            dist_from_table = cube_pos[2] - self.object_start_height
            is_lifted = dist_from_table > 0.05

            dist_z = abs(cube_pos[2] - self.dynamic_goal_pos[2])
            is_goal = dist_z < 0.05

            if self.reward_type == "sparse":
                reward = 0.0
                # Using has_solid_grip instead of distance check for robustness
                if is_lifted: reward += 0.5
                if is_lifted and is_goal: reward += 0.5
                return reward

            elif self.reward_type == "dense":
                reach_reward = 1.0 - np.tanh(10.0 * dist_ee_cube)
                hoist_reward = np.tanh(7.0 * dist_from_table)

                dist_to_goal = np.linalg.norm(cube_pos - self.dynamic_goal_pos)
                precision_reward = 1.0 - np.tanh(10.0 * dist_to_goal)
                if dist_to_goal < 0.05:
                    precision_reward += 2.0

                return reach_reward + precision_reward + hoist_reward

        return 0.0

    def step(self, action: np.ndarray):
        if self.control_mode == 0:
            current_arm_ctrl = self.data.ctrl[:5].copy()
            target_arm_ctrl = current_arm_ctrl + (action[:5] * self.action_scale)

            target_arm_ctrl = np.clip(target_arm_ctrl, self.joint_min[:5], self.joint_max[:5])

            self.data.ctrl[:5] = target_arm_ctrl
        elif self.control_mode == 1:
            # Sync configuration with current physical state
            self.configuration.update(self.data.qpos)

            current_ee_pos = self.data.site("gripperframe").xpos.copy()
            new_mocap_pos = current_ee_pos + (action[:3] * self.ee_pos_scale)
            new_mocap_pos[0] = np.clip(new_mocap_pos[0], -0.6 + self.base_pos_world[0], 0.6 + self.base_pos_world[0])  # X
            new_mocap_pos[1] = np.clip(new_mocap_pos[1], -0.6 + self.base_pos_world[1], 0.6 + self.base_pos_world[1])  # Y
            new_mocap_pos[2] = np.clip(new_mocap_pos[2], -0.02 + self.base_pos_world[2], 0.6 + self.base_pos_world[2])
            self.data.mocap_pos[self.mocap_id] = new_mocap_pos

            current_xmat = self.data.site("gripperframe").xmat.copy()
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_xmat)
            quat = self._apply_delta_orientation(current_quat, action[3:6])
            self.data.mocap_quat[self.mocap_id] = quat

            T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
            self.end_effector_task.set_target(T_wt)

            dt_solver = 0.002 * self.FRAME_SKIP
            for i in range(self.max_iters):
                vel = mink.solve_ik(
                    self.configuration,
                    self.mink_tasks,
                    dt_solver,
                    solver="daqp",
                    limits=self.limits
                )
                self.configuration.integrate_inplace(vel, dt_solver)
                # Removed this for speed and less overhead
                # err = self.end_effector_task.compute_error(self.configuration)
                # pos_achieved = np.linalg.norm(err[:3]) <= 1e-3
                # ori_achieved = np.linalg.norm(err[3:]) <= 1e-4
                # if pos_achieved and ori_achieved:
                #     break
            self.data.ctrl[:5] = self.configuration.q[:5]

        if self.task == "reach":
            self.data.ctrl[5] = self.joint_min[5]  # Permanently closed
        else:
            gripper_action = action[-1]

            if gripper_action > 0.1:
                self.data.ctrl[5] = self.joint_max[5]  # Open
            elif gripper_action < -0.1:
                self.data.ctrl[5] = self.joint_min[5]  # Closed
            # If between -0.1 and 0.1, keep previous state (do nothing)

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.timesteps += 1

        target_cube_id = self.cube_a_id if self.object_focus_idx == 0 else self.cube_b_id
        cube_z = self.data.xpos[target_cube_id][2]

        # Table is usually at Z=0. If cube goes below -0.05, it fell off.
        # TODO: If any cube has fallen then terminate
        has_fallen = False
        if cube_z < self.base_pos_world[2] - 0.1:
            has_fallen = True

        truncated = bool(self.timesteps >= self.max_episode_steps)
        terminated = has_fallen

        reward = self._compute_reward(has_fallen=has_fallen)

        if self.task != "base":
            obs, fb_obs = self._get_obs()
            info = self._get_info()
            info["fb_obs"] = fb_obs
        else:
            _, obs = self._get_obs()
            info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _apply_delta_orientation(self, current_quat, euler_action):
        """
        Applies a local delta Euler rotation to the current quaternion.
        """
        scaled_action = euler_action * self.ee_rot_scale
        mujoco.mju_euler2Quat(self.delta_quat, scaled_action, "xyz")
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, current_quat, self.delta_quat)
        return new_quat

    def _rot6d_to_mat_quat(self, rot6d):
        """Unused because I have decided to use delta euler angles"""
        a1, a2 = rot6d[:3], rot6d[3:]
        x = a1 / (np.linalg.norm(a1) + 1e-8)
        z = np.cross(x, a2)
        z = z / (np.linalg.norm(z) + 1e-8)
        y = np.cross(z, x)
        mat = np.stack([x, y, z], axis=1)
        mujoco.mju_mat2Quat(self.quat_action, mat.flatten())
        return mat

    def _set_cube_pos_quat(self, cube_joint_name, pos, quat):
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, cube_joint_name)
        adr = self.model.jnt_qposadr[cube_id]
        self.data.qpos[adr: adr + 3] = pos
        self.data.qpos[adr + 3: adr + 7] = quat

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        # ========================================================================
        chosen_indices = np.random.choice(len(self.cube_start_positions), size=2, replace=False)
        cube_joints = ("cube_a_joint", "cube_b_joint")
        active_cube_idx = 0 if np.random.rand() > 0.5 else 1
        other_cube_idx = 1 - active_cube_idx
        self.object_focus_idx = active_cube_idx
        self.object_focus_one_hot = np.zeros(2)
        self.object_focus_one_hot[self.object_focus_idx] = 1.0

        active_cube_name = cube_joints[active_cube_idx]
        other_cube_name = cube_joints[other_cube_idx]
        stage_roll = np.random.rand()

        loc_name = np.random.choice(list(POSITIONS.keys()))
        loc_data = POSITIONS[loc_name]

        if self.total_episodes < 150:
            stage_probs = [0.8, 0.2, 0.0]  # 80% Hold, 20% Hoist, 0% Random
        elif self.total_episodes < 400:
            stage_probs = [0.2, 0.6, 0.2]
        else:
            stage_probs = [0.1, 0.2, 0.7]  # 10% Hold, 20% Hoist, 70% Random

        if stage_roll < stage_probs[0]:
            self._set_cube_pos_quat(other_cube_name, self.cube_neutral_start_position, np.array([1, 0, 0, 0]))
            self._set_cube_pos_quat(active_cube_name, loc_data["air_pos"], np.array([1, 0, 0, 0]))

            qpos = loc_data["air_qpos"].copy()
            self.data.qpos[:6] = qpos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_min[5]  # auto close
            self.dynamic_goal_pos = np.array(loc_data["air_pos"])

        elif stage_roll < stage_probs[1]:
            self._set_cube_pos_quat(other_cube_name, self.cube_neutral_start_position, np.array([1, 0, 0, 0]))
            self._set_cube_pos_quat(active_cube_name, loc_data["table_pos"], np.array([1, 0, 0, 0]))

            qpos = loc_data["table_qpos"].copy()
            self.data.qpos[:6] = qpos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_max[5] # open
            self.dynamic_goal_pos = np.array(loc_data["air_pos"])
        else:
            pos_a = self.cube_start_positions[chosen_indices[0]]
            pos_b = self.cube_start_positions[chosen_indices[1]]

            # Set Cube A
            # Randomize Orientation
            theta = np.random.uniform(-np.pi, np.pi)
            quat_a = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])
            self._set_cube_pos_quat("cube_a_joint", pos_a, quat_a)
            # Set Orientation [w, x, y, z] - Identity Quaternion (No rotation)
            # self.data.qpos[adr_a + 3: adr_a + 7] = [1, 0, 0, 0]

            # Set Cube B2
            theta = np.random.uniform(-np.pi, np.pi)
            quat_b = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])
            self._set_cube_pos_quat("cube_b_joint", pos_b, quat_b)
            # self.data.qpos[adr_b + 3: adr_b + 7] = [1, 0, 0, 0]

            self.data.qpos[:6] = self._arm_start_pos
            self.data.qvel[:6] = 0.0  # no arm movement
            self.data.ctrl[:6] = self.data.qpos[:6].copy()

            self._sample_target()

        # ========================================================================
        # Update Kinematics
        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        mink.move_mocap_to_frame(self.model, self.data, "target", "gripperframe", "site")

        # Settle Physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.base_pos_world = self.data.body("base").xpos

        target_cube_id = self.cube_a_id if self.object_focus_idx == 0 else self.cube_b_id
        target_cube_pos = self.data.xpos[target_cube_id].copy()
        self.object_start_height = target_cube_pos[2]

        self.timesteps = 0
        self.total_episodes += 1

        if self.task != "base":
            obs, fb_obs = self._get_obs()
            info = self._get_info()
            info["fb_obs"] = fb_obs
        else:
            _, obs = self._get_obs()
            info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == "human":
            # 1. Launch Viewer if it doesn't exist yet
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False)

            if self.viewer.is_running():
                self._render_geoms(self.viewer.user_scn)
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                # Hack: Renderer constructor doesn't store cam_id permanently in some versions,
                # so we update the scene with the specific camera
                self.renderer.update_scene(self.data, camera="main_cam")
            else:
                self.renderer.update_scene(self.data, camera="main_cam")

            self._render_geoms(self.renderer.scene)
            return self.renderer.render()

    def _render_geoms(self, scene):
        # TODO: Maybe remove mocap body target if not used for control modes
        scene.ngeom += 2
        # Render left jaw sensor
        left_sensor_pos = self.data.site("left_pad_site").xpos
        left_sensor_mat = self.data.site("left_pad_site").xmat
        mujoco.mjv_initGeom(
            scene.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.model.site("left_pad_site").size,  # Radius 0.02
            pos=left_sensor_pos,  # Position at our target
            mat=left_sensor_mat,  # Identity matrix for orientation
            rgba=[0, 1, 0, 0.3]  # Green color, opaque
        )
        # Render right jaw sensor
        right_sensor_pos = self.data.site("right_pad_site").xpos
        right_sensor_mat = self.data.site("right_pad_site").xmat
        mujoco.mjv_initGeom(
            scene.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.model.site("right_pad_site").size,  # Radius 0.02
            pos=right_sensor_pos,  # Position at our target
            mat=right_sensor_mat,  # Identity matrix for orientation
            rgba=[0, 1, 0, 0.3]  # Green color, opaque
        )
        if self.task != "base":
            scene.ngeom += 1
            if self.task == "reach":
                # Render target position
                mujoco.mjv_initGeom(
                    scene.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],  # Radius 2cm
                    pos=self.dynamic_goal_pos,  # <--- Use the class variable
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.4]
                )
            elif self.task == "lift":
                color = [1, 0, 0, 0.4] if self.object_focus_idx == 0 else [0, 0, 1, 0.4]
                # Render target position
                mujoco.mjv_initGeom(
                    scene.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],  # Radius 2cm
                    pos=self.dynamic_goal_pos,  # <--- Use the class variable
                    mat=np.eye(3).flatten(),
                    rgba=color
                )

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
    end_effector_testing = False
    if not end_effector_testing:
        env = RobotArmEnv(render_mode="human", task="lift", control_mode="delta_end_effector")
        obs, info = env.reset(seed=0)
        done = False
        while not done:
            action = np.zeros(env.action_space.shape[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
        env.close()

    else:
        import time

        # Initialize with human rendering to see what's happening
        env = RobotArmEnv(render_mode="human", task="lift", control_mode="delta_end_effector")
        obs, info = env.reset(seed=42)

        print("=== STARTING ACTION TEST ===")


        def run_sequence(action_vec, steps, description):
            print(f"Testing: {description}")
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(action_vec)
                env.render()
                # Sleep slightly to make visual debugging easier (remove for real training)
                time.sleep(0.1)

                # Action definition: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]


        # 1. Test Z-Axis (Up/Down)
        # Move Up
        run_sequence(np.array([0, 0, 1.0, 0, 0, 0, 1.0]), 20, "Moving UP")
        # Move Down
        run_sequence(np.array([0, 0, -1.0, 0, 0, 0, 1.0]), 20, "Moving DOWN")

        # 2. Test Y-Axis (Left/Right)
        # Move Left (positive Y usually)
        run_sequence(np.array([0, 1.0, 0, 0, 0, 0, 1.0]), 20, "Moving LEFT")
        # Move Right
        run_sequence(np.array([0, -1.0, 0, 0, 0, 0, 1.0]), 20, "Moving RIGHT")

        # 3. Test X-Axis (Forward/Back)
        # Move Forward
        run_sequence(np.array([1.0, 0, 0, 0, 0, 0, 1.0]), 20, "Moving FORWARD")
        # Move Backward
        run_sequence(np.array([-1.0, 0, 0, 0, 0, 0, 1.0]), 20, "Moving BACKWARD")

        # 4. Test Orientation (Wrist)
        # Roll
        run_sequence(np.array([0, 0, 0, 1.0, 0, 0, 1.0]), 60, "Rolling Wrist (+)")
        run_sequence(np.array([0, 0, 0, -1.0, 0, 0, 1.0]), 60, "Rolling Wrist (-)")

        # Pitch (Up/Down Tilt)
        run_sequence(np.array([0, 0, 0, 0, 1.0, 0, 1.0]), 60, "Pitching Wrist (+)")
        run_sequence(np.array([0, 0, 0, 0, -1.0, 0, 1.0]), 60, "Pitching Wrist (-)")

        # 5. Test Gripper
        # Close
        run_sequence(np.array([0, 0, 0, 0, 0, 0, -1.0]), 30, "Closing Gripper")
        # Open
        run_sequence(np.array([0, 0, 0, 0, 0, 0, 1.0]), 30, "Opening Gripper")

        print("=== TEST COMPLETE ===")
        env.close()
