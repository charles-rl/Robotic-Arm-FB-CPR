import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import mink
from collections import deque

class SO101BaseEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    FRAME_SKIP = 17

    def __init__(self, render_mode=None, reward_type="dense", control_mode="delta_joint_position"):
        self.render_mode = render_mode
        self.reward_type = reward_type

        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None

        # Mink / IK Setup
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
        self.dt_solver = 0.002 * self.FRAME_SKIP

        # Robot Specific Constants
        self.joint_min = self.model.jnt_range[:6, 0]
        self.joint_max = self.model.jnt_range[:6, 1]
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.left_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_sensor")
        self.right_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_sensor")
        self.cube_a_id = self.model.body("cube_a").id
        self.cube_b_id = self.model.body("cube_b").id
        self.cube_a_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_joint")
        self.cube_b_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_joint")

        # Control Logic
        control_modes = ("delta_joint_position", "delta_end_effector")
        if control_mode not in control_modes:
            raise ValueError(f"Control mode must be one of {control_modes}")
        self.control_mode = control_modes.index(control_mode)

        self.delta_jnt_scale = np.pi / 20.0
        self.ee_pos_scale = 0.01
        self.ee_rot_scale = np.pi / 150.0
        self.max_iters = 2
        self.last_action_smoothed = None
        self.smoothing_alpha = 0.3
        self.delta_quat = np.zeros(4)
        self.dynamic_goal_pos = np.zeros(3)
        self._arm_start_pos = np.array([
            0.0,
            self.joint_min[1],
            self.joint_max[2],
            0.5,
            np.pi / 2,
            self.joint_min[-1],  # Close gripper
        ], dtype=np.float32)
        self.cube_start_positions = [
            [-0.25, 0.00, 0.43],  # 1. Center
            [-0.25, 0.15, 0.43],  # 2. Left
            [-0.25, -0.15, 0.43],  # 3. Right
            [-0.10, 0.10, 0.43],  # 4. Far Left
            [-0.10, -0.10, 0.43],  # 5. Far Right
        ]

        # Base Helper Variables
        self.base_pos_world = self.data.body("base").xpos
        self.rot6d_idxs = np.array([0, 3, 6, 1, 4, 7], dtype=np.int32)
        self.robot_state_buffer = np.zeros(29, dtype=np.float64)
        self.cube_a_buffer = np.zeros(18, dtype=np.float64)
        self.cube_b_buffer = np.zeros(18, dtype=np.float64)
        self.rot6d_mat_obs = np.zeros(9)
        self.timesteps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset Physics
        mujoco.mj_resetData(self.model, self.data)

        # Reset Action Smoothing
        self.last_action_smoothed = np.zeros(self.action_space.shape[0])

        return None

    def _get_cube_data(self, buffer, body_id):
        ee_pos_world = self.data.site_xpos[self.gripper_site_id]
        # Cube Relative Position (3 dims)
        buffer[0:3] = self.data.xpos[body_id] - self.base_pos_world
        # End Effector relative to Cube (3 dims)
        buffer[3:6] = ee_pos_world - self.data.xpos[body_id]

        # Orientation -> Rotate6D (6 dims)
        quat = self.data.xquat[body_id]
        mujoco.mju_quat2Mat(self.rot6d_mat_obs, quat)
        buffer[6:12] = self.rot6d_mat_obs[self.rot6d_idxs]

        # Velocity (6 dims)
        # data.cvel returns 6D spatial velocity: [Angular(3), Linear(3)]
        buffer[12:18] = self.data.cvel[body_id]
        return buffer

    def _get_robot_state(self):
        """
        :return: Unnormalized qpos and qvel, must be normalized outside the environment
        """
        # 1. Joint State (Indices 0-12)
        self.robot_state_buffer[0:6] = self.data.qpos[:6]
        self.robot_state_buffer[6:12] = self.data.qvel[:6]

        # 2. EE Position (Indices 12-15)
        self.robot_state_buffer[12:15] = self.data.site_xpos[self.gripper_site_id] - self.base_pos_world
        # 3. EE Orientation Rotate6D (Indices 15-21)
        self.robot_state_buffer[15:21] = self.data.site_xmat[self.gripper_site_id][self.rot6d_idxs]
        # 4. EE Velocity (Indices 21-27)
        self.robot_state_buffer[21:27] = self.data.cvel[self.gripper_body_id]

        # Heuristic Sensor Logic
        left_jaw_force = self.data.sensordata[self.left_sensor_id]
        right_jaw_force = self.data.sensordata[self.right_sensor_id]
        # Total grip because sometimes in simulation left force = 0 and right force = 50
        # Sometimes it is left force = 50 and right force = 50
        total_force_norm = np.tanh((left_jaw_force + right_jaw_force) / 30.0)
        # To account for this imbalance we include this feature to help the model
        # In my testing, the right jaw force is always bigger than the left one
        balance_norm = np.tanh((right_jaw_force - left_jaw_force) / 30.0)
        self.robot_state_buffer[27:29] = np.array([total_force_norm, balance_norm])

        return self.robot_state_buffer

    def _apply_action(self, action: np.ndarray, use_gripper=True):
        """
        Handles the messy details of converting Neural Network output to MuJoCo ctrl.
        """
        # Kinematics Logic
        if self.control_mode == 0:  # Delta Joint
            target_arm_ctrl = self.data.ctrl[:5].copy() + (action[:5] * self.delta_jnt_scale)
            self.data.ctrl[:5] = np.clip(target_arm_ctrl, self.joint_min[:5], self.joint_max[:5])

        elif self.control_mode == 1:  # Delta EE (Mink)
            # Sync Mink with current state
            self.configuration.update(self.data.qpos)

            self.last_action_smoothed[:3] = (self.smoothing_alpha * action[:3].copy()) + \
                                            ((1.0 - self.smoothing_alpha) * self.last_action_smoothed[:3].copy())
            self.last_action_smoothed[3:] = action[3:].copy()
            action = self.last_action_smoothed.copy()

            # Update Mocap Target
            current_ee_pos = self.data.site_xpos[self.gripper_site_id].copy()
            new_mocap_pos = current_ee_pos + (action[:3] * self.ee_pos_scale)
            new_mocap_pos[0] = np.clip(new_mocap_pos[0], -0.6 + self.base_pos_world[0],
                                       0.6 + self.base_pos_world[0])  # X
            new_mocap_pos[1] = np.clip(new_mocap_pos[1], -0.6 + self.base_pos_world[1],
                                       0.6 + self.base_pos_world[1])  # Y
            new_mocap_pos[2] = np.clip(new_mocap_pos[2], -0.02 + self.base_pos_world[2],
                                       0.6 + self.base_pos_world[2])  # Z
            self.data.mocap_pos[self.mocap_id] = new_mocap_pos

            # Update Mocap Orientation
            current_xmat = self.data.site_xmat[self.gripper_site_id].copy()
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_xmat)
            self.data.mocap_quat[self.mocap_id] = self._apply_delta_orientation(current_quat, action[3:6])

            # Solve IK
            T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
            self.end_effector_task.set_target(T_wt)

            for _ in range(self.max_iters):
                vel = mink.solve_ik(
                    self.configuration,
                    self.mink_tasks,
                    self.dt_solver,
                    solver="daqp",
                    limits=self.limits
                )
                self.configuration.integrate_inplace(vel, self.dt_solver)
            # TODO: Record this for FB training
            self.data.ctrl[:5] = self.configuration.q[:5]

        # 3. Gripper Logic
        if use_gripper:
            self.data.ctrl[5] = self.joint_max[5] if action[-1] > 0.0 else self.joint_min[5]
        else:
            self.data.ctrl[5] = self.joint_min[5]  # Permanently closed

    def _apply_delta_orientation(self, current_quat, euler_action):
        scaled_action = euler_action * self.ee_rot_scale
        mujoco.mju_euler2Quat(self.delta_quat, scaled_action, "xyz")
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, current_quat, self.delta_quat)
        return new_quat

    def _check_cube_fallen(self):
        cube_az = self.data.xpos[self.cube_a_id][2]
        cube_bz = self.data.xpos[self.cube_b_id][2]
        has_fallen = False
        if (cube_az < self.base_pos_world[2] - 0.1) or (cube_bz < self.base_pos_world[2] - 0.1):
            has_fallen = True
        return has_fallen

    def _get_info(self):
        """
        Returns dictionary with 'physics' key containing full qpos/qvel state.
        This allows the FB algorithm to reset the sim to this exact moment.
        """
        # Concatenate full qpos and qvel (flattens them into 1D array)
        physics_state = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return {"physics": physics_state}

    def set_physics_state(self, state_vector):
        """
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
                self.renderer.update_scene(self.data, camera="main_cam")
            else:
                self.renderer.update_scene(self.data, camera="main_cam")

            self._render_geoms(self.renderer.scene)
            return self.renderer.render()

    def _render_pads(self, scene):
        scene.ngeom += 2
        # Render left jaw sensor
        left_sensor_pos = self.data.site("left_pad_site").xpos
        left_sensor_mat = self.data.site("left_pad_site").xmat
        mujoco.mjv_initGeom(
            scene.geoms[0],
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
            scene.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.model.site("right_pad_site").size,  # Radius 0.02
            pos=right_sensor_pos,  # Position at our target
            mat=right_sensor_mat,  # Identity matrix for orientation
            rgba=[0, 1, 0, 0.3]  # Green color, opaque
        )

    def _render_geoms(self, scene):
        self._render_pads(scene)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


POSITIONS = {
    "Center": {
        "air_pos": [-0.25, 0.0, 0.63],
        "air_qpos": np.array([0.0553, -1.5220, 0.9610, 0.5584, 1.6101, 0.4974]),
        "table_pos": [-0.25, 0.0, 0.43],
        "table_qpos": np.array([0.0909, -1.0126, 1.5656, 0.3595, 1.5721, 0.6596]),
        "pretable_pos": [-0.25, 0.0, 0.43],
        "pretable_qpos": np.array([0.0861, -1.4377, 1.5888, 0.5182, 1.5713, 0.6376])
    },
    "Left": {
        "air_pos": [-0.25, 0.15, 0.63],
        "air_qpos": np.array([-0.7155, -0.9421, 0.9299, -0.0953, 1.5903, 0.5242]),
        "table_pos": [-0.25, 0.15, 0.43],
        "table_qpos": np.array([-0.6565, -0.1500, 1.1644, -0.2361, 1.5724, 0.9172])
    },
    "Right": {
        "air_pos": [-0.25, -0.15, 0.63],
        "air_qpos": np.array([0.7965, -0.9694, 0.8854, 0.0442, 1.5839, 0.5733]),
        "table_pos": [-0.25, -0.15, 0.43],
        "table_qpos": np.array([0.8350, -0.0737, 1.1920, -0.4043, 1.6566, 1.0319])
    },
    "Far_Left": {
        "air_pos": [-0.1, 0.1, 0.63],
        "air_qpos": np.array([-0.2764, 0.0184, 0.1065, -0.1694, 1.6029, 0.5976]),
        "table_pos": [-0.1, 0.1, 0.43],
        "table_qpos": np.array([-0.2537, 0.8692, 0.2735, -0.9194, 1.5768, 0.6902])
    },
    "Far_Right": {
        "air_pos": [-0.1, -0.1, 0.63],
        "air_qpos": np.array([0.3337, 0.0156, 0.2294, -0.3951, 1.5766, 0.2972]),
        "table_pos": [-0.1, -0.1, 0.43],
        "table_qpos": np.array([0.3746, 0.8199, 0.3387, -0.9249, 1.6066, 0.9172])
    },
}


class SO101LiftEnv(SO101BaseEnv):
    def __init__(self, render_mode=None, reward_type="dense", control_mode="delta_joint_position", fb_train=False, evaluate=False, forced_cube_pos_idx=0):
        super().__init__(render_mode, reward_type, control_mode)
        self.fb_train = fb_train
        self.evaluate = evaluate
        self.forced_cube_pos_idx = forced_cube_pos_idx
        self.success_history = deque(maxlen=50)

        self.z_height_achieved = False
        self.cube_focus_idx = None
        self.target_cube_id = None
        self.object_start_height = None

        self.max_episode_steps = 200
        # 64 bit chosen for 64 bit computers

        if self.fb_train:
            self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float64)
        else:
            # 65 FB Obs + 6 Task Obs - 18 Distractor Cube States
            self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float64)
        actions_dims = 7 if self.control_mode == 1 else 6
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(actions_dims,), dtype=np.float32)
        self.obs_buffer = np.zeros(self.observation_space.shape[0], dtype=np.float64)
        self.task_obs_buffer = np.zeros(6, dtype=np.float64)
        self.fb_obs_buffer = np.zeros(65, dtype=np.float64)

    def step(self, action):
        self._apply_action(action, use_gripper=True)

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.timesteps += 1

        truncated = bool(self.timesteps >= self.max_episode_steps)
        reward = self._compute_reward(action)

        has_fallen = self._check_cube_fallen()
        terminated = has_fallen

        cube_z = self.data.xpos[self.target_cube_id][2]
        if not self.z_height_achieved:
            self.z_height_achieved = bool(cube_z > self.base_pos_world[2] + 0.08)
        elif cube_z < self.base_pos_world[2] + 0.05:
            self.z_height_achieved = False

        if has_fallen: reward = -10.0

        obs, info = self._get_obs_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        self.obs_buffer[0:29] = self._get_robot_state()
        buffer = self.cube_a_buffer if self.cube_focus_idx == 0 else self.cube_b_buffer
        self.obs_buffer[29:47] = self._get_cube_data(buffer, self.target_cube_id)
        self.obs_buffer[47:53] = self._get_task_state()

        self.fb_obs_buffer[0:29] = self.obs_buffer[0:29]
        self.fb_obs_buffer[29:47] = self._get_cube_data(self.cube_a_buffer, self.cube_a_id)
        self.fb_obs_buffer[47:65] = self._get_cube_data(self.cube_b_buffer, self.cube_b_id)

        return self.obs_buffer.copy()

    def _get_obs_info(self):
        info = self._get_info()
        if self.fb_train:
            self._get_obs()
            obs = self.fb_obs_buffer.copy()
        else:
            obs = self._get_obs()
            info["fb_obs"] = self.fb_obs_buffer.copy()
        return obs, info

    def _get_task_state(self):
        dynamic_goal_rel = self.dynamic_goal_pos - self.base_pos_world
        self.task_obs_buffer[0:3] = dynamic_goal_rel
        target_cube_id = self.cube_a_id if self.cube_focus_idx == 0 else self.cube_b_id
        target_cube_pos = self.data.xpos[target_cube_id].copy()
        obj_to_goal = self.dynamic_goal_pos - target_cube_pos
        self.task_obs_buffer[3:6] = obj_to_goal
        return self.task_obs_buffer

    def _compute_reward(self, action):
        ee_pos = self.data.site_xpos[self.gripper_site_id]
        cube_pos = self.data.xpos[self.target_cube_id].copy()

        dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_from_table = cube_pos[2] - self.object_start_height
        dist_to_goal = np.linalg.norm(cube_pos - self.dynamic_goal_pos)

        is_lifted = dist_from_table > 0.05

        dist_z = abs(cube_pos[2] - self.dynamic_goal_pos[2])
        is_goal = dist_z < 0.05

        left_jaw_force = self.data.sensordata[self.left_sensor_id]
        right_jaw_force = self.data.sensordata[self.right_sensor_id]

        total_force = left_jaw_force + right_jaw_force
        has_force = bool(total_force > 20.0)
        is_near_cube = dist_ee_cube < 0.04
        is_above_table = ee_pos[2] > (self.base_pos_world[2] + 0.008)
        is_grasping = has_force and is_near_cube and is_above_table

        if self.reward_type == "sparse":
            # TODO: Fix this
            reward = 0.0
            # Using has_solid_grip instead of distance check for robustness
            if is_lifted: reward += 0.5
            if is_lifted and is_goal: reward += 0.5
            return reward

        elif self.reward_type == "dense":
            # Will be 1 if within 0.03 or so
            reach_reward = np.clip(1.0 - np.tanh(10.0 * (dist_ee_cube - 0.03)), 0.0, 1.0)

            # Will rise sharply after 20N, so 50N is 0.99
            maximum_jaw_force = max(left_jaw_force, right_jaw_force)
            grasp_signal = 0.5 * (1.0 + np.tanh(0.15 * (maximum_jaw_force - 20.0)))
            near_signal = np.clip(1.0 - np.tanh(15.0 * (dist_ee_cube - 0.03)), 0.0, 1.0)

            grasp_reward = 0.0
            hoist_reward = 0.0
            precision_reward = 0.0
            if ee_pos[2] > (self.base_pos_world[2] + 0.008):
                grasp_reward = 3.0 * grasp_signal * near_signal
                hoist_reward = np.tanh(7.0 * dist_from_table)
                precision_reward = (1.0 - np.tanh(10.0 * dist_to_goal)) + (2.0 * (1.0 - np.tanh(50.0 * dist_to_goal)))

            # print(f"reach: {reach_reward}\tgrasp: {grasp_reward}\thoist: {hoist_reward}\tprecision: {precision_reward}")
            total_reward = reach_reward + grasp_reward + hoist_reward + precision_reward
            return total_reward

    def reset(self, seed=None, options=None):
        if self.timesteps > 0:
            current_z = self.data.xpos[self.target_cube_id][2]
            is_success = self.z_height_achieved and (current_z > self.base_pos_world[2] + 0.05)
            self.success_history.append(1 if is_success else 0)

        if len(self.success_history) > 0:
            success_rate = sum(self.success_history) / len(self.success_history)
        else:
            success_rate = 0.0

        if success_rate < 0.2:
            stage_probs = [0.45, 0.45, 0.1, 0.0]  # 45% Hold, 45% Hoist, 10% Pre-Hoist, 0% Random
            self.current_curriculum_stage = 0
        elif success_rate < 0.4:
            stage_probs = [0.2, 0.2, 0.3, 0.3]  # 20% Hold, 20% Hoist, 15% Pre-Hoist, 30% Random
            self.current_curriculum_stage = 1
        elif success_rate < 0.6:
            stage_probs = [0.15, 0.15, 0.2, 0.5]  # 20% Hold, 20% Hoist, 15% Pre-Hoist, 30% Random
            self.current_curriculum_stage = 2
        else:
            stage_probs = [0.1, 0.1, 0.1, 0.7]  # 10% Hold, 10% Hoist, 10% Pre-Hoist, 70% Random
            self.current_curriculum_stage = 3

        if self.evaluate == "hold":
            stage_probs = [1.0, 0.0, 0.0, 0.0]
        elif self.evaluate == "hoist":
            stage_probs = [0.0, 1.0, 0.0, 0.0]
        elif self.evaluate == "prehoist":
            stage_probs = [0.0, 0.0, 1.0, 0.0]
        elif self.evaluate is True:
            stage_probs = [0.0, 0.0, 0.0, 1.0]
        # stage_probs = [0.0, 0.0, 0.0, 1.0]
        stage_roll = np.random.rand()  # Default behavior

        super().reset(seed=seed)

        other_pos_idx = int(np.random.choice(np.delete(np.arange(len(self.cube_start_positions)), self.forced_cube_pos_idx)))
        active_cube_idx = 0 if np.random.rand() > 0.5 else 1
        self.cube_focus_idx = active_cube_idx
        self.target_cube_id = self.cube_a_id if self.cube_focus_idx == 0 else self.cube_b_id
        active_cube_jnt_id = self.cube_a_jnt_id if self.cube_focus_idx == 0 else self.cube_b_jnt_id
        other_cube_jnt_id = self.cube_b_jnt_id if self.cube_focus_idx == 0 else self.cube_a_jnt_id

        loc_name = list(POSITIONS.keys())[self.forced_cube_pos_idx]
        loc_data = POSITIONS[loc_name]

        if stage_roll < stage_probs[0]:
            self._set_cube_pos_quat(other_cube_jnt_id, self.cube_start_positions[other_pos_idx], np.array([1, 0, 0, 0]))
            self._set_cube_pos_quat(active_cube_jnt_id, loc_data["air_pos"], np.array([1, 0, 0, 0]))

            qpos = loc_data["air_qpos"].copy()
            self.data.qpos[:6] = qpos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_min[5]  # auto close
            self.dynamic_goal_pos = np.array(loc_data["air_pos"])
        elif stage_roll < stage_probs[0] + stage_probs[1]:
            self._set_cube_pos_quat(other_cube_jnt_id, self.cube_start_positions[other_pos_idx], np.array([1, 0, 0, 0]))
            self._set_cube_pos_quat(active_cube_jnt_id, loc_data["table_pos"], np.array([1, 0, 0, 0]))

            qpos = loc_data["table_qpos"].copy()
            self.data.qpos[:6] = qpos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_max[5] # open

        elif stage_roll < stage_probs[0] + stage_probs[1] + stage_probs[2]:
            self._set_cube_pos_quat(other_cube_jnt_id, self.cube_start_positions[other_pos_idx], np.array([1, 0, 0, 0]))
            self._set_cube_pos_quat(active_cube_jnt_id, loc_data["pretable_pos"], np.array([1, 0, 0, 0]))

            qpos = loc_data["pretable_qpos"].copy()
            self.data.qpos[:6] = qpos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_max[5]  # open
        else:
            other_quat = np.array([1, 0, 0, 0])
            active_quat = np.array([1, 0, 0, 0])
            self._set_cube_pos_quat(other_cube_jnt_id, self.cube_start_positions[other_pos_idx], other_quat)
            self._set_cube_pos_quat(active_cube_jnt_id, self.cube_start_positions[self.forced_cube_pos_idx], active_quat)

            qpos = loc_data["pretable_qpos"].copy()
            self.data.qpos[:6] = self._arm_start_pos
            self.data.qvel[:6] = 0.0
            self.data.ctrl[:6] = qpos
            self.data.ctrl[5] = self.joint_max[5]  # open

        # Update Kinematics
        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        mink.move_mocap_to_frame(self.model, self.data, "target", "gripperframe", "site")

        # Settle Physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.object_start_height = self.base_pos_world[2] + 0.01
        self.z_height_achieved = False

        # Done after physics has settled
        if stage_roll >= stage_probs[0]:
            self._sample_target()

        self.timesteps = 0

        return self._get_obs_info()

    def _set_cube_pos_quat(self, cube_id, pos, quat):
        adr = self.model.jnt_qposadr[cube_id]
        self.data.qpos[adr: adr + 3] = pos
        self.data.qpos[adr + 3: adr + 7] = quat

    def _sample_target(self):
        cube_pos_world = self.data.xpos[self.target_cube_id].copy()
        lift_height = np.random.uniform(0.18, 0.23)
        self.dynamic_goal_pos = cube_pos_world.copy()
        self.dynamic_goal_pos[2] += lift_height

    def _render_geoms(self, scene):
        self._render_pads(scene)
        scene.ngeom += 1
        color = [0.8, 0, 0, 0.3] if self.cube_focus_idx == 0 else [0, 0, 0.8, 0.3]
        # Render target position
        mujoco.mjv_initGeom(
            scene.geoms[-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=self.dynamic_goal_pos,
            mat=np.eye(3).flatten(),
            rgba=color
        )

class SO101ReachEnv(SO101BaseEnv):
    def __init__(self, render_mode=None, reward_type="dense", control_mode="delta_joint_position", fb_train=False, evaluate=False, forced_cube_pos_idx=0):
        super().__init__(render_mode, reward_type, control_mode)
        self.fb_train = fb_train
        self.evaluate = evaluate

        self.max_episode_steps = 100
        # 64 bit chosen for 64 bit computers
        # 65 FB Obs + 6 Task Obs
        if self.fb_train:
            self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float64)
        else:
            self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(71,), dtype=np.float64)
        actions_dims = 7 if self.control_mode == 1 else 6
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(actions_dims,), dtype=np.float32)
        self.obs_buffer = np.zeros(self.observation_space.shape[0], dtype=np.float64)
        self.task_obs_buffer = np.zeros(6, dtype=np.float64)
        self.fb_obs_buffer = np.zeros(65, dtype=np.float64)

    def step(self, action):
        self._apply_action(action, use_gripper=False)

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.timesteps += 1

        truncated = bool(self.timesteps >= self.max_episode_steps)
        reward = self._compute_reward(action)

        has_fallen = self._check_cube_fallen()
        terminated = has_fallen

        if has_fallen: reward = -10.0

        obs, info = self._get_obs_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        self.obs_buffer[0:29] = self._get_robot_state()
        self.obs_buffer[29:47] = self._get_cube_data(self.cube_a_buffer, self.cube_a_id)
        self.obs_buffer[47:65] = self._get_cube_data(self.cube_b_buffer, self.cube_b_id)
        self.obs_buffer[65:71] = self._get_task_state()

        self.fb_obs_buffer = self.obs_buffer.copy()

        return self.obs_buffer.copy()

    def _get_obs_info(self):
        info = self._get_info()
        if self.fb_train:
            self._get_obs()
            obs = self.fb_obs_buffer.copy()
        else:
            obs = self._get_obs()
            info["fb_obs"] = self.fb_obs_buffer.copy()
        return obs, info

    def _get_task_state(self):
        ee_pos_world = self.data.site_xpos[self.gripper_site_id]
        dynamic_goal_rel = self.dynamic_goal_pos - self.base_pos_world
        self.task_obs_buffer[0:3] = dynamic_goal_rel
        # end effector relative to goal position
        self.task_obs_buffer[3:6] = ee_pos_world - self.dynamic_goal_pos
        return self.task_obs_buffer

    def _compute_reward(self, action):
        ee_pos = self.data.site_xpos[self.gripper_site_id]
        distance = np.linalg.norm(ee_pos - self.dynamic_goal_pos)

        if self.reward_type == "sparse":
            return 1.0 if distance < 0.05 else 0.0
        elif self.reward_type == "dense":
            return 1.0 - np.tanh(3.0 * distance)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:6] = self._arm_start_pos
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = self._arm_start_pos
        self.data.ctrl[5] = self.joint_max[5]

        # Settle Physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        mink.move_mocap_to_frame(self.model, self.data, "target", "gripperframe", "site")

        self._sample_target()

        self.timesteps = 0
        return self._get_obs_info()

    def _sample_target(self):
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

    def _render_geoms(self, scene):
        # Call Base for pads
        self._render_pads(scene)

        # Add Goal Sphere
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],  # 2cm radius
            pos=self.dynamic_goal_pos,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 0.4]  # Transparent Red
        )

if __name__ == "__main__":
    end_effector_testing = False
    if not end_effector_testing:
        env = SO101LiftEnv(render_mode="human", control_mode="delta_end_effector")
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
        env = SO101LiftEnv(render_mode="human", control_mode="delta_end_effector", evaluate=True)
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
        run_sequence(np.array([0, 0, 0, 1.0, 0, 0, 1.0]), 20, "Rolling Wrist (+)")
        run_sequence(np.array([0, 0, 0, -1.0, 0, 0, 1.0]), 20, "Rolling Wrist (-)")

        # Pitch (Up/Down Tilt)
        run_sequence(np.array([0, 0, 0, 0, 1.0, 0, 1.0]), 20, "Pitching Wrist (+)")
        run_sequence(np.array([0, 0, 0, 0, -1.0, 0, 1.0]), 20, "Pitching Wrist (-)")

        # 5. Test Gripper
        # Close
        run_sequence(np.array([0, 0, 0, 0, 0, 0, -1.0]), 30, "Closing Gripper")
        # Open
        run_sequence(np.array([0, 0, 0, 0, 0, 0, 1.0]), 30, "Opening Gripper")

        print("=== TEST COMPLETE ===")
        env.close()
