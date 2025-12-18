import time
import numpy as np
import mujoco
import mujoco.viewer
import imageio
import pygame
from scipy.spatial.transform import Rotation as R
import mink


# --- 1. HELPER FUNCTION FROM MINK EXAMPLE ---
def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False

# --- 2. MAIN CONTROL LOOP ---
def action_joints_mink_ik_super_final():
    # Setup Model
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    # Setup Mink Configuration
    configuration = mink.Configuration(model)

    # Setup Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Mink: Converge IK Mode")
    clock = pygame.time.Clock()

    # --- DEFINE TASKS (Faithful to Example) ---
    # 1. End Effector Task (The target we control)
    ee_task = mink.FrameTask(
        frame_name="gripperframe",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # 2. Posture Task (Keeps arm natural)
    posture_task = mink.PostureTask(model, cost=1e-2)
    tasks = [ee_task, posture_task]

    # # --- INITIALIZATION ---
    # # 1. Sync physics
    # mujoco.mj_forward(model, data)
    #
    # # 2. Sync Configuration to current robot state
    # configuration.update(data.qpos)
    # posture_task.set_target_from_configuration(configuration)
    #
    # # 3. Create a Virtual Target (Simulating the 'mocap' from the example)
    # # We initialize it exactly where the gripper site is right now
    # site_id = model.site("gripperframe").id
    # start_pos = data.site_xpos[site_id].copy()
    # start_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    #
    # # Create the SE3 object for the target
    # target_rotation = mink.SO3.from_matrix(start_mat)
    # target_translation = start_pos

    # Parameters
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20  # How hard the solver tries per frame
    dt = model.opt.timestep

    move_speed = 0.005
    rot_speed = 0.05

    print("Control Ready. Using converge_ik strategy.")

    with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset simulation data to the 'home' keyframe
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Move the mocap target to the end-effector's current pose
        mink.move_mocap_to_frame(model, data, "target", "gripperframe", "site")
        initial_target_position = data.mocap_pos[0].copy()
        amp = 0.10
        freq = 0.2
        while viewer.is_running():
            clock.tick(60)

            # --- A. INPUT (Updates the Virtual Target) ---
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]: break

            # Position Update
            d_pos = np.zeros(3)
            if keys[pygame.K_w]: d_pos[0] += move_speed
            if keys[pygame.K_s]: d_pos[0] -= move_speed
            if keys[pygame.K_a]: d_pos[1] += move_speed  # Y-Axis Left
            if keys[pygame.K_d]: d_pos[1] -= move_speed  # Y-Axis Right
            if keys[pygame.K_SPACE]: d_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: d_pos[2] -= move_speed

            target_translation += d_pos

            # Rotation Update
            d_rot = mink.SO3.identity()
            if keys[pygame.K_r]: d_rot = d_rot @ mink.SO3.from_x_radians(rot_speed)
            if keys[pygame.K_f]: d_rot = d_rot @ mink.SO3.from_x_radians(-rot_speed)
            if keys[pygame.K_t]: d_rot = d_rot @ mink.SO3.from_y_radians(rot_speed)
            if keys[pygame.K_g]: d_rot = d_rot @ mink.SO3.from_y_radians(-rot_speed)
            if keys[pygame.K_y]: d_rot = d_rot @ mink.SO3.from_z_radians(rot_speed)
            if keys[pygame.K_h]: d_rot = d_rot @ mink.SO3.from_z_radians(-rot_speed)

            target_rotation = d_rot @ target_rotation

            # --- B. UPDATE TASK TARGET ---
            # Combine pos/rot into an SE3 target
            target_pose = mink.SE3.from_rotation_and_translation(target_rotation, target_translation)
            ee_task.set_target(target_pose)

            # Run the iterative solver
            converge_ik(
                configuration,
                tasks,
                dt,
                solver,
                pos_threshold,
                ori_threshold,
                max_iters,
            )

            # --- D. APPLY CONTROLS ---
            # The 'configuration.q' now holds the solution found by IK
            data.ctrl[:model.nu] = configuration.q[:model.nu]

            # Step Physics
            mujoco.mj_step(model, data)

            # --- E. VISUALIZE TARGET ---
            # Draw a box where we WANT the robot to be
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.01, 0.01],
                pos=target_translation,
                mat=target_rotation.as_matrix().flatten(),
                rgba=[0, 1, 0, 0.5]
            )
            viewer.sync()


def action_joints_open_space():
    # --- 1. SETUP PYGAME FOR INPUT ---
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("W/S: X | A/D: Y | Space/Ctrl: Z")

    # 2. Load the Model and Data
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    # Perform a forward kinematic step to get initial positions
    mujoco.mj_forward(model, data)

    # --- 3. SETUP END EFFECTOR CONTROL ---
    # Find the ID of the body you want to control (the gripper)
    # Change 'gripper' to 'link6' or 'hand' if your XML uses different names
    ee_name = 'gripper'
    ee_id = model.body(ee_name).id

    # Initialize the target position to the current robot position so it doesn't snap away
    target_ee_pos = data.xpos[ee_id].copy()
    target_positions = np.zeros(6)
    move_speed = 0.0005  # Meters per loop

    # 4. Launch the Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # A. HANDLE INPUT (Cartesian Control)
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            # X Axis
            if keys[pygame.K_w]: target_ee_pos[0] += move_speed
            if keys[pygame.K_s]: target_ee_pos[0] -= move_speed

            # Y Axis
            if keys[pygame.K_a]: target_ee_pos[1] += move_speed
            if keys[pygame.K_d]: target_ee_pos[1] -= move_speed

            # Z Axis
            if keys[pygame.K_SPACE]: target_ee_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: target_ee_pos[2] -= move_speed

            # B. INVERSE KINEMATICS (Calculate joint angles to reach target)
            # 1. Calculate the error
            current_ee_pos = data.xpos[ee_id]
            error = target_ee_pos - current_ee_pos

            # 2. Calculate the Jacobian
            # 3xN matrix. N (nv) is 18 in your case.
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, ee_id)

            # 3. Solve for joint velocities (dq)
            limit = 0.1
            error_clamped = np.clip(error, -limit, limit)
            jacp_pinv = np.linalg.pinv(jacp)
            dq = jacp_pinv @ error_clamped  # dq has shape (18,)

            # 4. Apply to controls
            # FIX: We only care about the robot actuators (model.nu, likely 6)
            # We assume the robot joints are the first ones in the list.

            n_actuators = model.nu  # Number of motors (6)

            # Get the current angles of JUST the robot arm
            current_robot_qpos = data.qpos[:n_actuators]
            if keys[pygame.K_e]:
                current_robot_qpos[4] += 0.005
            if keys[pygame.K_q]:
                current_robot_qpos[4] -= 0.005

            # Joint 6 Control
            if keys[pygame.K_1]:
                current_robot_qpos[5] += 0.005
            if keys[pygame.K_2]:
                current_robot_qpos[5] -= 0.005

            # Get the calculated velocity for JUST the robot arm
            robot_dq = dq[:n_actuators]

            # Add them together
            q_target = current_robot_qpos + robot_dq

            # Update control
            data.ctrl[:n_actuators] = q_target

            # C. PHYSICS STEP
            mujoco.mj_step(model, data)

            # D. VISUALIZATION (Draw the Dot)
            # We use the user_scn (User Scene) to add geometric primitives
            viewer.user_scn.ngeom = 3
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],  # Radius 0.02
                pos=target_ee_pos,  # Position at our target
                mat=np.eye(3).flatten(),  # Identity matrix for orientation
                rgba=[1, 0, 0, 1]  # Red color, opaque
            )
            left_sensor_pos = data.site("left_pad_site").xpos
            left_sensor_mat = data.site("left_pad_site").xmat
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=model.site("left_pad_site").size,  # Radius 0.02
                pos=left_sensor_pos,  # Position at our target
                mat=left_sensor_mat,  # Identity matrix for orientation
                rgba=[0, 1, 0, 0.3]  # Green color, opaque
            )
            right_sensor_pos = data.site("right_pad_site").xpos
            right_sensor_mat = data.site("right_pad_site").xmat
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=model.site("right_pad_site").size,  # Radius 0.02
                pos=right_sensor_pos,  # Position at our target
                mat=right_sensor_mat,  # Identity matrix for orientation
                rgba=[0, 1, 0, 0.3]  # Green color, opaque
            )

            # E. RENDER & SYNC
            viewer.sync()

            # Frame rate limit
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        pygame.quit()

def action_joints_mink_ik_final():
    # --- 1. SETUP ---
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    # Initialize Mink Configuration
    configuration = mink.Configuration(model)

    # Sync configuration to the robot's starting position (CRITICAL to prevent drooping)
    # This tells the solver: "The robot is currently HERE, not at 0,0,0"
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # --- 2. DEFINE TASKS ---
    # Task 1: Hand Tracking (using site 'gripperframe')
    ee_task = mink.FrameTask(
        frame_name="gripperframe",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0
    )

    # Task 2: Posture (CRITICAL for stability)
    # This acts like a "spring" pulling joints towards a comfortable pose
    # so the arm doesn't flop around when there are multiple ways to reach a point.
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)

    tasks = [ee_task, posture_task]

    # --- 3. PYGAME SETUP ---
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Mink Control: " + ee_task.frame_name)
    clock = pygame.time.Clock()

    # --- FIX: Manually get SE3 from Site ---
    site_id = model.site("gripperframe").id

    # 1. Get raw data from Mujoco
    start_pos = data.site_xpos[site_id].copy()
    start_mat = data.site_xmat[site_id].reshape(3, 3).copy()

    # 2. Create Mink objects manually
    start_rot = mink.SO3.from_matrix(start_mat)
    target_pose = mink.SE3.from_rotation_and_translation(start_rot, start_pos)

    # Initialize task target
    ee_task.set_target(target_pose)

    move_speed = 0.005
    rot_speed = 0.05

    solver = "quadprog"
    dt = model.opt.timestep

    print(f"Controlling Site: {ee_task.frame_name}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            clock.tick(60)

            # --- A. INPUT HANDLING ---
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]: break

            # 1. Position Delta
            d_pos = np.zeros(3)
            if keys[pygame.K_w]: d_pos[0] += move_speed
            if keys[pygame.K_s]: d_pos[0] -= move_speed
            if keys[pygame.K_a]: d_pos[1] += move_speed
            if keys[pygame.K_d]: d_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: d_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: d_pos[2] -= move_speed

            # 2. Rotation Delta
            d_rot = mink.SO3.identity()
            if keys[pygame.K_r]: d_rot = d_rot @ mink.SO3.from_x_radians(rot_speed)
            if keys[pygame.K_f]: d_rot = d_rot @ mink.SO3.from_x_radians(-rot_speed)
            if keys[pygame.K_t]: d_rot = d_rot @ mink.SO3.from_y_radians(rot_speed)
            if keys[pygame.K_g]: d_rot = d_rot @ mink.SO3.from_y_radians(-rot_speed)
            if keys[pygame.K_y]: d_rot = d_rot @ mink.SO3.from_z_radians(rot_speed)
            if keys[pygame.K_h]: d_rot = d_rot @ mink.SO3.from_z_radians(-rot_speed)

            # --- B. UPDATE TARGET STATE ---
            new_pos = target_pose.translation() + d_pos

            # Apply Rotation: new = delta * old (Global rotation)
            new_rot = d_rot @ target_pose.rotation()

            target_pose = mink.SE3.from_rotation_and_translation(new_rot, new_pos)
            ee_task.set_target(target_pose)

            # --- C. SOLVE IK ---
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                solver,
                damping=1e-3
            )

            # Update Virtual Robot
            configuration.integrate_inplace(vel, dt)

            # --- D. APPLY TO REAL ROBOT ---
            data.ctrl[:model.nu] = configuration.q[:model.nu]

            # --- E. STEP PHYSICS ---
            mujoco.mj_step(model, data)

            # Visualize Target
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.01, 0.01],
                pos=target_pose.translation(),
                mat=target_pose.rotation().as_matrix().flatten(),
                rgba=[0, 1, 0, 0.5]
            )
            viewer.sync()

def action_joints_6d_ik():
    # --- 1. SETUP PYGAME ---
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("POS: W/S A/D Space/Ctrl | ROT: R/F T/G Y/H")

    # 2. Load Model
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 3. SETUP CONTROL
    ee_name = 'gripper'  # Ensure this matches your XML
    ee_id = model.body(ee_name).id

    # Initialize Targets (Position AND Orientation)
    target_pos = data.xpos[ee_id].copy()
    target_mat = data.xmat[ee_id].reshape(3, 3).copy()  # 3x3 Rotation Matrix

    move_speed = 0.002
    rot_speed = 0.005  # Radians per loop

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. HANDLE INPUT ---
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            # Translation (XYZ) - Global Frame
            if keys[pygame.K_w]: target_pos[0] += move_speed
            if keys[pygame.K_s]: target_pos[0] -= move_speed
            if keys[pygame.K_a]: target_pos[1] += move_speed
            if keys[pygame.K_d]: target_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: target_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: target_pos[2] -= move_speed

            # Rotation (Roll/Pitch/Yaw) - Global Frame
            # We apply rotation by multiplying the target matrix by a small rotation matrix
            rot_delta = np.eye(3)

            # Rotate around X (Roll) - Keys R / F
            if keys[pygame.K_r]:
                angle = rot_speed
                rot_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
                target_mat = rot_x @ target_mat  # Global rotation
            if keys[pygame.K_f]:
                angle = -rot_speed
                rot_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
                target_mat = rot_x @ target_mat

            # Rotate around Y (Pitch) - Keys T / G
            if keys[pygame.K_t]:
                angle = rot_speed
                rot_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
                target_mat = rot_y @ target_mat
            if keys[pygame.K_g]:
                angle = -rot_speed
                rot_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
                target_mat = rot_y @ target_mat

            # Rotate around Z (Yaw) - Keys Y / H
            if keys[pygame.K_y]:
                angle = rot_speed
                rot_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                target_mat = rot_z @ target_mat
            if keys[pygame.K_h]:
                angle = -rot_speed
                rot_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                target_mat = rot_z @ target_mat

            # --- B. 6-DOF INVERSE KINEMATICS ---

            # 1. Get Current State
            curr_pos = data.xpos[ee_id]
            curr_mat = data.xmat[ee_id].reshape(3, 3)

            # 2. Position Error
            err_pos = target_pos - curr_pos

            # 3. Orientation Error
            # We need the rotation vector that takes curr_mat to target_mat
            # Math: err_rot = curr_mat * unskew(curr_mat.T @ target_mat - target_mat.T @ curr_mat) / 2
            # Simplified approximation using cross product of columns:
            err_rot = 0.5 * (
                    np.cross(curr_mat[:, 0], target_mat[:, 0]) +
                    np.cross(curr_mat[:, 1], target_mat[:, 1]) +
                    np.cross(curr_mat[:, 2], target_mat[:, 2])
            )

            # 4. Stack Errors (6D Vector)
            error_6d = np.concatenate([err_pos, err_rot])

            # 5. Get Jacobians (Now we need BOTH)
            jacp = np.zeros((3, model.nv))  # Translation
            jacr = np.zeros((3, model.nv))  # Rotation
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

            # Stack Jacobians (6 x nv)
            jac_full = np.vstack([jacp, jacr])

            # 6. Solve for dq
            # Damping factor helps stability near singularities
            diag = 1e-4 * np.eye(6)

            # Solve J * dq = error  ->  dq = pinv(J) * error
            # Using simple pinv with clamping
            dq = np.linalg.pinv(jac_full) @ np.clip(error_6d, -0.5, 0.5)

            # --- C. APPLY CONTROLS ---
            # Filter just the arm joints
            n_actuators = model.nu
            q_target = data.qpos[:n_actuators] + dq[:n_actuators]
            data.ctrl[:n_actuators] = q_target

            # --- D. PHYSICS & RENDER ---
            mujoco.mj_step(model, data)

            # Visualize the target orientation
            # We draw a box instead of a sphere so you can see the rotation
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.01, 0.005],  # Box shape helps visualize rotation
                pos=target_pos,
                mat=target_mat.flatten(),  # Use our target rotation!
                rgba=[0, 1, 0, 0.5]  # Green transparent
            )
            viewer.sync()

def action_joints_6d_control():
    # --- 1. SETUP PYGAME ---
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("POS: W/S A/D Space/Ctrl | ROT: U/I O/P J/K")

    # --- 2. LOAD MODEL ---
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- 3. SETUP CONTROL ---
    ee_name = 'gripper'  # Ensure this matches your XML body name
    ee_id = model.body(ee_name).id

    # A. Initialize Target Position (XYZ)
    target_ee_pos = data.xpos[ee_id].copy()

    # B. Initialize Target Orientation (Quaternion [w, x, y, z])
    target_ee_quat = data.xquat[ee_id].copy()

    # Constants
    move_speed = 0.002
    rot_speed = 0.05

    # Define how many joints belong to the ARM (exclude gripper)
    # SO-101 has 5 arm joints. Gripper is the 6th.
    n_arm_joints = 5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. HANDLE INPUT ---
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            # 1. Position Inputs (XYZ)
            if keys[pygame.K_w]: target_ee_pos[0] += move_speed
            if keys[pygame.K_s]: target_ee_pos[0] -= move_speed
            if keys[pygame.K_a]: target_ee_pos[1] += move_speed
            if keys[pygame.K_d]: target_ee_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: target_ee_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: target_ee_pos[2] -= move_speed

            # 2. Orientation Inputs (Incremental Rotation)
            # We create a small "delta quaternion" based on keys and apply it to target

            # Reset delta rotation for this frame
            d_quat = np.array([1.0, 0.0, 0.0, 0.0])

            # Roll (Local X) - U / I
            if keys[pygame.K_u]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([1., 0., 0.]), rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)
            if keys[pygame.K_i]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([1., 0., 0.]), -rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)

            # Pitch (Local Y) - O / P
            if keys[pygame.K_o]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([0., 1., 0.]), rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)
            if keys[pygame.K_p]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([0., 1., 0.]), -rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)

            # Yaw (Local Z) - J / K
            if keys[pygame.K_j]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([0., 0., 1.]), rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)
            if keys[pygame.K_k]:
                mujoco.mju_axisAngle2Quat(d_quat, np.array([0., 0., 1.]), -rot_speed)
                mujoco.mju_mulQuat(target_ee_quat, target_ee_quat, d_quat)

            # Normalize quaternion to prevent drift
            mujoco.mju_normalize4(target_ee_quat)

            # --- B. 6D INVERSE KINEMATICS ---

            # 1. Calculate Positional Error
            current_ee_pos = data.xpos[ee_id]
            err_pos = target_ee_pos - current_ee_pos

            # 2. Calculate Rotational Error
            # We use mju_subQuat to get the 3D angular velocity needed to go from current -> target
            current_ee_quat = data.xquat[ee_id]
            err_rot = np.zeros(3)
            # Computes 3D rotational error: target - current
            mujoco.mju_subQuat(err_rot, target_ee_quat, current_ee_quat)

            # 3. Combine Error (6D vector)
            err_full = np.hstack([err_pos, err_rot])

            # 4. Calculate Full Jacobian (6 x nv)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

            # Stack them: Top 3 rows are Pos, Bottom 3 are Rot
            jac_full = np.vstack([jacp, jacr])

            # 5. Solve (Damped Least Squares for stability)
            # We only want to solve for the ARM joints (columns 0 to 4)
            # The gripper (column 5) should not move to help the arm.

            jac_arm = jac_full[:, :n_arm_joints]  # Slice 6x5 matrix

            # Calculate dq for arm
            dq_arm = np.linalg.pinv(jac_arm) @ err_full

            # --- C. APPLY CONTROLS ---

            # 1. Arm Control
            current_arm_qpos = data.qpos[:n_arm_joints]

            # Scale dq to prevent explosions (integration step)
            integration_dt = 0.5
            q_target = current_arm_qpos + (dq_arm * integration_dt)

            data.ctrl[:n_arm_joints] = q_target

            # 2. Gripper Control (Manual Overrides)
            # Gripper is typically the last actuator
            if keys[pygame.K_1]:  # Open
                data.ctrl[n_arm_joints] = 1.0
            if keys[pygame.K_2]:  # Close
                data.ctrl[n_arm_joints] = -1.0

            # --- D. STEP & RENDER ---
            mujoco.mj_step(model, data)

            # Visualize Target
            viewer.user_scn.ngeom = 1
            mat_target = np.zeros(9)
            mujoco.mju_quat2Mat(mat_target, target_ee_quat)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.01, 0.005],  # Box shape helps verify rotation
                pos=target_ee_pos,
                mat=mat_target,
                rgba=[0, 1, 0, 0.5]  # Green transparent
            )
            viewer.sync()

            # Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    pygame.quit()

def action_joints_manual():
    # --- 1. SETUP PYGAME FOR INPUT ---
    # Pygame needs a tiny window to capture keyboard focus
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Click here to control")

    # 2. Load the Model and Data
    # If loading from file: model = mujoco.MjModel.from_xml_path("so100.xml")
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    # --- 3. CONTROL STATE ---
    # Initialize target angles at 0
    # We use this variable to store "Where we want the robot to be"
    target_positions = np.zeros(6)
    move_speed = 0.01 # How fast joints move per key press

    # 3. Launch the Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        # Simulation Loop
        while viewer.is_running():
            step_start = time.time()

            # A. HANDLE INPUT (Pygame)
            pygame.event.pump()  # Process event queue
            keys = pygame.key.get_pressed()

            # Joint 1 Control
            if keys[pygame.K_q]:
                target_positions[0] += move_speed
            if keys[pygame.K_a]:
                target_positions[0] -= move_speed

            # Joint 2 Control
            if keys[pygame.K_w]:
                target_positions[1] += move_speed
            if keys[pygame.K_s]:
                target_positions[1] -= move_speed

            # Joint 3 Control
            if keys[pygame.K_e]:
                target_positions[2] += move_speed
            if keys[pygame.K_d]:
                target_positions[2] -= move_speed

            # Joint 4 Control
            if keys[pygame.K_r]:
                target_positions[3] += move_speed
            if keys[pygame.K_f]:
                target_positions[3] -= move_speed

            # Joint 5 Control
            if keys[pygame.K_t]:
                target_positions[4] += move_speed
            if keys[pygame.K_g]:
                target_positions[4] -= move_speed

            # Joint 6 Control
            if keys[pygame.K_y]:
                target_positions[5] += move_speed
            if keys[pygame.K_h]:
                target_positions[5] -= move_speed

            # B. APPLY CONTROL TO MUJOCO
            # Ideally we clip these to joint limits, but for testing raw is fine
            data.ctrl[:] = target_positions[:]

            # C. PHYSICS STEP
            mujoco.mj_step(model, data)

            # D. OBSERVATION (Print every 0.5s)
            # Showing you how to read the data
            if int(time.time() * 10) % 5 == 0:
                # Current actual angles
                current_qpos = data.qpos[:]
                # Current XYZ of the last link (End Effector)
                # Note: You might need to change 'link6' to whatever your EE body name is
                ee_id = model.body('gripper').id
                ee_pos = data.xpos[ee_id]

                # Print (carriage return \r to keep line clean)
                print(f"Targets: {target_positions[:].round(2)} | Actual: {current_qpos[:].round(2)} | EE Pos: {ee_pos.round(2)}")

            # E. RENDER
            viewer.sync()

            # Frame rate limit
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        pygame.quit()

if __name__ == "__main__":
    action_joints_open_space()
