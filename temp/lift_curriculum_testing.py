import time
import numpy as np
import mujoco
import mujoco.viewer
import pygame

"""
Lift:
Center
Cube Position (Air): [-0.25, 0.0, 0.63]
perfect_grasp_qpos = np.array([0.0553, -1.5220, 0.9610, 0.5584, 1.6101, 0.4974])

Left
Cube Position (Air): [-0.25, 0.15, 0.63]
perfect_grasp_qpos = np.array([-0.7155, -0.9421, 0.9299, -0.0953, 1.5903, 0.5242])

Right
Cube Position (Air): [-0.25, -0.15, 0.63]
perfect_grasp_qpos = np.array([0.7965, -0.9694, 0.8854, 0.0442, 1.5839, 0.5733])

Far Left
Cube Position (Air): [-0.1, 0.1, 0.63]
perfect_grasp_qpos = np.array([-0.2764, 0.0184, 0.1065, -0.1694, 1.6029, 0.5976])

Far Right
Cube Position (Air): [-0.1, -0.1, 0.63]
perfect_grasp_qpos = np.array([0.3337, 0.0156, 0.2294, -0.3951, 1.5766, 0.2972])


Hoist:
Center
Cube Position (Table): [-0.25, 0.0, 0.43]
perfect_grasp_qpos = np.array([0.0909, -1.0126, 1.5656, 0.3595, 1.5721, 0.6596])

Left
Cube Position (Air): [-0.25, 0.15, 0.43]
perfect_grasp_qpos = np.array([-0.6565, -0.1500, 1.1644, -0.2361, 1.5724, 0.9172])

Right
Cube Position (Air): [-0.25, -0.15, 0.43]
perfect_grasp_qpos = np.array([0.8350, -0.0737, 1.1920, -0.4043, 1.6566, 1.0319])

Far Left
Cube Position (Air): [-0.1, 0.1, 0.43]
perfect_grasp_qpos = np.array([-0.2537, 0.8692, 0.2735, -0.9194, 1.5768, 0.6902])

Far Right
Cube Position (Table): [-0.1, -0.1, 0.43]
perfect_grasp_qpos = np.array([0.3746, 0.8199, 0.3387, -0.9249, 1.6066, 0.9172])


Pre-Grasp
Center
Cube Position (Table): [-0.25, 0.0, 0.43]
perfect_grasp_qpos = np.array([0.0861, -1.4377, 1.5888, 0.5182, 1.5713, 0.6376])

Left
Cube Position (Air): [-0.25, 0.15, 0.43]
perfect_grasp_qpos = np.array([-0.7156, -0.9924, 1.5539, -0.1082, 1.5717, 0.9175])

Right
Cube Position (Air): [-0.25, -0.15, 0.43]
perfect_grasp_qpos = np.array([0.8224, -0.7102, 1.3881, -0.0829, 1.5715, 0.5755])

Far Left
Cube Position (Air): [-0.1, 0.1, 0.43]
perfect_grasp_qpos = np.array([-0.2833, 0.1671, 0.7488, -0.5424, 1.5720, 0.6259])

Far Right
Cube Position (Table): [-0.1, -0.1, 0.43]
perfect_grasp_qpos = np.array([0.3778, 0.2592, 0.5301, -0.3884, 1.5718, 0.7248])
"""


def check_grasp_heuristic(model, data, ee_pos, cube_pos, table_height):
    """
    Determines grasp based on Force + Position logic.
    Robust to asymmetry and table-pushing hacks.
    """
    # 1. READ SENSORS
    try:
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_sensor")
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_sensor")
        l_force = data.sensordata[left_id]
        r_force = data.sensordata[right_id]
    except:
        return False, 0, 0, "XML_ERROR"

    # 2. SUM FORCES (Solves the "Left=0, Right=50" bug)
    # If one finger does all the work, the sum still detects it.
    total_force = l_force + r_force

    # Threshold based on your data (42-52 typical).
    # We use 20.0 to be safe (half of a single finger).
    FORCE_THRESHOLD = 20.0
    has_force = total_force > FORCE_THRESHOLD

    # 3. PROXIMITY CHECK (Solves "Pushing table far away")
    # The gripper center must be very close to the cube center.
    dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)
    is_near_cube = dist_ee_cube < 0.04  # 4cm radius

    # 4. TABLE CLEARANCE CHECK (Solves "Pushing table near cube")
    # The gripper tips must not be grinding against the floor.
    # ee_pos is usually the center of the gripper base/fingers.
    # Check if Z is safely above the table.
    is_above_table = ee_pos[2] > (table_height - 0.022)  # 1cm tolerance

    # 5. FINAL LOGIC
    # We are grasping IF: We have force AND we are near cube AND we aren't humping the table
    is_grasping = has_force and is_near_cube and is_above_table

    # Diagnostics for you
    status = "GRASP" if is_grasping else "NONE"
    if has_force and not is_near_cube: status = "TABLE_PUSH (Far)"
    if has_force and not is_above_table: status = "TABLE_PUSH (Low)"

    return is_grasping, l_force, r_force, status


def calculate_reward_v2(dist, is_grasping, cube_z, table_height, total_force):
    """
    Reward function tuned for the 40-60N force range.
    """
    # 1. Reach (Standard)
    r_reach = 1.0 - np.tanh(10.0 * dist)

    # 2. Grasp (Binary + Continuous Bonus)
    # We add a small continuous bonus for force so the gradient exists
    # but cap it so 60N doesn't give infinite reward.
    r_grasp = 0.0
    if is_grasping:
        r_grasp = 1.0 + np.tanh(total_force / 100.0)  # Max ~1.5

    # 3. Lift (The Big Payout)
    r_lift = 0.0
    # Strict check: Must be grasping to get lift reward
    if is_grasping and cube_z > (table_height + 0.03):
        height_gain = (cube_z - table_height)
        r_lift = height_gain * 20.0  # Huge multiplier to encourage lifting

    return r_reach, r_grasp, r_lift


def laboratory_mode_with_sensors():
    # --- 1. SETUP ---
    pygame.init()
    screen = pygame.display.set_mode((600, 200))  # Wider window for text
    pygame.display.set_caption("TESTING SENSORS | Press F to Freeze Cube")

    # Load Model
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    # --- CRITICAL: MATCH ENV START POSE ---
    # Copying the exact logic from your Environment
    joint_min = model.jnt_range[:6, 0]
    joint_max = model.jnt_range[:6, 1]

    # Using the "Center" start pose or "Home" pose
    start_pos = np.array([
        0.0, joint_min[1], joint_max[2],
        0.5, np.pi / 2, joint_min[-1]
    ], dtype=np.float32)

    data.qpos[:6] = start_pos
    data.ctrl[:6] = start_pos
    mujoco.mj_forward(model, data)

    # IDs
    ee_id = model.body("gripper").id
    # Try to find cube joint

    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_joint")
    cube_adr = model.jnt_qposadr[cube_joint_id]

    # Get table height for reward calc (assuming base is at 0)
    base_pos = data.body("base").xpos
    table_height = base_pos[2]

    # State Variables
    target_ee_pos = data.xpos[ee_id].copy()
    cube_frozen = True
    # The "Center" Table position from your dict
    cube_target_pos = np.array([-0.25, 0.00, 0.43])

    print("\n=== SENSOR DEBUG MODE ===")
    print(" [WASD] : Move XY  | [Space/Ctrl] : Move Z")
    print(" [ [ ] ] : Open/Close Gripper")
    print(" [ F ]  : Freeze/Unfreeze Cube")
    print("========================\n")

    with mujoco.viewer.launch_passive(
            model, data, show_right_ui=False, show_left_ui=False,
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            # --- CUBE LOGIC ---
            if keys[pygame.K_r]: cube_frozen = True
            if keys[pygame.K_f]:
                cube_frozen = not cube_frozen
                time.sleep(0.2)

            if cube_frozen and cube_adr != -1:
                data.qpos[cube_adr:cube_adr + 3] = cube_target_pos
                data.qpos[cube_adr + 3:cube_adr + 7] = [1, 0, 0, 0]
                data.qvel[cube_adr:cube_adr + 6] = 0

            # --- CONTROL LOGIC ---
            move_speed = 0.0005  # Slightly faster for testing
            if keys[pygame.K_w]: target_ee_pos[0] += move_speed
            if keys[pygame.K_s]: target_ee_pos[0] -= move_speed
            if keys[pygame.K_a]: target_ee_pos[1] += move_speed
            if keys[pygame.K_d]: target_ee_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: target_ee_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: target_ee_pos[2] -= move_speed

            current_ee_pos = data.xpos[ee_id]
            error = target_ee_pos - current_ee_pos
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, ee_id)
            lmbda = 0.01
            jac_pinv = jacp.T @ np.linalg.inv(jacp @ jacp.T + lmbda * np.eye(3))
            dq = jac_pinv @ np.clip(error, -0.1, 0.1)

            n_actuators = 6
            current_qpos = data.qpos[:n_actuators].copy()
            new_qpos = current_qpos + dq[:n_actuators]

            manual_speed = 0.05
            if keys[pygame.K_UP]: new_qpos[3] -= manual_speed
            if keys[pygame.K_DOWN]: new_qpos[3] += manual_speed
            if keys[pygame.K_LEFT]: new_qpos[4] += manual_speed
            if keys[pygame.K_RIGHT]: new_qpos[4] -= manual_speed

            # Gripper
            if keys[pygame.K_RIGHTBRACKET]: new_qpos[5] -= manual_speed
            if keys[pygame.K_LEFTBRACKET]: new_qpos[5] += manual_speed
            new_qpos[5] = np.clip(new_qpos[5], model.jnt_range[5, 0], model.jnt_range[5, 1])

            data.ctrl[:n_actuators] = new_qpos

            # --- PHYSICS STEP ---
            mujoco.mj_step(model, data)

            ee_pos = data.site("gripperframe").xpos
            # Get cube position safely
            if cube_adr != -1:
                cube_id = model.body("cube_a").id
                cube_pos = data.xpos[cube_id]
                cube_z = cube_pos[2]
            else:
                cube_pos = np.zeros(3)
                cube_z = 0.0

            # --- CHECK GRASP ---
            # 0.43 is your table height based on previous context
            is_grasping, l_f, r_f, status = check_grasp_heuristic(
                model, data, ee_pos, cube_pos, table_height=0.43
            )

            # --- CALC REWARD ---
            dist = np.linalg.norm(ee_pos - cube_pos)
            r_reach, r_grasp, r_lift = calculate_reward_v2(
                dist, is_grasping, cube_z, 0.43, l_f + r_f
            )
            total = r_reach + r_grasp + r_lift

            # --- VISUALIZATION ---
            # Green = Valid Grasp
            # Yellow = Force detected but invalid (Table push)
            # Red = No Force
            if is_grasping:
                color = [0, 1, 0, 0.8]  # Green
            elif (l_f + r_f) > 20.0:
                color = [1, 1, 0, 0.5]  # Yellow (Warning)
            else:
                color = [1, 0, 0, 0.3]  # Red

            pygame.display.set_caption(f"[{status}] L:{l_f:.0f} R:{r_f:.0f} | Rew: {total:.2f}")

            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0, 0],
                pos=target_ee_pos,
                mat=np.eye(3).flatten(),
                rgba=color
            )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    pygame.quit()

def laboratory_mode():
    # --- 1. SETUP ---
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("LABORATORY MODE | Press ENTER to Print State")

    # Load Model
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)

    joint_min = model.jnt_range[:6, 0]
    joint_max = model.jnt_range[:6, 1]

    start_pos = np.array([
        0.0,
        joint_min[1],
        joint_max[2],
        0.5,
        np.pi / 2,
        joint_min[-1],
    ], dtype=np.float32)

    # Initialize the robot here BEFORE you start moving to the cube
    data.qpos[:6] = start_pos
    data.qvel[:6] = 0.0
    data.ctrl[:6] = start_pos

    mujoco.mj_forward(model, data)

    # IDs
    ee_id = model.body("gripper").id
    # Assuming the cube joint is named "cube_a_joint" based on your env code
    try:
        cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_joint")
        cube_adr = model.jnt_qposadr[cube_joint_id]
    except:
        print("Warning: Could not find 'cube_a_joint'. Check XML names.")
        cube_adr = -1

    # State Variables
    target_ee_pos = data.xpos[ee_id].copy()

    # Cube Control State
    cube_frozen = True  # Starts floating
    cube_target_pos = np.array([-0.25, 0.0, 0.43])  # The "Air" position you want to test

    # Manual Joint Offsets (For Wrist/Gripper)
    # [ShoulderPan, ShoulderLift, Elbow, WristFlex, WristRoll, Gripper]
    # We will read current state, but add manual offsets for the last 3 joints

    print("\n=== LABORATORY CONTROLS ===")
    print(" [W/S/A/D/Spc/Ctrl] : Move XYZ (IK)")
    print(" [Up/Down Arrows]   : Wrist Pitch")
    print(" [Left/Right Arrows]: Wrist Roll")
    print(" [ ] / [ ]          : Open / Close Gripper")
    print(" [ F ]              : Freeze/Unfreeze Cube (Test Grasp)")
    print(" [ R ]              : Reset Cube Position")
    print(" [ ENTER ]          : PRINT COORDINATES FOR CODE")
    print("===========================\n")

    with mujoco.viewer.launch_passive(
            model, data, show_right_ui=False, show_left_ui=False,
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            # --- 1. CUBE LOGIC (Floating vs Physics) ---
            if keys[pygame.K_r]:  # Reset Position
                cube_frozen = True

            # Toggle Freeze with F (Debounced logic omitted for simplicity, just hold F to toggle unlikely)
            if keys[pygame.K_f]:
                cube_frozen = not cube_frozen
                time.sleep(0.2)  # Debounce

            if cube_frozen and cube_adr != -1:
                # Force cube to target position, zero velocity
                data.qpos[cube_adr:cube_adr + 3] = cube_target_pos
                # Reset orientation to flat
                data.qpos[cube_adr + 3:cube_adr + 7] = [1, 0, 0, 0]
                data.qvel[cube_adr:cube_adr + 6] = 0

            # --- 2. ROBOT CONTROL ---
            # A. XYZ IK Target Update
            move_speed = 0.00025
            if keys[pygame.K_w]: target_ee_pos[0] += move_speed
            if keys[pygame.K_s]: target_ee_pos[0] -= move_speed
            if keys[pygame.K_a]: target_ee_pos[1] += move_speed
            if keys[pygame.K_d]: target_ee_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: target_ee_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: target_ee_pos[2] -= move_speed

            # B. Calculate IK for first 3 joints (Position)
            current_ee_pos = data.xpos[ee_id]
            error = target_ee_pos - current_ee_pos
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, ee_id)

            # Damped Least Squares for stability
            lmbda = 0.01
            jac_pinv = jacp.T @ np.linalg.inv(jacp @ jacp.T + lmbda * np.eye(3))

            limit = 0.1
            error_clamped = np.clip(error, -limit, limit)
            dq = jac_pinv @ error_clamped

            # C. Apply Control
            # We assume first 6 actuators are the arm
            n_actuators = 6
            current_qpos = data.qpos[:n_actuators].copy()

            # Base arm movement from IK
            new_qpos = current_qpos + dq[:n_actuators]

            # D. Manual Wrist/Gripper Overrides (The "Human Expert" part)
            manual_speed = 0.02

            # Wrist Flex (Joint 4) - Pitch
            if keys[pygame.K_UP]: new_qpos[3] -= manual_speed
            if keys[pygame.K_DOWN]: new_qpos[3] += manual_speed

            # Wrist Roll (Joint 5) - Rotation
            if keys[pygame.K_LEFT]: new_qpos[4] += manual_speed
            if keys[pygame.K_RIGHT]: new_qpos[4] -= manual_speed

            # Gripper (Joint 6)
            if keys[pygame.K_RIGHTBRACKET]:  # Close ]
                new_qpos[5] -= manual_speed
            if keys[pygame.K_LEFTBRACKET]:  # Open [
                new_qpos[5] += manual_speed

            # Clip Gripper to limits
            new_qpos[5] = np.clip(new_qpos[5], model.jnt_range[5, 0], model.jnt_range[5, 1])

            # Apply to Simulation
            data.ctrl[:n_actuators] = new_qpos

            # --- 3. DATA EXPORT ---
            if keys[pygame.K_RETURN]:
                print("\n" + "=" * 30)
                print(">>> COPY THIS INTO YOUR RESET() <<<")
                print(f"Cube Position (Air): {cube_target_pos.tolist()}")

                # Format numpy array string nicely
                q_str = ", ".join([f"{x:.4f}" for x in data.qpos[:6]])
                print(f"perfect_grasp_qpos = np.array([{q_str}])")
                print("=" * 30 + "\n")
                time.sleep(0.5)  # Prevent spam

            # --- 4. PHYSICS & RENDER ---
            mujoco.mj_step(model, data)

            # Visualize the IK Target
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=target_ee_pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.5]
            )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    pygame.quit()


if __name__ == "__main__":
    laboratory_mode()
