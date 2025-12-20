import time
import numpy as np
import mujoco
import mujoco.viewer
import pygame

"""
Lift:
Center
Cube Position (Air): [-0.25, 0.0, 0.63]
perfect_grasp_qpos = np.array([-0.0256, -1.6549, 1.0216, 0.5479, -1.5147, 0.4505])

Left
Cube Position (Air): [-0.25, 0.15, 0.63]
perfect_grasp_qpos = np.array([-0.7104, -0.9144, 0.7525, 0.2296, 1.6324, 0.4714])

Right
Cube Position (Air): [-0.25, -0.15, 0.63]
perfect_grasp_qpos = np.array([0.7957, -0.9497, 0.8520, 0.0732, 1.5838, 0.4697])

Far Left
Cube Position (Air): [-0.1, 0.1, 0.63]
perfect_grasp_qpos = np.array([-0.2480, 0.0105, 0.0497, -0.0548, 1.6172, 0.6541])

Far Right
Cube Position (Air): [-0.1, -0.1, 0.63]
perfect_grasp_qpos = np.array([0.2271, -0.0655, 0.1389, -0.0051, -1.5699, 0.6396])


Hoist:
Center
Cube Position (Table): [-0.25, 0.0, 0.43]
perfect_grasp_qpos = np.array([-0.1105, -0.6478, 1.2145, 0.6488, -1.5149, 0.7772])

Left
Cube Position (Air): [-0.25, 0.15, 0.43]
perfect_grasp_qpos = np.array([-0.8048, -0.1693, 0.9395, 0.2268, -1.6803, 0.4066])

Right
Cube Position (Air): [-0.25, -0.15, 0.43]
perfect_grasp_qpos = np.array([0.8181, -0.1458, 0.9392, 0.1610, 1.6450, 0.5912])

Far Left
Cube Position (Air): [-0.1, 0.1, 0.43]
perfect_grasp_qpos = np.array([-0.3462, 0.7594, 0.3037, -0.7716, -1.6587, 0.5156])

Far Right
Cube Position (Table): [-0.1, -0.1, 0.43]
perfect_grasp_qpos = np.array([0.2556, 0.8585, 0.2434, -0.8558, -1.5711, 0.7101])
"""


def check_smart_contact(model, data):
    """
    Returns True if AND ONLY IF both gripper pads are touching the cube.
    This prevents reward hacking (pushing table) because it checks specific Geom IDs.
    """
    try:
        # Get Geom IDs from XML names
        # Make sure your XML has <geom name="cube_a_geom"...> etc.
        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_a_geom")
        left_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad_geom")
        right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad_geom")

        # If IDs are -1, the name doesn't exist in XML
        if cube_id == -1 or left_pad_id == -1 or right_pad_id == -1:
            return False, "XML_ERROR"

    except Exception as e:
        return False, "XML_ERROR"

    left_touch = False
    right_touch = False

    # Iterate through all active contacts in the physics engine
    for i in range(data.ncon):
        contact = data.contact[i]

        # Contacts involve two geoms (geom1, geom2)
        c1 = contact.geom1
        c2 = contact.geom2

        # Check Left Pad <-> Cube
        if (c1 == left_pad_id and c2 == cube_id) or (c2 == left_pad_id and c1 == cube_id):
            left_touch = True

        # Check Right Pad <-> Cube
        if (c1 == right_pad_id and c2 == cube_id) or (c2 == right_pad_id and c1 == cube_id):
            right_touch = True

    is_grasping = left_touch and right_touch
    return is_grasping, "OK"


def laboratory_mode_with_sensors():
    # --- 1. SETUP ---
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("TESTING SENSORS | Press F to Freeze Cube")

    # Load Model
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # IDs
    ee_id = model.body("gripper").id
    try:
        cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_a_joint")
        cube_adr = model.jnt_qposadr[cube_joint_id]
    except:
        print("Warning: Could not find 'cube_a_joint'. Check XML names.")
        cube_adr = -1

    # State Variables
    target_ee_pos = data.xpos[ee_id].copy()
    cube_frozen = True
    cube_target_pos = np.array([-0.10, 0.10, 0.43])

    print("\n=== SENSOR TEST MODE ===")
    print(" [W/S/A/D/Spc/Ctrl] : Move")
    print(" [ ] / [ ]          : Gripper")
    print(" [ F ]              : Toggle Physics")
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

            # --- CONTROL LOGIC (Same as yours) ---
            move_speed = 0.0005
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
            limit = 0.1
            error_clamped = np.clip(error, -limit, limit)
            dq = jac_pinv @ error_clamped

            n_actuators = 6
            current_qpos = data.qpos[:n_actuators].copy()
            new_qpos = current_qpos + dq[:n_actuators]

            manual_speed = 0.02
            if keys[pygame.K_UP]: new_qpos[3] -= manual_speed
            if keys[pygame.K_DOWN]: new_qpos[3] += manual_speed
            if keys[pygame.K_LEFT]: new_qpos[4] += manual_speed
            if keys[pygame.K_RIGHT]: new_qpos[4] -= manual_speed
            if keys[pygame.K_RIGHTBRACKET]: new_qpos[5] -= manual_speed
            if keys[pygame.K_LEFTBRACKET]: new_qpos[5] += manual_speed
            new_qpos[5] = np.clip(new_qpos[5], model.jnt_range[5, 0], model.jnt_range[5, 1])
            data.ctrl[:n_actuators] = new_qpos

            # --- PHYSICS STEP ---
            mujoco.mj_step(model, data)

            # --- SENSOR CHECK ---
            is_grasping, status = check_smart_contact(model, data)

            # Visual Feedback
            # Red Sphere = Not Grasping
            # Green Sphere = SOLID GRASP (Both pads touching cube)
            sphere_color = [0, 1, 0, 0.8] if is_grasping else [1, 0, 0, 0.3]
            sphere_size = [0.02, 0, 0] if is_grasping else [0.01, 0, 0]

            if status == "XML_ERROR":
                pygame.display.set_caption("ERROR: Check XML Geom Names!")
            else:
                caption = f"GRASP: {'YES' if is_grasping else 'NO'} | Cube Frozen: {cube_frozen}"
                pygame.display.set_caption(caption)

            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=sphere_size,
                pos=target_ee_pos,
                mat=np.eye(3).flatten(),
                rgba=sphere_color
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
    cube_target_pos = np.array([-0.10, 0.10, 0.43])  # The "Air" position you want to test

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
            move_speed = 0.0005
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
    laboratory_mode_with_sensors()