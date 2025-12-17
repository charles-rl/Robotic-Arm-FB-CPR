import mujoco
import mujoco.viewer
import numpy as np
import mink
import pygame
import time


# --- HELPER: Iterative Solver ---
def converge_ik(configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters):
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # --- 1. SETUP ---
    model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    # Setup Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Controlling Mocap Body 'target'")
    clock = pygame.time.Clock()

    # --- 2. LOCATE BODIES ---
    # Find the IDs we need
    try:
        # The mocap body defined in your XML
        mocap_id = model.body("target").mocapid[0]
    except KeyError:
        print("Error: Could not find body named 'target' with mocap='true' in XML.")
        return

    # The site on the robot we want to control
    ee_frame_name = "gripperframe"

    # --- 3. DEFINE TASKS ---
    ee_task = mink.FrameTask(
        frame_name=ee_frame_name,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=1e-2)
    tasks = [ee_task, posture_task]

    # --- 4. INITIALIZATION ---
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

    # --- CRITICAL: Snap Mocap to Robot ---
    # We move the red box (mocap) to where the gripper currently is.
    # This prevents the robot from "jumping" to the box's default location (0.5, 0, 0.5)
    site_id = model.site(ee_frame_name).id
    start_pos = data.site_xpos[site_id].copy()
    start_mat = data.site_xmat[site_id].reshape(3, 3).copy()

    # Write to MuJoCo data
    data.mocap_pos[mocap_id] = start_pos
    # Convert Matrix to Quat for mocap (MuJoCo expects [w, x, y, z])
    start_quat = np.zeros(4)
    mujoco.mju_mat2Quat(start_quat, start_mat.flatten())
    data.mocap_quat[mocap_id] = start_quat

    # Maintain a local target for math operations
    target_pose = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(start_mat),
        start_pos
    )

    # Settings
    solver = "quadprog"
    dt = model.opt.timestep
    move_speed = 0.005
    rot_speed = 0.05

    print("System Ready. Moving Mocap Body.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            clock.tick(60)

            # --- A. INPUT (Modify Target Pose) ---
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]: break

            # Position Delta
            d_pos = np.zeros(3)
            if keys[pygame.K_w]: d_pos[0] += move_speed
            if keys[pygame.K_s]: d_pos[0] -= move_speed
            if keys[pygame.K_a]: d_pos[1] += move_speed
            if keys[pygame.K_d]: d_pos[1] -= move_speed
            if keys[pygame.K_SPACE]: d_pos[2] += move_speed
            if keys[pygame.K_LCTRL]: d_pos[2] -= move_speed

            # Rotation Delta
            d_rot = mink.SO3.identity()
            if keys[pygame.K_r]: d_rot = d_rot @ mink.SO3.from_x_radians(rot_speed)
            if keys[pygame.K_f]: d_rot = d_rot @ mink.SO3.from_x_radians(-rot_speed)
            if keys[pygame.K_t]: d_rot = d_rot @ mink.SO3.from_y_radians(rot_speed)
            if keys[pygame.K_g]: d_rot = d_rot @ mink.SO3.from_y_radians(-rot_speed)
            if keys[pygame.K_y]: d_rot = d_rot @ mink.SO3.from_z_radians(rot_speed)
            if keys[pygame.K_h]: d_rot = d_rot @ mink.SO3.from_z_radians(-rot_speed)

            # Apply Deltas to Local Target
            new_trans = target_pose.translation() + d_pos
            new_rot = d_rot @ target_pose.rotation()  # Global rotation
            target_pose = mink.SE3.from_rotation_and_translation(new_rot, new_trans)

            # --- B. UPDATE MUJOCO MOCAP BODY ---
            # This makes the red box move in the viewer
            data.mocap_pos[mocap_id] = target_pose.translation()

            # Convert Mink SO3 -> MuJoCo Quat
            # We get the matrix from Mink, then convert to Quat using MuJoCo's helper
            mat = target_pose.rotation().as_matrix()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat.flatten())
            data.mocap_quat[mocap_id] = quat

            # --- C. UPDATE MINK TASK ---
            # Tell the IK solver to chase the Mocap body's new position
            ee_task.set_target(target_pose)

            # --- D. SOLVE IK ---
            configuration.update(data.qpos)
            converge_ik(
                configuration,
                tasks,
                dt,
                solver,
                pos_threshold=1e-4,
                ori_threshold=1e-4,
                max_iters=20,
            )

            # --- E. APPLY & STEP ---
            data.ctrl[:model.nu] = configuration.q[:model.nu]
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
