import numpy as np

def simp_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    return np.array([x, y])

def inverse_kinematics(target, l1, l2, l3, initial_guess, iterations=1000, alpha=0.1):
    t1, t2, t3 = initial_guess
    for _ in range(iterations):
        current_pos = forward_kinematics(t1, t2, t3, l1, l2, l3)
        error = target - current_pos
        if np.linalg.norm(error) < 1e-3:
            break

        # Jacobian transpose method
        s1 = np.sin(t1)
        c1 = np.cos(t1)
        s12 = np.sin(t1 + t2)
        c12 = np.cos(t1 + t2)
        s123 = np.sin(t1 + t2 + t3)
        c123 = np.cos(t1 + t2 + t3)
        J_t = np.array([
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123]
        ]).T
        delta_theta = alpha * J_t @ error
        t1 += delta_theta[0]
        t2 += delta_theta[1]
        t3 += delta_theta[2]
    return t1, t2, t3

# Define the arm
lengths = (1.0, 1.0, 0.5) # Lengths of the three links
# initial_thetas = (3.4, 3.139, -0.6116)
initial_thetas = (3.4, 3.139, 5.672)

# Define the target
target_position = np.array([0.712, 1.107])

print(f"Target Position: {target_position}")
print(f"Initial Thetas: {initial_thetas}")

# Calculate the inverse kinematics
final_thetas = inverse_kinematics(target_position, *lengths, initial_thetas)
# final_thetas = []
# for i, theta in enumerate(final_thetas_):
#     final_thetas.append(simp_angle(theta))
print(f"\nFinal Joint Angles (radians): {final_thetas}")
print(f"Final Joint Angles (degrees): {np.rad2deg(final_thetas)}")
"""
Counter clockwise is positive here and clockwise is negative
These angles are relative to the previous link's angle
"""

# --- Verification ---
# Use the calculated angles to see where the arm ends up.
final_position = forward_kinematics(*final_thetas, *lengths)
print(f"\nPosition reached with final angles: {final_position}")
print(f"Error (distance to target): {np.linalg.norm(target_position - final_position):.6f}")

final_thetas_ = np.array([final_thetas[0], final_thetas[0] + final_thetas[1], final_thetas[0] + final_thetas[1] + final_thetas[2]])
print(f"Final Joint Angles (degrees): {np.rad2deg(final_thetas_)}")
"""
These angles are relative to the positive x-direction of the cartesian plane
"""
