import numpy as np
def forward_kinematics(theta1, theta2, l1, l2):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])

def inverse_kinematics(target, l1, l2, initial_guess, iterations=1000, alpha=0.1):
    theta1, theta2 = initial_guess
    for _ in range(iterations):
        current_pos = forward_kinematics(theta1, theta2, l1, l2)
        error = target - current_pos
        if np.linalg.norm(error) < 1e-3:
            break

        # Jacobian transpose method
        J_t = np.array([[-l1*np.sin(theta1) - l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)],
        [l1*np.cos(theta1) + l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)]]).T
        delta_theta = alpha * J_t @ error
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]

    return theta1, theta2

l1, l2 = 1.0, 1.2
initial_guess = (0.1, -0.1)
target_position = np.array([1.0, 1.0])
theta1, theta2 = inverse_kinematics(target_position, l1, l2, initial_guess)

print(f"Joint angles: theta1={theta1}, theta2={theta2}")
print(f"Joint angles(degrees): theta1={np.rad2deg(theta1)}, theta2={np.rad2deg(theta2)}")

# Use the calculated angles to see where the arm ends up.
final_position = forward_kinematics(theta1, theta2, l1, l2)
print(f"\nPosition reached with final angles: {final_position}")
print(f"Error (distance to target): {np.linalg.norm(target_position - final_position):.6f}")