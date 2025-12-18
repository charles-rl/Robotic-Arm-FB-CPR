import numpy as np
import torch
import mujoco


# ==========================================
# 1. The Official PyTorch Implementation
#    (Copied exactly from your snippet)
# ==========================================
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    # v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def compute_rotation_matrix_from_ortho6d(poses):
    """
    The Ground Truth logic.
    Constructs rotation matrix from 6D input.
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# ==========================================
# 2. My NumPy Implementation (From RobotEnv)
# ==========================================
def _rot6d_to_mat_numpy(rot6d):
    """
    My implementation using Gram-Schmidt logic in NumPy.
    Input: np.array of shape (6,)
    Output: np.array of shape (3,3)
    """
    # 1. Extract vectors
    a1, a2 = rot6d[:3], rot6d[3:]

    # 2. Normalize first vector (x-axis)
    x = a1 / (np.linalg.norm(a1) + 1e-8)

    # 3. Orthogonalize second vector (y-axis)
    # Note: The PyTorch code does: z = cross(x, y_raw), then y = cross(z, x).
    # This is mathematically equivalent to Gram-Schmidt projection
    # followed by a cross product, ensuring orthogonality.

    # Let's use the explicit PyTorch logic flow, but in NumPy,
    # to guarantee exact parity:

    # A. Get Z first (Perpendicular to X and Y_raw)
    z = np.cross(x, a2)
    z = z / (np.linalg.norm(z) + 1e-8)

    # B. Get Y (Perpendicular to Z and X)
    y = np.cross(z, x)

    # 4. Construct Matrix
    # Columns: [x, y, z]
    mat = np.stack([x, y, z], axis=1)
    return mat


def _rot6d_to_quat_custom(rot6d):
    """
    The full function I gave you, calculating Quat via MuJoCo
    """
    mat = _rot6d_to_mat_numpy(rot6d)
    quat = np.zeros(4)
    # MuJoCo expects flat matrix array
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


# ==========================================
# 3. Verification Loop
# ==========================================
def run_comparison():
    print(f"{'ITER':<6} | {'DIFF (Matrix)':<15} | {'RESULT':<10}")
    print("-" * 40)

    success_count = 0
    num_tests = 10000

    for i in range(num_tests):
        # 1. Generate Random Action in range [-1, 1]
        # This simulates your Neural Network output
        raw_action = np.random.uniform(-1, 1, size=(6,)).astype(np.float32)

        # --- Run PyTorch (Ground Truth) ---
        torch_input = torch.tensor(raw_action, dtype=torch.float32).unsqueeze(0)  # Batch size 1
        torch_mat = compute_rotation_matrix_from_ortho6d(torch_input)
        gt_matrix = torch_mat[0].numpy()  # Convert to numpy (3x3)

        # --- Run NumPy (My Implementation) ---
        my_matrix = _rot6d_to_mat_numpy(raw_action)

        # --- Compare ---
        # We verify if the rotation matrices are identical
        diff = np.abs(gt_matrix - my_matrix).max()

        passed = diff < 2e-5
        if passed: success_count += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i:<6} | {diff:.8f}        | {status}")

    print("-" * 40)
    print(f"Final Score: {success_count}/{num_tests} matches.")

    if success_count == num_tests:
        print("\n🏆 CONCLUSION: The NumPy implementation is mathematically identical to the PyTorch reference.")
        print("   Your [-1, 1] bounded actions are perfectly fine.")


if __name__ == "__main__":
    run_comparison()