import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation


# --- 1. THE PYTORCH FUNCTION (From your snippet) ---
# Code from https://github.com/papagina/RotationContinuity/blob/master/Inverse_Kinematics/code/tools.py#L142
def compute_rotation_matrix_from_quaternion(quaternion):
    # Setup for single batch
    batch = quaternion.shape[0]

    quat = quaternion

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3
    return matrix


# --- 2. MY HELPER FUNCTION (MuJoCo) ---
def _quat_to_rot6d_mujoco(quat):
    # Expects [w, x, y, z]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    mat = mat.reshape(3, 3)
    # Return first two columns (Rotate6D)
    return np.concatenate([mat[:, 0], mat[:, 1]])


# --- 3. THE TEST ---
def test_quat_to_rot6d():
    # A random rotation (normalized) to be sure
    # Format: [w, x, y, z]

    # Create a rotation object from Euler angles specifying axes of rotation

    rot = Rotation.from_euler('xyz', np.random.uniform(-180, 180, size=(3,)), degrees=True)

    # Convert to quaternions and print
    rot_quat = rot.as_quat()
    test_quat = rot_quat.astype(np.float64)

    # test_quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)  # 90 deg around X

    print(f"Testing Quaternion [w,x,y,z]: {test_quat}")

    # A. MuJoCo Result
    mujoco_6d = _quat_to_rot6d_mujoco(test_quat)
    print(f"\nMuJoCo 6D Output:\n{mujoco_6d.round(4)}")

    # B. PyTorch Result
    # 1. Get the 3x3 Matrix
    torch_input = torch.tensor(test_quat, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    torch_mat = compute_rotation_matrix_from_quaternion(torch_input)
    # 2. Slice it manually to get Rotate6D (First two columns)
    # Shape is (1, 3, 3). We want [:, 0] and [:, 1]
    col1 = torch_mat[0, :, 0].numpy()
    col2 = torch_mat[0, :, 1].numpy()
    torch_6d = np.concatenate([col1, col2])

    print(f"\nPyTorch 6D Output:\n{torch_6d.round(4)}")

    # C. Verify
    diff = np.abs(mujoco_6d - torch_6d)
    if np.all(diff < 2e-5):
        print("\n✅ SUCCESS: MuJoCo and PyTorch implementations match!")
    else:
        print("\n❌ FAIL: They do not match.")

def _test_quat_to_rot6d():
    # A random rotation (normalized) to be sure
    # Format: [w, x, y, z]

    # Create a rotation object from Euler angles specifying axes of rotation

    rot = Rotation.from_euler('xyz', np.random.uniform(-180, 180, size=(3,)), degrees=True)

    # Convert to quaternions and print
    rot_quat = rot.as_quat()
    test_quat = rot_quat.astype(np.float64)

    # test_quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)  # 90 deg around X

    # A. MuJoCo Result
    mujoco_6d = _quat_to_rot6d_mujoco(test_quat)

    # B. PyTorch Result
    # 1. Get the 3x3 Matrix
    torch_input = torch.tensor(test_quat, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    torch_mat = compute_rotation_matrix_from_quaternion(torch_input)
    # 2. Slice it manually to get Rotate6D (First two columns)
    # Shape is (1, 3, 3). We want [:, 0] and [:, 1]
    col1 = torch_mat[0, :, 0].numpy()
    col2 = torch_mat[0, :, 1].numpy()
    torch_6d = np.concatenate([col1, col2])

    # C. Verify
    diff = np.abs(mujoco_6d - torch_6d)
    return np.all(diff < 2e-5)

