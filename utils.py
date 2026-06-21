import torch

def quaternion_to_rotation_matrix(quat):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    Args:
        quat: A tensor of shape (..., 4) in [w, x, y, z] format.
    Returns:
        A tensor of shape (..., 3, 3) representing the rotation matrix.
    """
    # Normalize the quaternion to ensure it represents a valid rotation
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Precompute squared terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    # Construct the 3x3 matrix
    rot_matrix = torch.stack([
        torch.stack([1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)], dim=-1),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)], dim=-1)
    ], dim=-2)
    
    return rot_matrix
