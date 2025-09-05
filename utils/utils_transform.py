import numpy as np
from torch.nn import functional as F
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose, matrot2aa
import torch

""" 
----- AvatarPoser Transform ----- 
"""

def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
    a2 = d6s[:,:,1]
    c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
    b2 = F.normalize(a2-c,p=2,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=-1)

def matrot2sixd(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    '''
    pose_6d = torch.cat([pose_matrot[:,:3,0], pose_matrot[:,:3,1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    '''
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    '''
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d

def sixd2matrot(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    '''
    rot_vec_1 = pose_6d[:,:3]
    rot_vec_2 = pose_6d[:,3:6]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)
    pose_matrot = torch.stack([rot_vec_1,rot_vec_2,rot_vec_3],dim=-1)
    return pose_matrot

def sixd2aa(pose_6d, batch = False):
    '''
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    '''
    if batch:
        B,J,C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1,6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B,J,3)
    return pose_aa

def sixd2quat(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    '''
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat((pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1)
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion

def quat2aa(pose_quat):
    '''
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    '''
    return tgm.quaternion_to_angle_axis(pose_quat)


""" 
----- SFBPE Transform -----
"""

EPSILON = 1e-12

def two_axis_to_angle_axis(two_axis: torch.Tensor) -> torch.Tensor:
    """
    Direct conversion from 6D two-axis representation to axis-angle representation.
    
    Args:
        two_axis: Rotation in two-axis representation, shape (..., 6)
        
    Returns:
        Rotations in axis-angle form, shape (..., 3)
    """
    *leading_shape, last_dim = two_axis.shape
    assert last_dim == 6
    
    # Extract and normalize the two axes
    a1 = two_axis[..., 0:3]
    a2 = two_axis[..., 3:6]
    
    # Normalize first axis
    norm_a1 = torch.linalg.norm(a1, dim=-1, keepdim=True)
    a1_norm = a1 / (norm_a1 + 1e-8)
    
    # Orthogonalize and normalize second axis
    dot = torch.sum(a2 * a1_norm, dim=-1, keepdim=True)
    a2_ortho = a2 - dot * a1_norm
    norm_a2_ortho = torch.linalg.norm(a2_ortho, dim=-1, keepdim=True)
    a2_norm = a2_ortho / (norm_a2_ortho + 1e-8)
    
    # Compute third axis (cross product)
    a3_norm = torch.linalg.cross(a1_norm, a2_norm, dim=-1)
    
    # Reconstruct rotation matrix columns
    col1, col2, col3 = a1_norm, a2_norm, a3_norm
    
    # Direct computation of axis-angle from matrix elements
    # Using the same logic as matrix_to_angle_axis but without building full matrix
    
    # Compute omega vector (matrix[2,1] - matrix[1,2], etc.)
    omega_x = col3[..., 1] - col2[..., 2]  # m[2,1] - m[1,2]
    omega_y = col1[..., 2] - col3[..., 0]  # m[0,2] - m[2,0]  
    omega_z = col2[..., 0] - col1[..., 1]  # m[1,0] - m[0,1]
    
    omegas = torch.stack([omega_x, omega_y, omega_z], dim=-1)
    
    # Compute trace: sum of diagonal elements
    trace = col1[..., 0] + col2[..., 1] + col3[..., 2]
    
    # Compute angle using atan2
    norm_omegas = torch.linalg.norm(omegas, dim=-1, keepdim=True)
    angles = torch.atan2(norm_omegas, trace.unsqueeze(-1) - 1)
    
    # Handle near-zero angle case
    zeros = torch.zeros_like(omegas)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)
    
    # Handle normal case (angle not near pi)
    axis_angles = torch.empty_like(omegas)
    normal_case = ~angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)
    
    if normal_case.any():
        sinc_vals = torch.sinc(angles[normal_case] / torch.pi)
        axis_angles[normal_case] = 0.5 * omegas[normal_case] / (sinc_vals + 1e-8)
    
    # Handle angle near pi case (special handling)
    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)
    if near_pi.any():
        # For near-pi case, we need the full matrix to compute the axis
        # This is the slow path, but should be rare
        matrix_near_pi = torch.stack([
            col1[near_pi], col2[near_pi], col3[near_pi]
        ], dim=-2)
        
        n = 0.5 * (
            matrix_near_pi[..., 0, :] + 
            torch.eye(3, dtype=two_axis.dtype, device=two_axis.device)
        )
        n_norm = torch.linalg.norm(n, dim=-1, keepdim=True)
        axis_angles[near_pi] = angles[near_pi] * n / (n_norm + 1e-8)
    
    return axis_angles

def angle_axis_to_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Converts from angle axis rotation representation to 3x3 matrix
    :param angle_axis: shape (..., 3)
    :return: shape (..., 3, 3)
    """
    angles = torch.linalg.norm(angle_axis + EPSILON, dim=-1)
    axes = torch.div(angle_axis, angles[..., None])
    angles_cos = torch.cos(angles)
    angles_sin = torch.sin(angles)

    device = angles.device
    dtype = angles.dtype
    dims = angles.shape

    k = torch.zeros(dims + (3, 3), dtype=dtype, device=device)
    k[..., 0, 1] = -axes[..., 2]
    k[..., 0, 2] = axes[..., 1]
    k[..., 1, 0] = axes[..., 2]
    k[..., 1, 2] = -axes[..., 0]
    k[..., 2, 0] = -axes[..., 1]
    k[..., 2, 1] = axes[..., 0]

    r = torch.zeros(dims + (3, 3), dtype=dtype, device=device)
    r[..., 0, 0] = 1
    r[..., 1, 1] = 1
    r[..., 2, 2] = 1
    r += angles_sin[..., None, None] * k
    r += (1 - angles_cos[..., None, None]) * torch.matmul(k, k)
    return r

def matrix_to_angle_axis(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = (
        0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles

def rotation_angle_radians(mat: torch.Tensor):
    # Input must be a rotation matrix; i.e. with shape (..., 3, 3)
    # Math: https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
    assert mat.shape[-2:] == (3, 3)
    trace = torch.einsum('...ii->...', mat)
    arg = (trace - 1.0) / 2.0
    arg = arg.clamp(-1.0, 1.0)  # clip to prevent NaNs
    angle = torch.arccos(arg)
    return angle

#old
def _matrix_to_angle_axis(matrix: torch.Tensor, warn: bool = True) -> torch.Tensor:
    """
    Converts from 3x3 matrix rotation representation to angle axis
    :param matrix: shape (..., 3, 3)
    :param warn: whether to warn user to use function
    :return: shape (..., 3)
    """
    if warn:
        print("WARNING: matrix_to_angle_axis usage incurs error; set warn=False to disable warning")
    *leading_shape, w, h = matrix.shape
    assert w == 3
    assert h == 3

    angle_axis = torch.empty(leading_shape + [3], dtype=matrix.dtype, device=matrix.device)

    # Compute axis
    angle_axis[..., 0] = matrix[..., 2, 1] - matrix[..., 1, 2]
    angle_axis[..., 1] = matrix[..., 0, 2] - matrix[..., 2, 0]
    angle_axis[..., 2] = matrix[..., 1, 0] - matrix[..., 0, 1]

    angle_axis_norm = torch.linalg.norm(angle_axis + EPSILON, dim=-1, keepdim=True)
    angle_axis = angle_axis / angle_axis_norm

    angle = rotation_angle_radians(matrix)

    angle_axis *= angle[..., None]
    return angle_axis


def matrix_to_two_axis(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts from 3x3 matrix rotation representation to 6D two-axis representation
    :param matrix: shape (..., 3, 3)
    :return: shape (..., 6)
    """
    col1 = matrix[..., 0]
    col2 = matrix[..., 1]
    return torch.cat((col1, col2), dim=-1)

def two_axis_to_matrix_onnx_friendly(two_axis: torch.Tensor) -> torch.Tensor:

    """
    Converts from 6D two-axis rotation representation to 3x3 matrix representation
    :param two_axis: shape (..., 6)
    :return: shape (..., 3, 3)
    """
    *leading_shape, last_dim = two_axis.shape
    assert last_dim == 6

    matrix = torch.empty(leading_shape + [3, 3], dtype=two_axis.dtype, device=two_axis.device)

    in1 = two_axis[..., 0:3]
    # Replace torch.linalg.norm with manual computation
    norm1 = torch.sqrt(torch.sum(in1 * in1, dim=-1, keepdim=True) + EPSILON)
    col1 = in1 / norm1
    matrix[..., 0] = col1

    in2 = two_axis[..., 3:6]
    # Replace torch.linalg.vecdot with torch.sum
    dot_product = torch.sum(in2 * col1, dim=-1, keepdim=True)
    col2 = in2 - dot_product * col1
    # Replace torch.linalg.norm with manual computation
    norm2 = torch.sqrt(torch.sum(col2 * col2, dim=-1, keepdim=True) + EPSILON)
    col2 = col2 / norm2
    matrix[..., 1] = col2

    col3 = torch.linalg.cross(col1, col2, dim=-1)
    matrix[..., 2] = col3

    return matrix

def matrix_to_angle_axis_onnx_friendly(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.
    Handles shapes like [1, 1, 3, 3] and [1, 1, 21, 3, 3].
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    # Original shape for reshaping later
    original_shape = matrix.shape[:-2]  # [1, 1] or [1, 1, 21]
    flat_shape = (-1, 3, 3)  # Flatten all batch dimensions
    matrix_flat = matrix.view(flat_shape)
    batch_size = matrix_flat.shape[0]

    # Compute omegas (skew-symmetric part)
    omegas = torch.stack([
        matrix_flat[..., 2, 1] - matrix_flat[..., 1, 2],
        matrix_flat[..., 0, 2] - matrix_flat[..., 2, 0],
        matrix_flat[..., 1, 0] - matrix_flat[..., 0, 1]
    ], dim=-1)  # shape: [batch_size, 3]

    # Compute norms and traces
    norms = torch.sqrt(torch.sum(omegas * omegas, dim=-1, keepdim=True) + 1e-8)
    traces = matrix_flat[..., 0, 0] + matrix_flat[..., 1, 1] + matrix_flat[..., 2, 2]
    traces = traces.unsqueeze(-1)
    
    # Compute angles
    angles = torch.atan2(norms, traces - 1)

    # Initialize axis_angles
    axis_angles = torch.zeros_like(omegas)

    # Handle normal case (angle not near zero or pi)
    zero_mask = (angles < 1e-6).squeeze(-1)
    pi_mask = (torch.abs(angles - torch.pi) < 1e-6).squeeze(-1)
    normal_mask = ~zero_mask & ~pi_mask

    if normal_mask.any():
        # For normal angles: axis_angle = 0.5 * omega / sinc(angle/pi)
        angle_vals = angles[normal_mask]
        omega_vals = omegas[normal_mask]
        
        # Manual sinc computation: sinc(x) = sin(πx)/(πx)
        x = angle_vals / torch.pi
        sinc_vals = torch.sin(x) / x
        
        axis_angles[normal_mask] = 0.5 * omega_vals / (sinc_vals + 1e-8)

    # Handle near-zero angles (set to zero)
    axis_angles[zero_mask.squeeze(-1)] = 0.0

    # Handle near-pi angles
    if pi_mask.any():
        # For angles near pi: use eigenvector approach
        pi_matrices = matrix_flat[pi_mask]
        # R + I
        r_plus_i = pi_matrices + torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        # Take first column as eigenvector approximation
        eigenvectors = r_plus_i[..., :1].squeeze(-2)  # shape: [n_pi, 3]
        
        # Normalize eigenvectors
        eigen_norms = torch.sqrt(torch.sum(eigenvectors * eigenvectors, dim=-1, keepdim=True) + 1e-8)
        eigenvectors_normalized = eigenvectors / eigen_norms
        
        # Scale by angle (pi)
        axis_angles[pi_mask] = angles[pi_mask] * eigenvectors_normalized

    # Reshape back to original format
    final_shape = original_shape + (3,)
    return axis_angles.view(final_shape)

    
def two_axis_to_matrix(two_axis: torch.Tensor) -> torch.Tensor:
    """
    Converts from to 6D two-axis rotation representation to 3x3 matrix representation
    :param two_axis: shape (..., 6)
    :return: shape (..., 3, 3)
    """
    *leading_shape, last_dim = two_axis.shape
    assert last_dim == 6

    matrix = torch.empty(leading_shape + [3, 3], dtype=two_axis.dtype, device=two_axis.device)

    in1 = two_axis[..., 0:3]
    col1 = in1 / torch.linalg.norm(in1 + EPSILON, dim=-1, keepdim=True)
    matrix[..., 0] = col1

    in2 = two_axis[..., 3:6]
    col2 = in2 - torch.linalg.vecdot(in2, col1)[..., None] * col1
    col2 = col2 / torch.linalg.norm(col2 + EPSILON, dim=-1, keepdim=True)
    matrix[..., 1] = col2

    col3 = torch.linalg.cross(col1, col2, dim=-1)
    matrix[..., 2] = col3

    return matrix


def rotational_fk(global_orient_3x3: torch.Tensor, body_pose_3x3: torch.Tensor):
    """
    Aggregates local rotations (body_pose_mat) to compute world-space rotations.
    Expects and returns in 3x3 matrix format.
    :param global_orient_3x3: shape (..., 3, 3)
    :param body_pose_3x3: (..., 21, 3, 3)
    :return: shape (..., 21, 3, 3)
    """
    import anim.data.amass as data
    assert global_orient_3x3.shape[-2:] == (3, 3)
    assert body_pose_3x3.shape[-3:] == (data.SmplxJoints.NUM_JTS - 1, 3, 3), f"shape={body_pose_3x3.shape}"
    hierarchy = data.SMPLX_BODY_HIERARCHY

    # We have to use lists of tensors for jts here or else pytorch anomaly detection will get annoyed at us :)
    body_pose_global_3x3_list = []
    for jt in range(1, data.SmplxJoints.NUM_JTS):
        parent_jt = hierarchy[jt]
        parent_mat = global_orient_3x3[..., None, :, :] if parent_jt == 0 else body_pose_global_3x3_list[parent_jt - 1]
        body_pose_global_3x3_list.append(parent_mat @ body_pose_3x3[..., jt - 1:jt, :, :])
    return torch.cat(body_pose_global_3x3_list, dim=-3)


def to_homogeneous(pos: torch.Tensor, rot: torch.Tensor):
    """
    Convert position and rotation matrix to a 4x4 homogeneous transformation matrix.
    
    Args:
        pos: Tensor of shape (batch_size, seq_len, 3)
        rot: Tensor of shape (batch_size, seq_len, 3, 3)
    
    Returns:
        Tensor of shape (batch_size, seq_len, 4, 4)
    """
    batch_size, seq_len, _, _ = rot.shape
    # Create homogeneous matrix initialized as identity
    homogeneous_matrix = torch.eye(4).expand(batch_size, seq_len, 4, 4).clone()
    # Assign rotation (top-left 3x3 block)
    homogeneous_matrix[:, :, :3, :3] = rot
    # Assign translation (last column before the last row)
    homogeneous_matrix[:, :, :3, 3] = pos

    return homogeneous_matrix
