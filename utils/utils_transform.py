"""
Inspired by https://github.com/georgedf1/sfbpe/blob/main/utils.py
"""
# External
import argparse
import time

import torch

# Internal
import data.data_config as bm_config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_time_str() -> str:
    """Return a string describing the current local time."""
    t = time.localtime()
    return '{}-{}-{}-{}-{}-{}'.format(
        str(t.tm_year)[-2:], t.tm_mon, t.tm_mday,
        t.tm_hour, t.tm_min, t.tm_sec,
    )


# ---------------------------------------------------------------------------
# Rotation representation conversions
# ---------------------------------------------------------------------------

def angle_axis_to_matrix(angle_axis: torch.Tensor,
                          norm_threshold: float = 1e-8) -> torch.Tensor:
    """
    Convert angle-axis rotation to a 3×3 rotation matrix.

    Args:
        angle_axis:     (..., 3)
        norm_threshold: Below this angle norm the rotation is treated as identity.

    Returns:
        (..., 3, 3)
    """
    assert norm_threshold >= 0.0

    angle = torch.linalg.norm(angle_axis, dim=-1)
    axis = angle_axis / angle[..., None]

    # Replace near-zero axes with [1, 0, 0] (identity rotation)
    axis_fix = torch.tensor([1, 0, 0], dtype=axis.dtype, device=axis.device)
    axis = torch.where(angle[..., None] >= norm_threshold, axis, axis_fix)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    dims  = angle.shape

    k = torch.zeros(dims + (3, 3), dtype=angle.dtype, device=angle.device)
    k[..., 0, 1] = -axis[..., 2]
    k[..., 0, 2] =  axis[..., 1]
    k[..., 1, 0] =  axis[..., 2]
    k[..., 1, 2] = -axis[..., 0]
    k[..., 2, 0] = -axis[..., 1]
    k[..., 2, 1] =  axis[..., 0]

    I = torch.zeros_like(k)
    I[..., 0, 0] = I[..., 1, 1] = I[..., 2, 2] = 1

    return I + sin_a[..., None, None] * k + (1 - cos_a[..., None, None]) * (k @ k)


def matrix_to_angle_axis(matrix: torch.Tensor,
                          warn: bool = True,
                          axis_norm_threshold: float = 1e-8) -> torch.Tensor:
    """
    Convert a 3×3 rotation matrix to angle-axis representation.

    Note: This conversion accumulates small numerical errors.

    Args:
        matrix:              (..., 3, 3)
        warn:                Print a warning about numerical error when True.
        axis_norm_threshold: Fix axis when its norm falls below this value.

    Returns:
        (..., 3)
    """
    assert axis_norm_threshold >= 0.0
    assert matrix.shape[-2:] == (3, 3)

    if warn:
        print("WARNING: matrix_to_angle_axis incurs numerical error; "
              "set warn=False to suppress.")

    angle = rotation_angle_radians(matrix)

    axis = torch.empty(matrix.shape[:-2] + (3,), dtype=matrix.dtype, device=matrix.device)
    axis[..., 0] = matrix[..., 2, 1] - matrix[..., 1, 2]
    axis[..., 1] = matrix[..., 0, 2] - matrix[..., 2, 0]
    axis[..., 2] = matrix[..., 1, 0] - matrix[..., 0, 1]

    axis_norm = torch.linalg.norm(axis, dim=-1)
    axis = axis / (axis_norm + 1e-8)[..., None]

    axis_fix = torch.tensor([1, 0, 0], dtype=axis.dtype, device=axis.device)
    axis = torch.where(axis_norm[..., None] >= axis_norm_threshold, axis, axis_fix)

    return angle[..., None] * axis


def matrix_to_two_axis(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3×3 rotation matrix to the 6-D two-axis representation.

    Args:
        matrix: (..., 3, 3)

    Returns:
        (..., 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def two_axis_to_matrix(two_axis: torch.Tensor,
                        axis_norm_threshold: float = 1e-8,
                        onnx_friendly: bool = False) -> torch.Tensor:
    """
    Convert the 6-D two-axis representation to a 3×3 rotation matrix.

    Normalises input axes so the function is safe to use inside learned models.

    Args:
        two_axis:            (..., 6)
        axis_norm_threshold: Fix degenerate axes below this norm.
        onnx_friendly:       Use ``torch.sum`` instead of ``torch.linalg.vecdot``
                             for ONNX export compatibility.

    Returns:
        (..., 3, 3)
    """
    assert axis_norm_threshold >= 0.0
    assert two_axis.shape[-1] == 6

    matrix = torch.empty(two_axis.shape[:-1] + (3, 3),
                         dtype=two_axis.dtype, device=two_axis.device)

    # ---- first column ----
    in1   = two_axis[..., 0:3]
    norm1 = torch.linalg.norm(in1, dim=-1)
    col1  = in1 / norm1[..., None]
    fix1  = torch.tensor([1, 0, 0], dtype=col1.dtype, device=col1.device)
    col1  = torch.where(norm1[..., None] >= axis_norm_threshold, col1, fix1)
    matrix[..., 0] = col1

    # ---- second column (orthogonalised against col1) ----
    in2 = two_axis[..., 3:6]
    if onnx_friendly:
        dot = torch.sum(col1 * in2, dim=-1, keepdim=True)
    else:
        dot = torch.linalg.vecdot(in2, col1)[..., None]
    col2  = in2 - dot * col1
    norm2 = torch.linalg.norm(col2, dim=-1)
    col2  = col2 / norm2[..., None]
    fix2  = torch.tensor([0, 1, 0], dtype=col2.dtype, device=col2.device)
    col2  = torch.where(norm2[..., None] >= axis_norm_threshold, col2, fix2)
    matrix[..., 1] = col2

    # ---- third column (cross product; orthonormal by construction) ----
    matrix[..., 2] = torch.linalg.cross(col1, col2, dim=-1)

    return matrix


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def rotational_fk(global_orient_3x3: torch.Tensor,
                  body_pose_3x3: torch.Tensor) -> torch.Tensor:
    """
    Aggregate local rotations into world-space rotations.

    Args:
        global_orient_3x3: (..., 3, 3)
        body_pose_3x3:     (..., NUM_JTS - 1, 3, 3)

    Returns:
        (..., NUM_JTS - 1, 3, 3)
    """
    assert global_orient_3x3.shape[-2:] == (3, 3)
    assert body_pose_3x3.shape[-3:] == (bm_config.SmplxJoints.NUM_JTS - 1, 3, 3)

    hierarchy = bm_config.SMPLX_BODY_HIERARCHY
    global_list = []

    for jt in range(1, bm_config.SmplxJoints.NUM_JTS):
        parent = hierarchy[jt]
        parent_mat = (
            global_orient_3x3[..., None, :, :]
            if parent == 0
            else global_list[parent - 1]
        )
        global_list.append(parent_mat @ body_pose_3x3[..., jt - 1:jt, :, :])

    return torch.cat(global_list, dim=-3)


# ---------------------------------------------------------------------------
# Angle from rotation matrix
# ---------------------------------------------------------------------------

def rotation_angle_radians(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation angle (in radians) encoded in a rotation matrix.

    Args:
        mat: (..., 3, 3)

    Returns:
        (...,)
    """
    assert mat.shape[-2:] == (3, 3)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    arg   = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
    return torch.arccos(arg)


# ---------------------------------------------------------------------------
# Frame-wise deltas
# ---------------------------------------------------------------------------

def compute_rotation_delta(rot: torch.Tensor) -> torch.Tensor:
    """
    Compute per-frame rotation deltas (relative to the previous frame).

    The first frame is extrapolated using a zero-jerk approximation.

    Args:
        rot: (..., T, 3, 3)  with T >= 3.

    Returns:
        (..., T, 3, 3)
    """
    assert rot.shape[-3] >= 3 and rot.shape[-2:] == (3, 3)

    delta_rest = rot[..., 1:, :, :] @ torch.linalg.inv(rot[..., :-1, :, :])
    d1 = delta_rest[..., 0:1, :, :]
    d2 = delta_rest[..., 1:2, :, :]
    delta_first = d1 @ torch.linalg.inv(d2) @ d1  # zero-jerk: d0 = d1 · d2⁻¹ · d1
    return torch.cat([delta_first, delta_rest], dim=-3)


def compute_positional_delta(pos: torch.Tensor) -> torch.Tensor:
    """
    Compute per-frame positional deltas (relative to the previous frame).

    The first frame is extrapolated using a zero-jerk approximation.

    Args:
        pos: (..., T, D)  with T >= 3.

    Returns:
        (..., T, D)
    """
    assert pos.shape[-2] >= 3

    delta_rest  = pos[..., 1:, :] - pos[..., :-1, :]
    d1 = delta_rest[..., 0:1, :]
    d2 = delta_rest[..., 1:2, :]
    delta_first = 2 * d1 - d2  # zero-jerk: d0 = 2·d1 - d2
    return torch.cat([delta_first, delta_rest], dim=-2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _assert_all_close(ts1, ts2, atol=1e-6, label=''):
    if not torch.all(torch.isclose(ts1, ts2, atol=atol)):
        import inspect
        fi = inspect.getframeinfo(inspect.currentframe().f_back)
        print(f"FAILED '{label}' (line {fi.lineno} in {fi.filename})")
        breakpoint()


def _run_tests(test_egobody: bool = False):
    print(f"Running basic tests at {generate_time_str()}")

    aa_in  = torch.tensor([[torch.pi / 2, 0, 0], [0, 0, 0]], dtype=torch.float32)
    mat_ok = torch.tensor(
        [[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
         [[1, 0, 0], [0, 1,  0], [0, 0, 1]]], dtype=torch.float32)
    ta_ok  = torch.tensor([[1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0]], dtype=torch.float32)

    mat    = angle_axis_to_matrix(aa_in)
    ta     = matrix_to_two_axis(mat)
    mat_re = two_axis_to_matrix(ta)
    aa_re  = matrix_to_angle_axis(mat_re, warn=False)

    _assert_all_close(mat,    mat_ok, label='aa→mat')
    _assert_all_close(ta,     ta_ok,  label='mat→two_axis')
    _assert_all_close(mat_re, mat,    label='two_axis→mat (round-trip)')
    _assert_all_close(aa_re,  aa_in,  label='mat→aa (round-trip)')

    # Positional delta
    pos       = torch.tensor([[0.0], [0.1], [0.3]])
    pos_delta = compute_positional_delta(pos)
    _assert_all_close(pos_delta, torch.tensor([[0.0], [0.1], [0.2]]), label='pos_delta')

    # Rotational delta
    rots = torch.cat([
        angle_axis_to_matrix(torch.tensor([v, 0.0, 0.0]))[None]
        for v in (0.0, 0.1, 0.3)
    ])
    rot_delta    = compute_rotation_delta(rots)
    rot_delta_aa = matrix_to_angle_axis(rot_delta, warn=False)
    _assert_all_close(rot_delta_aa,
                      torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
                      label='rot_delta')

    print('All basic tests passed!')

    if test_egobody:
        import tqdm
        import egobody
        dataset = egobody.EgoBodyDataset(egobody.get_recording_names())
        print('Running egobody tests …')
        for item in tqdm.tqdm(dataset):
            go   = item['global_orient']
            bp   = item['body_pose']
            T    = bp.shape[0]
            rots = torch.cat((go[:, None], bp.view(T, -1, 3, 3)), dim=1)
            rots_6d = matrix_to_two_axis(rots)
            rots_aa = matrix_to_angle_axis(rots, warn=False)
            _assert_all_close(rots, angle_axis_to_matrix(rots_aa), atol=3e-3, label='egobody aa rt')
            _assert_all_close(rots, two_axis_to_matrix(rots_6d),              label='egobody 6d rt')
        print('Egobody tests passed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_egobody', action=argparse.BooleanOptionalAction, default=False,
                        help='Also run (slow) tests on the EgoBody dataset.')
    _run_tests(test_egobody=parser.parse_args().test_egobody)
