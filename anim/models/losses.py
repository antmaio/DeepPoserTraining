"""
Loss functions for HMDPoserExtHeadCentered.
"""
# External
import torch
import torch.nn as nn

# Internal
from data.data_config import SmplxJoints
from utils.utils_transform import matrix_to_two_axis, rotational_fk


# ---------------------------------------------------------------------------
# Standalone loss functions
# ---------------------------------------------------------------------------

def two_axis_orthonormality_loss(
    two_axis: torch.Tensor,
    epsilon: float = 1e-8,
    det_weight: float = 1.0,
) -> torch.Tensor:
    """
    Orthonormality loss for 6-D rotation representations.

    Penalises deviations from unit-norm axes, non-orthogonality,
    and (via the cross-product norm) improper rotations.

    Args:
        two_axis:   (..., 6) tensor of two-axis rotation representations.
        epsilon:    Small value for safe normalisation.
        det_weight: Weight applied to the determinant component.

    Returns:
        Scalar mean loss.
    """
    assert two_axis.shape[-1] == 6

    axis1, axis2 = two_axis[..., :3], two_axis[..., 3:]

    norm1 = torch.norm(axis1, dim=-1)
    norm2 = torch.norm(axis2, dim=-1)

    axis1_n = axis1 / (norm1.unsqueeze(-1) + epsilon)
    axis2_n = axis2 / (norm2.unsqueeze(-1) + epsilon)

    norm_loss   = torch.square(norm1 - 1.0) + torch.square(norm2 - 1.0)
    ortho_loss  = torch.square(torch.sum(axis1_n * axis2_n, dim=-1))
    cross       = torch.linalg.cross(axis1_n, axis2_n, dim=-1)
    det_loss    = torch.square(torch.sum(cross * cross, dim=-1) - 1.0)

    return (norm_loss + ortho_loss + det_weight * det_loss).mean()


# ---------------------------------------------------------------------------
# Loss configuration dataclass
# ---------------------------------------------------------------------------

class LossWeights:
    """Plain container for all named loss weights."""

    def __init__(
        self,
        loss_func: str                          = 'l1',
        global_orient_loss_weight: float        = 1.0,
        body_pose_loss_weight: float            = 5.0,
        body_pose_global_loss_weight: float     = 1.0,
        joints_loss_weight: float               = 1.0,
        smooth_loss_weight: float               = 0.5,
        shape_loss_weight: float                = 0.1,
        extra_shape_loss_weight: float          = 0.0,
        orth_loss: float                        = 0.1,
        jitter_loss_weight: float               = 0.0,
        extra_hand_pose_loss_weight: float      = 0.0,
        extra_hand_pose_global_loss_weight: float = 0.0,
        extra_hand_joints_loss_weight: float    = 0.0,
        orthonormality_loss_weight: float       = 1.0,
    ):
        if loss_func == 'l1':
            self.func = nn.functional.l1_loss
        elif loss_func == 'mse':
            self.func = nn.functional.mse_loss
        else:
            raise NotImplementedError(f"loss_func '{loss_func}' not supported (use 'l1' or 'mse').")

        self.global_orient               = global_orient_loss_weight
        self.body_pose                   = body_pose_loss_weight
        self.body_pose_global            = body_pose_global_loss_weight
        self.joints                      = joints_loss_weight
        self.smooth                      = smooth_loss_weight
        self.shape                       = shape_loss_weight
        self.extra_shape                 = extra_shape_loss_weight
        self.orth                        = orth_loss
        self.jitter                      = jitter_loss_weight
        self.extra_hand_pose             = extra_hand_pose_loss_weight
        self.extra_hand_pose_global      = extra_hand_pose_global_loss_weight
        self.extra_hand_joints           = extra_hand_joints_loss_weight
        self.orthonormality              = orthonormality_loss_weight


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

_WRIST_INDICES     = [SmplxJoints.LEFT_WRIST - 1, SmplxJoints.RIGHT_WRIST - 1]
_WRIST_JT_INDICES  = [SmplxJoints.LEFT_WRIST,     SmplxJoints.RIGHT_WRIST    ]


def compute_losses(
    w: LossWeights,
    model_output,
    model_target,
    global_orient_6d_pred: torch.Tensor,
    body_pose_6d_pred: torch.Tensor,
) -> dict:
    """
    Compute all training losses and return a dict of named scalar tensors.

    Args:
        w:                    LossWeights instance.
        model_output:         BaseModelOutput from forward().
        model_target:         Ground-truth BaseModelOutput.
        global_orient_6d_pred: Raw 6-D global orientation prediction (pre-matrix conversion).
        body_pose_6d_pred:     Raw 6-D body pose prediction (pre-matrix conversion).

    Returns:
        loss_dict: Dict mapping loss name → scalar tensor; includes 'loss' (total).
    """
    loss_dict = {}
    f = w.func

    # ---- orientation (6-D space) ----
    global_orient_6d_gt  = matrix_to_two_axis(model_target.global_orient)
    global_orient_6d_loss = f(global_orient_6d_pred, global_orient_6d_gt)
    loss = w.global_orient * global_orient_6d_loss
    loss_dict['global_orient_6d_loss'] = global_orient_6d_loss

    # ---- local body pose (6-D space) ----
    body_pose_6d_gt  = matrix_to_two_axis(model_target.body_pose)
    body_pose_6d_loss = f(body_pose_6d_pred, body_pose_6d_gt)
    loss += w.body_pose * body_pose_6d_loss
    loss_dict['body_pose_6d_loss'] = body_pose_6d_loss

    if w.extra_hand_pose > 0.0:
        extra_hand_pose_loss = f(
            model_output.body_pose[:, :, _WRIST_INDICES],
            model_target.body_pose[:, :, _WRIST_INDICES],
        )
        loss += w.extra_hand_pose * extra_hand_pose_loss
        loss_dict['extra_hand_pose_loss'] = extra_hand_pose_loss

    # ---- global body pose (FK space) ----
    with torch.no_grad():
        body_pose_global_gt   = rotational_fk(model_target.global_orient, model_target.body_pose)
    body_pose_global_pred = rotational_fk(model_output.global_orient, model_output.body_pose)
    body_pose_global_loss  = f(body_pose_global_pred, body_pose_global_gt)
    loss += w.body_pose_global * body_pose_global_loss
    loss_dict['body_pose_global_3x3_loss'] = body_pose_global_loss

    if w.extra_hand_pose_global > 0.0:
        extra_hand_pose_global_loss = f(
            body_pose_global_pred[:, :, _WRIST_INDICES],
            body_pose_global_gt[:, :,   _WRIST_INDICES],
        )
        loss += w.extra_hand_pose_global * extra_hand_pose_global_loss
        loss_dict['extra_hand_pose_global_loss'] = extra_hand_pose_global_loss

    # ---- joints ----
    joints_pred   = model_output.joints
    joints_target = model_target.joints
    joints_loss   = f(joints_pred, joints_target)
    loss += w.joints * joints_loss
    loss_dict['joints_loss'] = joints_loss

    if w.extra_hand_joints > 0.0:
        extra_hand_joints_loss = f(
            joints_pred[:, :, _WRIST_JT_INDICES],
            joints_target[:, :, _WRIST_JT_INDICES],
        )
        loss += w.extra_hand_joints * extra_hand_joints_loss
        loss_dict['extra_hand_joints_loss'] = extra_hand_joints_loss

    # ---- smoothness (second-order finite difference) ----
    accel_pred   = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1]   + joints_pred[:, 2:]
    accel_target = joints_target[:, :-2] - 2 * joints_target[:, 1:-1] + joints_target[:, 2:]
    smooth_loss  = f(accel_pred, accel_target)
    loss += w.smooth * smooth_loss
    loss_dict['smooth_loss'] = smooth_loss

    # ---- jitter (third-order finite difference) ----
    if w.jitter > 0.0 and joints_pred.shape[1] > 3:
        jitter_pred   = joints_pred[:, 3:] - 3*joints_pred[:, 2:-1]   + 3*joints_pred[:, 1:-2]   - joints_pred[:, :-3]
        jitter_target = joints_target[:, 3:] - 3*joints_target[:, 2:-1] + 3*joints_target[:, 1:-2] - joints_target[:, :-3]
        jitter_loss   = f(jitter_pred, jitter_target)
        loss += w.jitter * jitter_loss
        loss_dict['jitter_loss'] = jitter_loss

    # ---- shape regularisation ----
    betas_pred = model_output.betas
    betas_temporal_mean = betas_pred.mean(dim=1, keepdim=True).expand_as(betas_pred)
    shape_loss  = f(betas_pred, betas_temporal_mean, reduction='mean')
    loss += w.shape * shape_loss
    loss_dict['shape_loss'] = shape_loss

    if w.extra_shape > 0.0:
        extra_shape_loss = f(betas_pred, torch.zeros_like(betas_pred))
        loss += w.extra_shape * extra_shape_loss
        loss_dict['extra_shape_loss'] = extra_shape_loss

    # ---- orthonormality ----
    ortho_loss = (
        two_axis_orthonormality_loss(global_orient_6d_pred)
        + two_axis_orthonormality_loss(body_pose_6d_pred)
    )
    loss += w.orthonormality * ortho_loss
    loss_dict['orthonormality_loss'] = ortho_loss

    # ---- metrics (no gradient) ----
    with torch.no_grad():
        loss_dict['MPJPE(cm)'] = 100 * (
            (joints_target - joints_pred).square().sum(dim=-1).sqrt().mean()
        )

    loss_dict['loss'] = loss
    return loss_dict
