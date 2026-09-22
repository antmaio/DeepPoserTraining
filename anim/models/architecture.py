"""
Neural-network architecture components for HMDPoserExtHeadCentered.
"""
# External
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm  # pytorch 2.0.1

# Internal
from data.data_config import SmplxJoints


# ---------------------------------------------------------------------------
# HMD embedding block
# ---------------------------------------------------------------------------

def build_hmd_embedding(hidden_size: int) -> nn.ModuleList:
    """
    Four-part embedding for a single HMD channel (rot, rot_vel, pos, pos_vel).
    Each part projects its input to hidden_size // 4.

    Returns:
        nn.ModuleList of four nn.Sequential modules.
    """
    quarter = hidden_size // 4
    return nn.ModuleList([
        nn.Sequential(nn.Linear(6, quarter), nn.LeakyReLU()),   # rot_6d
        nn.Sequential(nn.Linear(6, quarter), nn.LeakyReLU()),   # rot_delta_6d
        nn.Sequential(nn.Linear(3, quarter), nn.LeakyReLU()),   # pos
        nn.Sequential(nn.Linear(3, quarter), nn.LeakyReLU()),   # pos_delta
    ])


# ---------------------------------------------------------------------------
# Joint (HMR) embedding block
# ---------------------------------------------------------------------------

def build_joint_embedding(
    hidden_size: int,
    use_velocities: bool,
    use_body_pose: bool,
) -> nn.ModuleList:
    """
    Variable-part embedding for a single HMR joint.

    Included features (in order):
        pos        always
        pos_vel    if use_velocities
        rot        if use_body_pose
        rot_vel    if use_body_pose and use_velocities

    Each part gets an equal share of hidden_size.

    Returns:
        nn.ModuleList of 1–4 nn.Sequential modules.
    """
    num_parts = 1
    if use_velocities:
        num_parts += 1
    if use_body_pose:
        num_parts += 1
        if use_velocities:
            num_parts += 1

    share = hidden_size // num_parts
    parts = [nn.Sequential(nn.Linear(3, share), nn.LeakyReLU())]   # pos
    if use_velocities:
        parts.append(nn.Sequential(nn.Linear(3, share), nn.LeakyReLU()))    # pos_vel
    if use_body_pose:
        parts.append(nn.Sequential(nn.Linear(6, share), nn.LeakyReLU()))    # rot_6d
        if use_velocities:
            parts.append(nn.Sequential(nn.Linear(6, share), nn.LeakyReLU()))  # rot_delta_6d
    return nn.ModuleList(parts)


# ---------------------------------------------------------------------------
# Temporal encoder
# ---------------------------------------------------------------------------

def build_temporal_encoder(
    hidden_size: int,
    num_channels: int,
    num_blocks: int,
    rnn_type: str,
    num_rnn_layers: int,
) -> nn.ModuleList:
    """
    num_blocks stacked layers of per-channel RNNs.

    Returns:
        nn.ModuleList of shape [num_blocks][num_channels].
    """
    assert rnn_type in ('lstm', 'gru')
    rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU

    encoder = nn.ModuleList([
        nn.ModuleList([
            rnn_cls(hidden_size, hidden_size, num_rnn_layers, batch_first=True)
            for _ in range(num_channels)
        ])
        for _ in range(num_blocks)
    ])

    # Weight-norm + orthogonal init (following HMD-Poser)
    for block in encoder:
        for rnn in block:
            for layer_idx in range(num_rnn_layers):
                weight_norm(rnn, f'weight_ih_l{layer_idx}')
                weight_norm(rnn, f'weight_hh_l{layer_idx}')
            for name, param in rnn.named_parameters():
                if name.startswith('weight'):
                    nn.init.orthogonal_(param)

    return encoder


# ---------------------------------------------------------------------------
# Spatial encoder
# ---------------------------------------------------------------------------

def build_spatial_encoder(
    hidden_size: int,
    num_blocks: int,
    num_heads: int,
    num_layers: int,
) -> nn.ModuleList:
    """
    num_blocks independent TransformerEncoders.

    Returns:
        nn.ModuleList of length num_blocks.
    """
    encoder_layer = nn.TransformerEncoderLayer(
        hidden_size, nhead=num_heads, batch_first=True
    )
    return nn.ModuleList([
        nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        for _ in range(num_blocks)
    ])


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

def build_pose_head(hidden_size: int, num_channels: int) -> nn.Sequential:
    """Linear → LeakyReLU → Linear projecting to NUM_JTS * 6 pose parameters."""
    return nn.Sequential(
        nn.Linear(hidden_size * num_channels, 256),
        nn.LeakyReLU(),
        nn.Linear(256, SmplxJoints.NUM_JTS * 6),
    )


def build_shape_head(hidden_size: int, num_channels: int, num_betas: int) -> nn.Sequential:
    """Linear → LeakyReLU → Linear projecting to num_betas shape parameters."""
    return nn.Sequential(
        nn.Linear(hidden_size * num_channels, 256),
        nn.LeakyReLU(),
        nn.Linear(256, num_betas),
    )
