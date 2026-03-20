import numpy as np
import torch 

"""
def zero_if_confidences_are_null(points3d: np.ndarray, conf: np.ndarray) -> np.ndarray:
    '''
    points3d: (nframes, njoints, 3)
    conf:     (nframes, njoints, ncam)
    
    Put 0 to 3D points value if conf is 0.0 in at least 2 cameras (for 3 cameras total)
    '''
    _, _, ncam = conf.shape
    
    # Count how many cameras have confidence = 0.0 for each joint in each frame
    zero_conf_count = (conf == 0.0).sum(axis=2)  # (nframes, njoints)
    
    # Create mask: True where confidence is 0.0 in at least 2 cameras
    mask = zero_conf_count >= (ncam-1)  # (nframes, njoints)
    
    # Expand mask to match points3d dimensions (nframes, njoints, 3)
    mask_expanded = np.expand_dims(mask, axis=-1)  # (nframes, njoints, 1)
    mask_expanded = np.broadcast_to(mask_expanded, points3d.shape)  # (nframes, njoints, 3)
    
    # Set points to zero where mask is True
    points3d_zeroed = points3d.copy()
    points3d_zeroed[mask_expanded] = 0.0
    
    return points3d_zeroed
def zero_if_confidences_are_below_threshold(points3d: np.ndarray, conf: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    '''
    points3d: (nframes, njoints, 3)
    conf:     (nframes, njoints, ncam)
    
    Set 3D point to 0 if confidence < threshold in at least 2 cameras (for 3 cameras total)
    '''
    _, _, ncam = conf.shape
    
    # Count how many cameras have confidence < threshold for each joint in each frame
    low_conf_count = (conf < threshold).sum(axis=2)  # (nframes, njoints)
    
    # Mask where confidence is low in at least 2 cameras
    mask = low_conf_count >= (ncam - 1)  # (nframes, njoints)
    
    # Expand mask to match points3d shape
    mask_expanded = np.expand_dims(mask, axis=-1)  # (nframes, njoints, 1)
    mask_expanded = np.broadcast_to(mask_expanded, points3d.shape)
    
    # Zero out points where mask is True
    points3d_zeroed = points3d.copy()
    points3d_zeroed[mask_expanded] = 0.0
    
    return points3d_zeroed
"""

def zero_if_confidences_are_null(points3d, conf):
    """
    points3d: (nframes, njoints, 3), np.ndarray or torch.Tensor
    conf:     (nframes, njoints, ncam), np.ndarray or torch.Tensor

    Set 3D points to 0 if confidence == 0 in at least ncam-1 cameras.
    Preserves input type (numpy or torch.Tensor).
    """
    is_tensor = torch.is_tensor(points3d)

    if is_tensor:
        ncam = conf.shape[2]
        zero_conf_count = (conf == 0.0).sum(dim=2)  # (nframes, njoints)
        mask = zero_conf_count >= (ncam - 1)        # (nframes, njoints)
        mask_expanded = mask.unsqueeze(-1).expand_as(points3d)
        points3d_zeroed = points3d.clone()
        points3d_zeroed[mask_expanded] = 0.0
    else:
        ncam = conf.shape[2]
        zero_conf_count = (conf == 0.0).sum(axis=2)
        mask = zero_conf_count >= (ncam - 1)
        mask_expanded = np.expand_dims(mask, axis=-1)
        mask_expanded = np.broadcast_to(mask_expanded, points3d.shape)
        points3d_zeroed = points3d.copy()
        points3d_zeroed[mask_expanded] = 0.0

    return points3d_zeroed
def zero_if_confidences_are_below_threshold(points3d, conf, threshold: float = 0.5):
    """
    points3d: (nframes, njoints, 3), np.ndarray or torch.Tensor
    conf:     (nframes, njoints, ncam), np.ndarray or torch.Tensor
    threshold: float, confidence threshold below which points are zeroed
    
    Set 3D point to 0 if confidence < threshold in at least ncam-1 cameras.
    Preserves input type (numpy or torch.Tensor).
    """
    is_tensor = torch.is_tensor(points3d)

    if is_tensor:
        ncam = conf.shape[2]
        low_conf_count = (conf < threshold).sum(dim=2)  # (nframes, njoints)
        mask = low_conf_count >= (ncam - 1)  # (nframes, njoints)
        mask_expanded = mask.unsqueeze(-1).expand_as(points3d)
        points3d_zeroed = points3d.clone()
        points3d_zeroed[mask_expanded] = 0.0
    else:
        ncam = conf.shape[2]
        low_conf_count = (conf < threshold).sum(axis=2)
        mask = low_conf_count >= (ncam - 1)
        mask_expanded = np.expand_dims(mask, axis=-1)
        mask_expanded = np.broadcast_to(mask_expanded, points3d.shape)
        points3d_zeroed = points3d.copy()
        points3d_zeroed[mask_expanded] = 0.0

    return points3d_zeroed