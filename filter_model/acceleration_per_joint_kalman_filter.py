import torch
from typing import Union, List, Dict

class AccelerationPerJointKalmanFilter:
    def __init__(self, 
                 dt: float = 1.0 / 30.0, 
                 joint_params: List[Dict[str, float]] = None,
                 active_joints: List[int] = None,
                 threshold: float = 0.0):
        """
        Constant Acceleration Kalman Filter for batched joint positions.
        
        Args:
            dt: Time step between frames.
            joint_params: List of dictionaries containing 'process_noise' and 'measurement_noise' for each joint.
            active_joints: Indices of joints to filter. If None, filters all joints in joint_params.
            threshold: Confidence threshold below which measurement is skipped.
        """
        self.dt = dt
        self.threshold = threshold
        
        if joint_params is None:
            raise ValueError("joint_params must be provided")
            
        self.active_joints = active_joints if active_joints is not None else list(range(len(joint_params)))
        self.n_active = len(self.active_joints)
        
        # Initialize noise matrices for active joints
        # Q: Process noise (9x9), R: Measurement noise (3x3)
        self.Q = torch.zeros((self.n_active, 9, 9))
        self.R = torch.zeros((self.n_active, 3, 3))
        
        for idx, j_idx in enumerate(self.active_joints):
            pn = joint_params[j_idx]['process_noise']
            mn = joint_params[j_idx]['measurement_noise']
            # Note: Following StudentsTFilter pattern using (noise ** 2) if desired, 
            # or just noise. Standard KF usually uses variance.
            # Here we use the direct value provided in joint_params.
            self.Q[idx] = torch.eye(9) * pn
            self.R[idx] = torch.eye(3) * mn

        # Transition matrix F: (9x9)
        # x_next = x + v*dt + 0.5*a*dt^2
        # v_next = v + a*dt
        # a_next = a
        self.F = torch.eye(9)
        for i in range(3):
            self.F[i, i + 3] = self.dt
            self.F[i, i + 6] = 0.5 * (self.dt ** 2)
            self.F[i + 3, i + 6] = self.dt
            
        # Observation matrix H: (3x9) extracts position [x, y, z]
        self.H = torch.zeros((3, 9))
        self.H[0:3, 0:3] = torch.eye(3)

    def filter_joints(self, 
                      positions: torch.Tensor, 
                      confidence_scores: torch.Tensor = None) -> torch.Tensor:
        """
        Filter a batch of sequences using a constant acceleration model.
        
        Args:
            positions: (batch_size, nframes, njoints, 3) 3D positions.
            confidence_scores: Optional, (batch_size, nframes, njoints) or 
                              (batch_size, nframes, njoints, num_cameras).
                              
        Returns:
            filtered_positions: (batch_size, nframes, njoints, 3) Filtered positions.
        """
        device = positions.device
        dtype = positions.dtype

        has_batch = len(positions.shape) == 4
        if has_batch:
            b, f, j_total, d = positions.shape
        else:
            f, j_total, d = positions.shape
            b = 1
            positions = positions.unsqueeze(0)
            if confidence_scores is not None:
                confidence_scores = confidence_scores.unsqueeze(0)

        N = b * self.n_active 
        
        # Extract active joints: (b, f, n_active, 3)
        pos_active = positions[:, :, self.active_joints, :]
        
        # Reshape to (f, b * n_active, 3) -> (f, N, 3)
        y = pos_active.permute(1, 0, 2, 3).reshape(f, N, 3)

        if confidence_scores is not None:
            # Handle possible multiple cameras by taking "all valid" or just checking shape
            if confidence_scores.dim() == 4: # (b, f, j, cameras)
                conf_active = confidence_scores[:, :, self.active_joints, :]
                valid_mask = torch.all(conf_active >= self.threshold, dim=-1)
            else: # (b, f, j)
                conf_active = confidence_scores[:, :, self.active_joints]
                valid_mask = conf_active >= self.threshold
            # Reshape mask to (f, N)
            valid_mask = valid_mask.permute(1, 0, 2).reshape(f, N)
        else:
            valid_mask = torch.ones((f, N), dtype=torch.bool, device=device)

        # Move static matrices to device/dtype
        F = self.F.to(device=device, dtype=dtype)
        H = self.H.to(device=device, dtype=dtype)
        HT = H.T
        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)
        I = torch.eye(9, device=device, dtype=dtype)

        # Expand Q and R to match N (tile over batch size)
        Q_N = Q.repeat(b, 1, 1)
        R_N = R.repeat(b, 1, 1)

        # Initialize State (N, 9) and Covariance (N, 9, 9)
        state = torch.zeros((N, 9), device=device, dtype=dtype)
        state[:, 0:3] = y[0]  # Initialize position with the first frame
        P = torch.eye(9, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)

        filtered_active_positions = torch.zeros((f, N, 3), device=device, dtype=dtype)
        filtered_active_positions[0] = state[:, 0:3]

        # Main Filtering Loop
        for t in range(1, f):
            y_t = y[t]              # (N, 3)
            mask_t = valid_mask[t]  # (N,)
            
            # --- 1. Predict Step ---
            # x = F @ x
            state = (F @ state.unsqueeze(-1)).squeeze(-1)
            # P = F @ P @ F.T + Q
            P = F @ P @ F.T + Q_N
            
            # --- 2. Update Step (only if measurement is valid) ---
            valid_indices = torch.where(mask_t)[0]
            
            if len(valid_indices) > 0:
                s_v = state[valid_indices]
                P_v = P[valid_indices]
                y_v = y_t[valid_indices]
                
                Q_v = Q_N[valid_indices]
                R_v = R_N[valid_indices]
                
                # Innovation y_res = z - Hx
                y_res = y_v - (H @ s_v.unsqueeze(-1)).squeeze(-1)
                
                # Innovation covariance S = HPH' + R
                S = H @ P_v @ HT + R_v
                
                # Kalman Gain K = PH'S^-1
                K = P_v @ HT @ torch.linalg.inv(S)
                
                # Update state: x = x + Ky
                state[valid_indices] = s_v + (K @ y_res.unsqueeze(-1)).squeeze(-1)
                
                # Update covariance: P = (I - KH)P
                P[valid_indices] = (I - K @ H) @ P_v
            
            # Store the filtered positions
            filtered_active_positions[t] = state[:, 0:3]

        # Reshape back to (b, f, n_active, 3)
        filtered_active_positions = filtered_active_positions.reshape(f, b, self.n_active, 3).permute(1, 0, 2, 3)
        
        # Put back into full joint structure
        result = positions.clone()
        result[:, :, self.active_joints, :] = filtered_active_positions

        if not has_batch:
            result = result.squeeze(0)

        return result
