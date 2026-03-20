import torch

class StudentsTFilter:
    def __init__(self, dt=1.0, joint_params=None, active_joints=None, dof=3.0):
        """
        Student's t Filter for 3D Joint Tracking (Constant Velocity Model).
        """
        self.dt = dt
        self.nu = dof # Degrees of freedom (keep low, e.g., 3, for heavy tails)
        self.dy = 3   # Measurement dimension (x, y, z)

        # Transition matrix F (6x6: 3 pos, 3 vel)
        self.F = torch.eye(6)
        self.F[0:3, 3:6] = torch.eye(3) * dt

        # Observation matrix H (3x6: extracts position)
        self.H = torch.zeros((3, 6))
        self.H[0:3, 0:3] = torch.eye(3)

        self.active_joints = active_joints if active_joints is not None else list(range(len(joint_params)))
        self.n_active = len(self.active_joints)
        
        self.Q = torch.zeros((self.n_active, 6, 6))
        self.R = torch.zeros((self.n_active, 3, 3))
        
        for idx, j_idx in enumerate(self.active_joints):
            pn = joint_params[j_idx]['process_noise']
            mn = joint_params[j_idx]['measurement_noise']
            self.Q[idx] = torch.eye(6) * (pn ** 2)
            self.R[idx] = torch.eye(3) * (mn ** 2)

    def filter_joints(self, observations):
        """
        Filters the joint sequences for active joints.
        Expected input shape: (batch_size, nframes, njoints, 3) or (nframes, njoints, 3) 
        where njoints is the total number of joints.
        Returns filtered positions with the same shape, with only active joints modified.
        """
        device = observations.device
        dtype = observations.dtype

        has_batch = len(observations.shape) == 4
        if has_batch:
            b, f, j_total, d = observations.shape
        else:
            f, j_total, d = observations.shape
            b = 1
            observations = observations.unsqueeze(0)

        N = b * self.n_active 
        
        # Extract only active joints
        y_active = observations[:, :, self.active_joints, :]
        
        # Reshape to (nframes, N, 3) to process all independent sequences simultaneously
        # y_active shape: (b, f, n_active, 3)
        # permute to (f, b, n_active, 3) -> reshape to (f, N, 3)
        y = y_active.permute(1, 0, 2, 3).reshape(f, N, 3)

        # Move static matrices to device/dtype
        F = self.F.to(device=device, dtype=dtype)
        H = self.H.to(device=device, dtype=dtype)
        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)

        # Initialize State (N, 6) and Covariance (N, 6, 6)
        x_est = torch.zeros((N, 6), device=device, dtype=dtype)
        x_est[:, 0:3] = y[0]  # Initialize position with the first frame's measurement
        P_est = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)

        # Expand Q and R to match N (tile over batch size)
        Q_N = Q.repeat(b, 1, 1)
        R_N = R.repeat(b, 1, 1)

        filtered_active_positions = torch.zeros((f, N, 3), device=device, dtype=dtype)
        
        I = torch.eye(6, device=device, dtype=dtype)

        for k in range(f):
            y_k = y[k] # Current measurements: (N, 3)

            # --- 1. TIME UPDATE (PREDICT) ---
            # x_pred = F * x_est
            x_pred = (F @ x_est.unsqueeze(-1)).squeeze(-1)
            
            # P_pred = F * P_est * F^T + Q
            P_pred = F @ P_est @ F.T + Q_N

            # --- 2. MEASUREMENT UPDATE (CORRECT) ---
            # Innovation (residual): y_res = y_k - H * x_pred
            y_res = y_k - (H @ x_pred.unsqueeze(-1)).squeeze(-1)

            # Innovation covariance: S = H * P_pred * H^T + R
            S = H @ P_pred @ H.T + R_N
            S_inv = torch.linalg.inv(S)

            # Kalman Gain: K = P_pred * H^T * S_inv
            K = P_pred @ H.T @ S_inv

            # Update state estimate: x_est = x_pred + K * y_res
            x_est = x_pred + (K @ y_res.unsqueeze(-1)).squeeze(-1)

            # Calculate the squared Mahalanobis distance (Delta^2) for the scaling factor
            delta_sq = (y_res.unsqueeze(-2) @ S_inv @ y_res.unsqueeze(-1)).squeeze(-1).squeeze(-1)

            # Scale factor based on the Student's t distribution properties
            scale = (self.nu + delta_sq) / (self.nu + self.dy)

            # Update covariance estimate: P_est = scale * (P_pred - K * H * P_pred)
            P_unscaled = (I - K @ H) @ P_pred
            P_est = scale.unsqueeze(-1).unsqueeze(-1) * P_unscaled

            # Store only the position components
            filtered_active_positions[k] = x_est[:, 0:3]

        # Reshape back to (f, b, n_active, 3) and permute to (b, f, n_active, 3)
        filtered_active_positions = filtered_active_positions.reshape(f, b, self.n_active, 3).permute(1, 0, 2, 3)
        
        # Put back into full joint structure
        filtered_positions = observations.clone()
        filtered_positions[:, :, self.active_joints, :] = filtered_active_positions

        if not has_batch:
            filtered_positions = filtered_positions.squeeze(0)

        return filtered_positions

# --- Usage Example ---
if __name__ == "__main__":
    batch_size, nframes, njoints, dims = 2, 100, 17, 3
    
    # Simulate some noisy joint data
    noisy_joints = torch.randn(batch_size, nframes, njoints, dims) * 5.0
    
    # Mock joint params
    joint_params = [{'process_noise': 0.01, 'measurement_noise': 0.1} for _ in range(njoints)]
    active_joints = list(range(5, 17))
    
    # Initialize and run filter
    tracker = StudentsTFilter(dt=1.0, joint_params=joint_params, active_joints=active_joints, dof=3.0)
    
    # Optionally test with simulated GPU data
    if torch.cuda.is_available():
        noisy_joints = noisy_joints.cuda()

    smoothed_joints = tracker.filter_joints(noisy_joints)
    
    print(f"Input shape: {noisy_joints.shape}")
    print(f"Output shape: {smoothed_joints.shape}")
    print(f"Output device: {smoothed_joints.device}")