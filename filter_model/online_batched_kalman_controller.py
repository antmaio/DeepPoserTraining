import torch

class OnlineBatchedKalmanController:
    def __init__(self, dt, njoints, process_noise=1e-2, measurement_noise=1e-1):
        self.delta_time = dt
        self.njoints = njoints
        
        # Initial state: [x, y, z, vx, vy, vz] for each joint
        # Shape: (njoints, 6)
        self.state = torch.zeros(njoints, 6, dtype=torch.float32)
        self.is_initialized = False
        
        # Transition matrix (position + velocity) - same for all joints
        self.transition = torch.eye(6, dtype=torch.float32)
        for i in range(3):
            self.transition[i, i + 3] = dt
        
        # Observation model: only positions - same for all joints
        self.H = torch.zeros(3, 6, dtype=torch.float32)
        for i in range(3):
            self.H[i, i] = 1
        
        # Covariances - same for all joints
        self.P = torch.eye(6, dtype=torch.float32)  # Single joint covariance
        self.Q = torch.eye(6, dtype=torch.float32) * process_noise
        self.R = torch.eye(3, dtype=torch.float32) * measurement_noise
        
        # For batched operations, we'll expand to (njoints, 6, 6) when needed
        self.P_batched = None
    
    def init(self, initial_positions):
        """
        Initialize the Kalman filter with initial positions for all joints
        
        Args:
            initial_positions: torch.Tensor of shape (njoints, 3) [x, y, z] for each joint
        """
        self.state[:, 0] = initial_positions[:, 0]  # x positions
        self.state[:, 1] = initial_positions[:, 1]  # y positions
        self.state[:, 2] = initial_positions[:, 2]  # z positions
        # velocities remain 0 initially
        
        # Initialize batched covariance matrix
        self.P_batched = self.P.unsqueeze(0).repeat(self.njoints, 1, 1)
        
        self.is_initialized = True
    
    def predict(self):
        """Prediction step for all joints"""
        if not self.is_initialized:
            raise RuntimeError("Kalman filter not initialized. Call init() first.")
        
        # Update state: (njoints, 6) = (njoints, 6, 6) @ (njoints, 6, 1) -> (njoints, 6)
        self.state = torch.bmm(self.transition.unsqueeze(0).repeat(self.njoints, 1, 1), 
                              self.state.unsqueeze(-1)).squeeze(-1)
        
        # Update covariance: P = F @ P @ F.T + Q
        # (njoints, 6, 6) = (njoints, 6, 6) @ (njoints, 6, 6) @ (njoints, 6, 6) + (njoints, 6, 6)
        FP = torch.bmm(self.transition.unsqueeze(0).repeat(self.njoints, 1, 1), self.P_batched)
        self.P_batched = torch.bmm(FP, self.transition.unsqueeze(0).repeat(self.njoints, 1, 1).transpose(1, 2)) + \
                        self.Q.unsqueeze(0).repeat(self.njoints, 1, 1)
    
    def update(self, measured_positions):
        """
        Update step with new measurements for all joints
        
        Args:
            measured_positions: torch.Tensor of shape (njoints, 3) [x, y, z] for each joint
            
        Returns:
            torch.Tensor: Updated position estimates of shape (njoints, 3) [x, y, z]
        """
        if not self.is_initialized:
            # Initialize with first measurement
            self.init(measured_positions)
            return measured_positions.clone()
        
        # Innovation: y = z - H @ state
        # (njoints, 3) = (njoints, 3) - (njoints, 3, 6) @ (njoints, 6, 1) -> (njoints, 3)
        H_batched = self.H.unsqueeze(0).repeat(self.njoints, 1, 1)
        H_state = torch.bmm(H_batched, self.state.unsqueeze(-1)).squeeze(-1)
        y = measured_positions - H_state
        
        # Innovation covariance: S = H @ P @ H.T + R
        # (njoints, 3, 3) = (njoints, 3, 6) @ (njoints, 6, 6) @ (njoints, 6, 3) + (njoints, 3, 3)
        HP = torch.bmm(H_batched, self.P_batched)
        S = torch.bmm(HP, H_batched.transpose(1, 2)) + \
            self.R.unsqueeze(0).repeat(self.njoints, 1, 1)
        
        # Kalman gain: K = P @ H.T @ S^-1
        # (njoints, 6, 3) = (njoints, 6, 6) @ (njoints, 6, 3) @ (njoints, 3, 3)
        PHt = torch.bmm(self.P_batched, H_batched.transpose(1, 2))
        S_inv = torch.linalg.inv(S)
        K = torch.bmm(PHt, S_inv)
        
        # Update state: state = state + K @ y
        # (njoints, 6) = (njoints, 6) + (njoints, 6, 3) @ (njoints, 3, 1) -> (njoints, 6)
        Ky = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
        self.state = self.state + Ky
        
        # Update covariance: P = (I - K @ H) @ P
        # (njoints, 6, 6) = (njoints, 6, 6) - (njoints, 6, 3) @ (njoints, 3, 6)) @ (njoints, 6, 6)
        I = torch.eye(6, dtype=torch.float32).unsqueeze(0).repeat(self.njoints, 1, 1)
        KH = torch.bmm(K, H_batched)
        self.P_batched = torch.bmm(I - KH, self.P_batched)
        
        return self.state[:, :3].clone()
    
    def get_current_positions(self):
        """Get current position estimates for all joints"""
        if not self.is_initialized:
            raise RuntimeError("Kalman filter not initialized. Call init() or update() first.")
        return self.state[:, :3].clone()
