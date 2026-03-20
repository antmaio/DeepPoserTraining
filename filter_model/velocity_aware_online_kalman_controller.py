import torch

class VelocityAwareOnlineBatchedKalmanController:
    def __init__(self, dt, njoints, threshold:float=0.0, process_noise:float=1e-2, measurement_noise:float=1e-1):
        self.delta_time = dt
        self.njoints = njoints
        self.threshold = threshold
        
        # Initial state: [x, y, z, vx, vy, vz] for each joint
        self.state = torch.zeros(njoints, 6, dtype=torch.float32)

        self.is_initialized = False
        
        # Track time since last valid measurement for each joint
        self.time_since_valid = torch.zeros(njoints, dtype=torch.float32)
        self.last_valid_dt = dt * torch.ones(njoints, dtype=torch.float32)
        
        # Transition matrix (position + velocity)
        self.base_transition = torch.eye(6, dtype=torch.float32)
        for i in range(3):
            self.base_transition[i, i + 3] = dt
        
        # Store individual transition matrices for each joint
        self.transition_matrices = self.base_transition.unsqueeze(0).repeat(njoints, 1, 1)
        
        # Observation models
        self.H_position_only = torch.zeros(3, 6, dtype=torch.float32)  # Only observe position
        for i in range(3):
            self.H_position_only[i, i] = 1
            
        self.H_position_velocity = torch.eye(6, dtype=torch.float32)  # Observe both position and velocity
        
        # Covariances
        self.P = torch.eye(6, dtype=torch.float32)
        self.Q = torch.eye(6, dtype=torch.float32) * process_noise
        self.R_position_only = torch.eye(3, dtype=torch.float32) * measurement_noise
        self.R_position_velocity = torch.eye(6, dtype=torch.float32) * measurement_noise
        
        # For batched operations
        self.P_batched = None
    
    def init_with_velocities(self, initial_positions, initial_velocities, confidence_scores=None):
        """
        Initialize with known positions AND velocities
        """
        if confidence_scores is not None:
            valid_joints = torch.all(confidence_scores > self.threshold, dim=-1)
            self.time_since_valid[~valid_joints] = 100.0
        else:
            valid_joints = torch.ones(self.njoints, dtype=torch.bool)
        
        # Set both positions and velocities
        self.state[:, 0:3] = initial_positions  # positions
        self.state[:, 3:6] = initial_velocities  # velocities

        # Initialize batched covariance matrix
        self.P_batched = self.P.unsqueeze(0).repeat(self.njoints, 1, 1)
        
        # Set lower initial uncertainty for velocities since we measured them
        for i in range(self.njoints):
            if valid_joints[i]:
                self.P_batched[i, 3:, 3:] = torch.eye(3) * 1e-3  # Low velocity uncertainty
        
        self.is_initialized = True
    
    def _update_transition_matrices(self):
        """Update transition matrices based on time since last valid measurement"""
        for j in range(self.njoints):
            effective_dt = self.last_valid_dt[j] + self.time_since_valid[j]
            
            # Create transition matrix with effective dt
            F = torch.eye(6, dtype=torch.float32)
            for i in range(3):
                F[i, i + 3] = effective_dt
            
            self.transition_matrices[j] = F
    
    def predict(self):
        """Prediction step for all joints"""
        if not self.is_initialized:
            return
        
        # Update transition matrices based on current occlusion states
        self._update_transition_matrices()
        
        # Update state using joint-specific transition matrices
        for j in range(self.njoints):
            self.state[j] = torch.matmul(self.transition_matrices[j], self.state[j])
        
        # Update covariance: P = F @ P @ F.T + Q
        for j in range(self.njoints):
            F = self.transition_matrices[j]
            FP = torch.matmul(F, self.P_batched[j])
            self.P_batched[j] = torch.matmul(FP, F.T) + self.Q
    
    def update_position_only(self, measured_positions, confidence_scores=None):
        """
        Standard update with only position measurements (original behavior)
        """
        return self._update(measured_positions, None, confidence_scores, self.H_position_only, self.R_position_only)
    
    def update_with_velocities(self, measured_positions, measured_velocities, confidence_scores=None):
        """
        Update with both position and velocity measurements
        """
        # Combine positions and velocities into full state measurement
        measured_full_state = torch.cat([measured_positions, measured_velocities], dim=1)
        return self._update(measured_full_state, measured_velocities, confidence_scores, self.H_position_velocity, self.R_position_velocity)
    
    def _update(self, measurements, measured_velocities, confidence_scores, H, R):
        """
        Generic update step that handles both position-only and full-state updates
        """
        if not self.is_initialized:
            # Auto-initialize with positions and zero velocities if only positions available
            if measured_velocities is None:
                self.init(measurements, confidence_scores)
            else:
                self.init_with_velocities(measurements[:, :3], measured_velocities, confidence_scores)
            return self.get_current_positions()
        
        # Determine which joints have valid measurements
        if confidence_scores is not None:
            valid_measurements = torch.all(confidence_scores > self.threshold, dim=-1)
        else:
            valid_measurements = torch.ones(self.njoints, dtype=torch.bool)
        
        # Update time tracking
        self.time_since_valid += self.delta_time
        self.time_since_valid[valid_measurements] = 0.0
        self.last_valid_dt[valid_measurements] = self.delta_time
        
        # Only update joints with valid measurements
        valid_indices = torch.where(valid_measurements)[0]
        
        if len(valid_indices) > 0:
            # Extract valid subsets
            valid_state = self.state[valid_indices]
            valid_P = self.P_batched[valid_indices]
            valid_measurements_data = measurements[valid_indices]
            
            # Kalman update math
            H_batched = H.unsqueeze(0).repeat(len(valid_indices), 1, 1)
            H_state = torch.bmm(H_batched, valid_state.unsqueeze(-1)).squeeze(-1)
            y = valid_measurements_data - H_state
            
            # Innovation covariance: S = H @ P @ H.T + R
            HP = torch.bmm(H_batched, valid_P)
            S = torch.bmm(HP, H_batched.transpose(1, 2)) + \
                R.unsqueeze(0).repeat(len(valid_indices), 1, 1)
            
            # Kalman gain: K = P @ H.T @ S^-1
            PHt = torch.bmm(valid_P, H_batched.transpose(1, 2))
            S_inv = torch.linalg.inv(S)
            K = torch.bmm(PHt, S_inv)
            
            # Update state
            Ky = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
            valid_state_updated = valid_state + Ky
            
            # Update covariance
            I = torch.eye(6, dtype=torch.float32).unsqueeze(0).repeat(len(valid_indices), 1, 1)
            KH = torch.bmm(K, H_batched)
            valid_P_updated = torch.bmm(I - KH, valid_P)
            
            # Update main arrays
            self.state[valid_indices] = valid_state_updated
            self.P_batched[valid_indices] = valid_P_updated
        
        return self.get_current_positions()
    
    def init(self, initial_positions, confidence_scores=None):
        """Backward compatibility - initialize with positions only (zero velocities)"""
        zero_velocities = torch.zeros_like(initial_positions)
        self.init_with_velocities(initial_positions, zero_velocities, confidence_scores)
    
    def get_current_positions(self):
        """Get current position estimates for all joints"""
        if not self.is_initialized:
            return torch.zeros(self.njoints, 3, dtype=torch.float32)
        return self.state[:, :3].clone()
    
    def get_current_velocities(self):
        """Get current velocity estimates for all joints"""
        if not self.is_initialized:
            return torch.zeros(self.njoints, 3, dtype=torch.float32)
        return self.state[:, 3:6].clone()
    
    def get_full_state(self):
        """Get both positions and velocities"""
        if not self.is_initialized:
            return torch.zeros(self.njoints, 6, dtype=torch.float32)
        return self.state.clone()

class VelocityAwareOnlineKalmanController(VelocityAwareOnlineBatchedKalmanController):
    def __init__(self, dt, njoints, process_noise=1e-2, measurement_noise=1e-1):
        super().__init__(dt, njoints, 0.0, process_noise, measurement_noise)
