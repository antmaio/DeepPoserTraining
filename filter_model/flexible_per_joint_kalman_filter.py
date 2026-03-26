import typing
import torch
import json
from filter_model.occlusion_handling_online_batched_kalman_controller import OcclusionHandlingOnlineBatchedKalmanController

class FlexiblePerJointKalmanFilter:
    def __init__(self, dt, njoints, joint_params=None, active_joints: typing.List[int] = None):
        self.njoints = njoints
        self.dt = dt
        self.frame_count = 0
        
        # Determine which joints are active
        if active_joints is None:
            self.active_joints = list(range(njoints))
        else:
            self.active_joints = active_joints
        
        # Default parameters for each joint
        default_params = {
            'process_noise': 1e-2,
            'measurement_noise': 5e-1
        }
        
        if joint_params is None:
            joint_params = [default_params.copy() for _ in range(njoints)]
        
        self.joint_filters = []
        self.joint_params = joint_params
        
        # Only create filters for active joints
        for j in self.active_joints:
            kf = OcclusionHandlingOnlineBatchedKalmanController(
                dt                  = dt,
                njoints             = 1,
                process_noise       = joint_params[j]['process_noise'],
                measurement_noise   = joint_params[j]['measurement_noise']
            )
            self.joint_filters.append(kf)
    
    def process_frame(self, measurements, confidence_scores=None):
        """
        Process a new frame with optional confidence scores for occlusion handling
        
        Args:
            measurements: torch.Tensor of shape (njoints, 3) - 3D positions
            confidence_scores: torch.Tensor of shape (njoints, 2) - confidence scores
        """
        self.frame_count += 1
        filtered_positions = torch.zeros_like(measurements)
        
        for idx, j in enumerate(self.active_joints):
            joint_measurement = measurements[j:j+1, :]
            
            if confidence_scores is not None:
                joint_confidence = confidence_scores[j:j+1, :]
            else:
                joint_confidence = None
            
            # Predict step (skip for first frame)
            if self.frame_count > 1:
                self.joint_filters[idx].predict()
            
            # Update step with confidence information for occlusion handling
            filtered_joint = self.joint_filters[idx].update(joint_measurement, joint_confidence)
            filtered_positions[j] = filtered_joint[0]
        
        return filtered_positions
    
    def get_current_state(self):
        """Get current state estimates for all joints"""
        states = torch.zeros(self.njoints, 3)
        for idx, j in enumerate(self.active_joints):
            states[j] = self.joint_filters[idx].get_current_positions()[0]
        return states

def set_kalman_parameters(kalman_params_path:str):
    with open(kalman_params_path, 'r') as f:
        params_config = json.load(f)

    # Create joint parameters list for the filter
    joint_params = []
    for joint_config in params_config['joint_parameters']:
        joint_params.append({
            'process_noise': joint_config['process_noise'],
            'measurement_noise': joint_config['measurement_noise']
        })

    return joint_params, params_config['default_dt'], params_config['njoints']

class OcclusionHandlingOnlineBatchedKalmanController:
    def __init__(self, dt, njoints, threshold:float=0.0, process_noise:float=1e-2, measurement_noise:float=1e-1):
        self.delta_time = dt
        self.njoints = njoints
        self.threshold = threshold # conf score threhsold, we set 3D value at 0.0 below this thresh
        
        # Initial state: [x, y, z, vx, vy, vz] for each joint
        self.state = torch.zeros(njoints, 6, dtype=torch.float32)
        self.is_initialized = False
        
        # Track time since last valid measurement for each joint
        self.time_since_valid = torch.zeros(njoints, dtype=torch.float32)
        self.last_valid_dt = dt * torch.ones(njoints, dtype=torch.float32)  # Use actual dt initially
        
        # Transition matrix (position + velocity)
        self.base_transition = torch.eye(6, dtype=torch.float32)
        for i in range(3):
            self.base_transition[i, i + 3] = dt
        
        # Store individual transition matrices for each joint
        self.transition_matrices = self.base_transition.unsqueeze(0).repeat(njoints, 1, 1)
        
        # Observation model: only positions
        self.H = torch.zeros(3, 6, dtype=torch.float32)
        for i in range(3):
            self.H[i, i] = 1
        
        # Covariances
        self.P = torch.eye(6, dtype=torch.float32)
        self.Q = torch.eye(6, dtype=torch.float32) * process_noise
        self.R = torch.eye(3, dtype=torch.float32) * measurement_noise
        
        # For batched operations
        self.P_batched = None
    
    def init(self, initial_positions, confidence_scores=None):
        """
        Initialize the Kalman filter with initial positions for all joints
        """
        if confidence_scores is not None:
            # Mark joints with valid initial measurements
            valid_joints = torch.all(confidence_scores >= self.threshold, dim=-1)
            # For invalid joints, we'll initialize but mark time since valid as high
            self.time_since_valid[~valid_joints] = 100.0  # Large value
        else:
            valid_joints = torch.ones(self.njoints, dtype=torch.bool)
        
        self.state[:, 0] = initial_positions[:, 0]
        self.state[:, 1] = initial_positions[:, 1]
        self.state[:, 2] = initial_positions[:, 2]
        
        # Initialize batched covariance matrix with high velocity uncertainty
        self.P_batched = self.P.unsqueeze(0).repeat(self.njoints, 1, 1)
        for i in range(self.njoints):
            self.P_batched[i, 3:, 3:] = torch.eye(3) * 10.0
        
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
    
    def update(self, measured_positions, confidence_scores=None):
        """
        Update step with new measurements - only update joints with valid confidence
        """
        if not self.is_initialized:
            self.init(measured_positions, confidence_scores)
            return self.get_current_positions()
        
        # Determine which joints have valid measurements in this frame
        if confidence_scores is not None:
            valid_measurements = torch.all(confidence_scores > self.threshold, dim=-1)
        else:
            valid_measurements = torch.ones(self.njoints, dtype=torch.bool)
        
        # Update time tracking
        self.time_since_valid += self.delta_time
        self.time_since_valid[valid_measurements] = 0.0  # Reset for valid joints
        self.last_valid_dt[valid_measurements] = self.delta_time  # Update last valid dt
        
        # Only update joints with valid measurements
        valid_indices = torch.where(valid_measurements)[0]
        
        if len(valid_indices) > 0:
            # Extract valid subsets for processing
            valid_state = self.state[valid_indices]
            valid_P = self.P_batched[valid_indices]
            valid_measurements_pos = measured_positions[valid_indices]
            
            # Standard Kalman update for valid joints
            H_batched = self.H.unsqueeze(0).repeat(len(valid_indices), 1, 1)
            H_state = torch.bmm(H_batched, valid_state.unsqueeze(-1)).squeeze(-1)
            y = valid_measurements_pos - H_state
            
            # Innovation covariance: S = H @ P @ H.T + R
            HP = torch.bmm(H_batched, valid_P)
            S = torch.bmm(HP, H_batched.transpose(1, 2)) + \
                self.R.unsqueeze(0).repeat(len(valid_indices), 1, 1)
            
            # Kalman gain: K = P @ H.T @ S^-1
            PHt = torch.bmm(valid_P, H_batched.transpose(1, 2))
            S_inv = torch.linalg.inv(S)
            K = torch.bmm(PHt, S_inv)
            
            # Update state: state = state + K @ y
            Ky = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
            valid_state_updated = valid_state + Ky
            
            # Update covariance: P = (I - K @ H) @ P
            I = torch.eye(6, dtype=torch.float32).unsqueeze(0).repeat(len(valid_indices), 1, 1)
            KH = torch.bmm(K, H_batched)
            valid_P_updated = torch.bmm(I - KH, valid_P)
            
            # Update the main arrays
            self.state[valid_indices] = valid_state_updated
            self.P_batched[valid_indices] = valid_P_updated
        
        # For joints without valid measurements, we only did prediction (no update)
        return self.get_current_positions()
    
    def get_current_positions(self):
        """Get current position estimates for all joints"""
        if not self.is_initialized:
            return torch.zeros(self.njoints, 3, dtype=torch.float32)
        return self.state[:, :3].clone()
