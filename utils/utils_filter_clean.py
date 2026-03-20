import torch 
import torch.nn as nn
from typing import Dict, List
import json

class AMASSKalmanFiltering(nn.Module):

    def __init__(self,
        dt:float,
        njoints:int,
        process_noise:float     = 1e-2,
        measurement_noise:float = 1e-2,
        active_joints:List[int] = None,
        kalman_params_path:str  = None
    ):

        super().__init__()        
        self.delta_time = dt
        self.njoints = njoints
        self.batch_size = None
        
        # Load joint-specific parameters from JSON file
        self.joint_params, self.active_joints = self._load_kalman_parameters(kalman_params_path, active_joints)
        
        # Register all tensors as buffers so they move with .to(device)
        # Transition matrix (position + velocity)
        F = torch.eye(6, dtype=torch.float32)
        for i in range(3):
            F[i, i + 3] = dt
        self.register_buffer('F', F)

        # Observation matrix - only observe position
        H = torch.zeros(3, 6, dtype=torch.float32)
        for i in range(3):
            H[i, i] = 1
        self.register_buffer('H', H)

        # Covariances
        P = torch.eye(6, dtype=torch.float32)
        self.register_buffer('P', P)
        
        Q = torch.eye(6, dtype=torch.float32) * process_noise
        self.register_buffer('Q', Q)
        
        R = torch.eye(3, dtype=torch.float32) * measurement_noise
        self.register_buffer('R', R)
        
        # State and covariance as buffers (will be initialized later)
        self.register_buffer('state', None)
        self.register_buffer('P_batched', None)
        
        self.is_initialized = False

    def _initialize_joint_matrices(self, batch_size: int):
        """Initialize Q and R matrices for each joint"""
        self.Q_matrices = torch.zeros(batch_size, self.njoints, 6, 6, dtype=torch.float32)
        self.R_matrices = torch.zeros(batch_size, self.njoints, 3, 3, dtype=torch.float32)
        
        for b in range(batch_size):
            for j in self.active_joints:
                process_noise = self.joint_params[j]['process_noise']
                measurement_noise = self.joint_params[j]['measurement_noise']
                
                # Initialize Q matrix for this joint
                self.Q_matrices[b, j] = torch.eye(6, dtype=torch.float32) * process_noise
                # Initialize R matrix for this joint  
                self.R_matrices[b, j] = torch.eye(3, dtype=torch.float32) * measurement_noise
    def _load_kalman_parameters(self, kalman_params_path: str, active_joints: List[int] = None):
        """Load joint-specific parameters from JSON file"""
        try:
            with open(kalman_params_path, 'r') as f:
                params_config = json.load(f)
            
            # Create joint parameters list
            joint_params = []
            for joint_config in params_config['joint_parameters']:
                joint_params.append({
                    'process_noise': joint_config['process_noise'],
                    'measurement_noise': joint_config['measurement_noise']
                })
            
            # Determine active joints
            if active_joints is None:
                active_joints = list(range(len(joint_params)))
            
            print(f"Loaded Kalman parameters for {len(active_joints)} active joints from {kalman_params_path}")
            
            # Print parameters for verification
            for j in active_joints:
                print(f"  Joint {j}: process_noise={joint_params[j]['process_noise']:.6f}, "
                      f"measurement_noise={joint_params[j]['measurement_noise']:.6f}")
            
            return joint_params, active_joints
            
        except FileNotFoundError:
            print(f"Warning: Kalman parameters file {kalman_params_path} not found. Using defaults.")
            # Create default parameters
            joint_params = [{'process_noise': 1e-2, 'measurement_noise': 1e-1} for _ in range(self.njoints)]
            active_joints = list(range(self.njoints))
            return joint_params, active_joints

    def set_state_from_positions_and_velocity(self, positions: torch.Tensor, velocity: torch.Tensor):
        """Helper method to set state from positions and velocities"""
        assert positions.shape == velocity.shape, f"positions and velocity should have the same shape, instead got resp. {positions.shape} {velocity.shape}"
        assert len(positions.shape) == 3  # bs, njoints, 3
        
        if not self.is_initialized:
            self.init_with_velocities(positions, velocity)
        else:
            self.state[..., :3] = positions
            self.state[..., 3:] = velocity
    def init_with_velocities(self, initial_positions: torch.Tensor, initial_velocities: torch.Tensor):
        """
        Initialize with known positions AND velocities
        
        Args:
            initial_positions: torch.Tensor of shape (batch_size, njoints, 3)
            initial_velocities: torch.Tensor of shape (batch_size, njoints, 3)
        """
        batch_size, njoints, _ = initial_positions.shape
        self.batch_size = batch_size
        
        # Initialize state as a buffer
        state = torch.zeros(batch_size, njoints, 6, dtype=torch.float32, device=self.F.device)
        state[:, :, 0:3] = initial_positions  # positions
        state[:, :, 3:6] = initial_velocities  # velocities
        self.register_buffer('state', state)
        
        # Initialize batched covariance as a buffer
        P_batched = self.P.unsqueeze(0).unsqueeze(0).repeat(batch_size, njoints, 1, 1)
        self.register_buffer('P_batched', P_batched)
        
        self.is_initialized = True
    def predict(self):
        """Prediction step for all joints and batches"""
        if not self.is_initialized:
            return
        
        batch_size, njoints, _ = self.state.shape
        
        # Update state: x = F @ x
        for b in range(batch_size):
            for j in range(njoints):
                self.state[b, j] = torch.matmul(self.F, self.state[b, j])
        
        # Update covariance: P = F @ P @ F.T + Q
        for b in range(batch_size):
            for j in range(njoints):
                FP = torch.matmul(self.F, self.P_batched[b, j])
                self.P_batched[b, j] = torch.matmul(FP, self.F.T) + self.Q
        
    def get_current_positions(self):
        """Get current position estimates for all joints and batches"""
        if not self.is_initialized:
            if self.batch_size is not None:
                return torch.zeros(self.batch_size, self.njoints, 3, dtype=torch.float32)
            return torch.zeros(self.njoints, 3, dtype=torch.float32)
        return self.state[:, :, :3].clone()

    def get_current_velocities(self):
        """Get current velocity estimates for all joints and batches"""
        if not self.is_initialized:
            if self.batch_size is not None:
                return torch.zeros(self.batch_size, self.njoints, 3, dtype=torch.float32)
            return torch.zeros(self.njoints, 3, dtype=torch.float32)
        return self.state[:, :, 3:6].clone()

    def update_position_only(self, measured_positions: torch.Tensor):
        """
        Standard update with only position measurements
        
        Args:
            measured_positions: torch.Tensor of shape (batch_size, njoints, 3)
        """
        if not self.is_initialized:
            # Auto-initialize with zero velocities
            zero_velocities = torch.zeros_like(measured_positions)
            self.init_with_velocities(measured_positions, zero_velocities)
            return self.get_current_positions()
        
        batch_size, njoints, _ = measured_positions.shape
        
        # Get the device from one of our buffers
        device = self.F.device
        
        # Process each batch and joint
        for b in range(batch_size):
            for j in range(njoints):
                # Extract single state and covariance
                state = self.state[b, j]  # (6,)
                P = self.P_batched[b, j]  # (6, 6)
                measurement = measured_positions[b, j]  # (3,)
                
                # Kalman update math for single joint
                H_state = torch.matmul(self.H, state)  # (3,)
                y = measurement - H_state  # (3,)

                # Innovation covariance: S = H @ P @ H.T + R
                HP = torch.matmul(self.H, P)  # (3, 6)
                S = torch.matmul(HP, self.H.T) + self.R  # (3, 3)
                
                # Kalman gain: K = P @ H.T @ S^-1
                PHt = torch.matmul(P, self.H.T)  # (6, 3)
                S_inv = torch.linalg.inv(S)  # (3, 3)
                K = torch.matmul(PHt, S_inv)  # (6, 3)
                
                # Update state
                Ky = torch.matmul(K, y)  # (6,)
                state_updated = state + Ky
                
                # Update covariance: P = (I - K @ H) @ P
                I = torch.eye(6, dtype=torch.float32, device=device)  # ← FIX: Add device here
                KH = torch.matmul(K, self.H)  # (6, 6)
                P_updated = torch.matmul(I - KH, P)
                
                # Store updated values
                self.state[b, j] = state_updated
                self.P_batched[b, j] = P_updated
        
        return self.get_current_positions()
    
    def get_joint_parameters(self, joint_idx: int) -> Dict[str, float]:
        """Get parameters for a specific joint"""
        if 0 <= joint_idx < len(self.joint_params):
            return self.joint_params[joint_idx]
        else:
            return {'process_noise': 1e-2, 'measurement_noise': 1e-1}
    


# Usage example
if __name__ == "__main__":
    # Initialize
    batch_size = 4
    njoints = 12
    nframes = 40

    kf = AMASSKalmanFiltering(
        dt=1/60.0,
        njoints=njoints,
        kalman_params_path='./kalman_parameters.json'
        #process_noise=1e-2,
        #measurement_noise=1e-1
    )

    # Generate synthetic data
    pos = torch.randn(batch_size, nframes, njoints, 3)

    # Process frames
    for frame_idx in range(nframes):
        if frame_idx == 0:
            # Initialize with first frame
            initial_vel = torch.zeros(batch_size, njoints, 3)  # Zero velocity for first frame
            kf.init_with_velocities(pos[:, frame_idx], initial_vel)
            prev_pos = pos[:, frame_idx]
        else:
            # Calculate velocity from position difference
            current_vel = (pos[:, frame_idx] - prev_pos) 
            
            # Set state from current positions and velocities
            kf.set_state_from_positions_and_velocity(pos[:, frame_idx], current_vel)
            
            # Predict step
            kf.predict()
            
            # New measurements (simulated noisy measurements)
            measurements = pos[:, frame_idx] + torch.randn_like(pos[:, frame_idx]) * 0.1
            
            # Update step
            filtered_positions = kf.update_position_only(measurements)
            
            # Get current states
            current_pos = kf.get_current_positions()  # (4, 12, 3)
            current_vel = kf.get_current_velocities()  # (4, 12, 3)
            
            prev_pos = pos[:, frame_idx]
            
            print(f"Frame {frame_idx}: Filtered positions shape: {filtered_positions.shape}")