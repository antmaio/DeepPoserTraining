import typing
import torch
from filter_model.velocity_aware_online_kalman_controller import VelocityAwareOnlineKalmanController

class FlexiblePerJointKalmanFilterWithVelocity:
    def __init__(self, dt, njoints, joint_params=None, active_joints: typing.List[int] = None, use_velocity_measurements: bool = False):
        self.njoints = njoints
        self.dt = dt
        self.frame_count = 0
        self.use_velocity_measurements = use_velocity_measurements
        
        # Determine which joints are active
        if active_joints is None:
            self.active_joints = list(range(njoints))
        else:
            self.active_joints = active_joints
        
        # Default parameters
        default_params = {
            'process_noise': 1e-2,
            'measurement_noise': 5e-1
        }
        
        if joint_params is None:
            joint_params = [default_params.copy() for _ in range(njoints)]
        
        self.joint_filters = []
        self.joint_params = joint_params
        
        # Create filters for active joints
        for j in self.active_joints:
            kf = VelocityAwareOnlineKalmanController(
                dt=dt,
                njoints=1,
                process_noise=joint_params[j]['process_noise'],
                measurement_noise=joint_params[j]['measurement_noise']
            )
            self.joint_filters.append(kf)

    def init_with_velocities(self, initial_pos, initial_vel, confidence_scores=None):
        """
        Initialize all per-joint Kalman filters using NN-predicted
        initial positions and velocities (first frame).
        """
        self.frame_count = 1  # first frame initialization

        print(initial_pos.shape, initial_vel.shape)

        # initialize each joint KF
        for idx, j in enumerate(self.active_joints):
            pos_j = initial_pos[j:j+1, :]   # (1,3)
            vel_j = initial_vel[j:j+1, :]   # (1,3)

            self.joint_filters[idx].init_with_velocities(
                pos_j, 
                vel_j,
                None if confidence_scores is None else confidence_scores[j:j+1]
            )
    
    def process_frame(self, measurements, confidence_scores=None, measured_velocities=None):
        """
        Process a new frame with optional velocity measurements
        
        Args:
            measurements: torch.Tensor of shape (njoints, 3) - 3D positions
            confidence_scores: torch.Tensor of shape (njoints, 2) - confidence scores  
            measured_velocities: torch.Tensor of shape (njoints, 3) - velocity measurements
        """
        self.frame_count += 1
        filtered_positions = torch.zeros_like(measurements)
        
        for idx, j in enumerate(self.active_joints):
            joint_measurement = measurements[j:j+1, :]
            
            if confidence_scores is not None:
                joint_confidence = confidence_scores[j:j+1, :]
            else:
                joint_confidence = None
                
            if measured_velocities is not None and self.use_velocity_measurements:
                joint_velocity = measured_velocities[j:j+1, :]
            else:
                joint_velocity = None
            
            # Predict step (skip for first frame)
            if self.frame_count > 1:
                self.joint_filters[idx].predict()
            
            # Update step
            if joint_velocity is not None:
                # Use velocity measurements if available
                filtered_joint = self.joint_filters[idx].update_with_velocities(
                    joint_measurement, joint_velocity, joint_confidence
                )
            else:
                # Standard position-only update
                filtered_joint = self.joint_filters[idx].update_position_only(
                    joint_measurement, joint_confidence
                )
            
            filtered_positions[j] = filtered_joint[0]
        
        return filtered_positions
    
    def calculate_velocities_from_history(self, current_positions, previous_positions):
        """
        Helper method to calculate velocities from position history
        """
        return (current_positions - previous_positions) / self.dt
    
    def get_current_state(self):
        """Get current state estimates for all joints"""
        states = torch.zeros(self.njoints, 3)
        for idx, j in enumerate(self.active_joints):
            states[j] = self.joint_filters[idx].get_current_positions()[0]
        return states
    
    def get_current_velocities(self):
        """Get current velocity estimates for all joints"""
        velocities = torch.zeros(self.njoints, 3)
        for idx, j in enumerate(self.active_joints):
            velocities[j] = self.joint_filters[idx].get_current_velocities()[0]
        return velocities
