import typing
import torch
import json
from filter_model.online_auto_kalman_controller import OnlineAutoKalmanController

class FlexiblePerJointKalmanFilter:
    def __init__(self, dt, njoints, joint_params=None, active_joints: typing.List[int] = None, occlusion_aware:bool=True):
        self.njoints = njoints
        self.dt = dt
        self.frame_count = 0
        self.occlusion_aware = occlusion_aware
        
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
            kf = OnlineAutoKalmanController(  # ✅ Use the updated class
                occlusion_aware     = self.occlusion_aware,
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
