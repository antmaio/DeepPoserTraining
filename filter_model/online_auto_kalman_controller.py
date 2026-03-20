import torch
from filter_model.occlusion_handling_online_batched_kalman_controller import OcclusionHandlingOnlineBatchedKalmanController
from filter_model.online_batched_kalman_controller import OnlineBatchedKalmanController

class _OnlineAutoKalmanControllerOcclusionAware(OcclusionHandlingOnlineBatchedKalmanController):
    def __init__(self, dt, njoints, process_noise=1e-2, measurement_noise=1e-1):
        super().__init__(dt, njoints, process_noise=process_noise, measurement_noise=measurement_noise)
    
    def update(self, measured_positions, confidence_scores=None):
        """
        Update step that auto-initializes on first measurement
        with occlusion handling support
        """
        if not self.is_initialized:
            # Initialize with first measurement - handle confidence if provided
            if confidence_scores is not None:
                valid_joints = torch.all(confidence_scores > 0.0, dim=-1)
                # For invalid joints, we'll initialize but mark as occluded
                self.time_since_valid[~valid_joints] = 100.0  # Large value
            else:
                valid_joints = torch.ones(self.njoints, dtype=torch.bool)
            
            self.state[:, 0] = measured_positions[:, 0]  # x
            self.state[:, 1] = measured_positions[:, 1]  # y
            self.state[:, 2] = measured_positions[:, 2]  # z
            
            # Initialize batched covariance with high velocity uncertainty
            self.P_batched = self.P.unsqueeze(0).repeat(self.njoints, 1, 1)
            # Set high initial velocity uncertainty for all joints
            for i in range(self.njoints):
                self.P_batched[i, 3:, 3:] = torch.eye(3) * 10.0
            
            self.is_initialized = True
            return self.get_current_positions()
        
        # Use parent class update with occlusion handling
        return super().update(measured_positions, confidence_scores)

    def get_current_state(self):
        """Get current state estimates"""
        return self.get_current_positions()

class _OnlineAutoKalmanControllerStandard(OnlineBatchedKalmanController):
    def __init__(self, dt, njoints, process_noise=1e-2, measurement_noise=1e-1):
        super().__init__(dt, njoints, process_noise=process_noise, measurement_noise=measurement_noise)
    
    def update(self, measured_positions, confidence_scores=None):
        """
        Update step that auto-initializes on first measurement.
        Ignores confidence scores for the underlying standard Kalman Filter.
        """
        if not self.is_initialized:
            self.state[:, 0] = measured_positions[:, 0]  # x
            self.state[:, 1] = measured_positions[:, 1]  # y
            self.state[:, 2] = measured_positions[:, 2]  # z
            
            # Initialize batched covariance with high velocity uncertainty
            self.P_batched = self.P.unsqueeze(0).repeat(self.njoints, 1, 1)
            # Set high initial velocity uncertainty for all joints
            for i in range(self.njoints):
                self.P_batched[i, 3:, 3:] = torch.eye(3) * 10.0
            
            self.is_initialized = True
            return self.get_current_positions()
        
        # Use parent class update (standard KF ignores confidence)
        return super().update(measured_positions)

    def get_current_state(self):
        """Get current state estimates"""
        return self.get_current_positions()

class OnlineAutoKalmanController:
    def __new__(cls, dt:float, njoints:int, process_noise:float=1e-2, measurement_noise:float=1e-1, occlusion_aware:bool=True):
        if occlusion_aware:
            return _OnlineAutoKalmanControllerOcclusionAware(dt, njoints, process_noise, measurement_noise)
        else:
            return OnlineBatchedKalmanController(dt, njoints, process_noise, measurement_noise)
