"""
This module has been refactored. 
All Kalman filter classes are now located in the root `filter/` directory.
They are imported here for legacy backward compatibility.
"""

from filter_model.online_batched_kalman_controller import OnlineBatchedKalmanController
from filter_model.occlusion_handling_online_batched_kalman_controller import OcclusionHandlingOnlineBatchedKalmanController
from filter_model.online_auto_kalman_controller import OnlineAutoKalmanController, _OnlineAutoKalmanControllerOcclusionAware, _OnlineAutoKalmanControllerStandard
from filter_model.velocity_aware_online_kalman_controller import VelocityAwareOnlineBatchedKalmanController, VelocityAwareOnlineKalmanController
from filter_model.flexible_per_joint_kalman_filter import FlexiblePerJointKalmanFilter, set_kalman_parameters
from filter_model.flexible_per_joint_kalman_filter_with_velocity import FlexiblePerJointKalmanFilterWithVelocity

__all__ = [
    'OnlineBatchedKalmanController',
    'OcclusionHandlingOnlineBatchedKalmanController',
    'OnlineAutoKalmanController',
    '_OnlineAutoKalmanControllerOcclusionAware',
    '_OnlineAutoKalmanControllerStandard',
    'VelocityAwareOnlineBatchedKalmanController',
    'VelocityAwareOnlineKalmanController',
    'FlexiblePerJointKalmanFilter',
    'set_kalman_parameters',
    'FlexiblePerJointKalmanFilterWithVelocity'
]