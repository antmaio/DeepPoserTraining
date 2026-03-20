import torch
#Internal 
import filter_model
import anim.data.amass as amass

class Filters:
    def __init__(self, filter_name:str, **kwargs):
        self.filter_name = filter_name
        self.joint_indices = [j.value for j in amass.YoloJoints if 5 <= j.value <= 16] #filter joints

        #apply kalman frame by frame
        kalman_params_path_from_args = kwargs.get('kalman_params_path') 
        self.kalman_params_path = kalman_params_path_from_args if kalman_params_path_from_args is not None else './utils/kalman_parameters.json'
        self.init_filter(filter_name)

    def init_filter(self, filter_name:str):

        joint_params, dt, njoints = filter_model.set_kalman_parameters(self.kalman_params_path)

        if filter_name == 'Gaussian':
            self.kf = filter_model.FlexiblePerJointKalmanFilter(
                dt=1/amass.FPS, #or dt from kalman patameters
                njoints=amass.YoloJoints.NUM_JTS, #or njoints from kalman parameters 
                joint_params=joint_params, 
                active_joints=self.joint_indices 
            )
        elif filter_name == 'TStudent':
            self.kf = filter_model.StudentsTFilter(
                dt=1/amass.FPS, 
                joint_params=joint_params,
                active_joints=self.joint_indices,
                dof=3.0
            )
        
        elif filter_name == 'ConstantAcc':
            self.kf = filter_model.AccelerationPerJointKalmanFilter(
                dt=1/amass.FPS,
                joint_params=joint_params,
                active_joints=self.joint_indices,
                threshold=0.5
            )


    def filter(self, keypoints:torch.Tensor, **kwargs)->torch.Tensor:
        nframes = keypoints.shape[0]
        if self.filter_name == 'Gaussian':

            filtered_positions = []

            for frameIdx in range(nframes):
                # Get 3D positions and confidence scores for this frame
                confidence_scores = torch.tensor(kwargs['conf_scores'][frameIdx], dtype=torch.float32)  # shape: (17, 2)
                
                # Process frame
                filtered_positions.append(self.kf.process_frame(
                    torch.tensor(keypoints[frameIdx], dtype=torch.float32),
                    confidence_scores
                ))

            keypoints = torch.stack(filtered_positions, dim=0)
        
        elif self.filter_name == 'TStudent':
            keypoints = self.kf.filter_joints(keypoints)
        elif self.filter_name == 'ConstantAcc':
            confidence_scores = torch.tensor(kwargs['conf_scores'], dtype=torch.float32)  # shape: (17, 2)
            keypoints = self.kf.filter_joints(keypoints, confidence_scores)
        else:
            raise NotImplementedError(f"Filter {self.filter_name} not implemented")

        return keypoints
