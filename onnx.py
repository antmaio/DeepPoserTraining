import numpy as np
import torch
import typing
import onnx
import onnxruntime as ort

def convert_mpl_model(model, input, onnx_path:str):

    """
    Convert PyTorch model to ONNX format
    """
    #njoints, ncam = shape
    #assert input.shape == (1, njoints, ncam, 7), "Invalid input shape"
    
    # Export to ONNX
    torch.onnx.export(
        model,
        input,
        onnx_path,
        export_params=True,
        opset_version=20,  # Use latest stable opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Dynamic batch size
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"Model exported to {onnx_path}")
    return onnx_path



def convert_anim_model(model, input_tuple, onnx_path:str):
    """
    Convert animation model to ONNX format with multiple inputs/outputs
    """
    # Ensure model and inputs are on CPU
    # model = model.cpu()
    # input_tuple = tuple(tensor.cpu() if torch.is_tensor(tensor) else tensor 
    #                 for tensor in input_tuple)
    
    # Define input names (descriptive names for each input)
    input_names = [
        'head_pos_global', 'head_rot_global', 'lh_pos_global', 'lh_rot_global', 
        'rh_pos_global', 'rh_rot_global', 'hmr_joints', 'hmr_body_pose', 
        'hmr_global_orient', 'betas', 'gender', 'conf', 'head_vel_global', 
        'lh_vel_global', 'lh_vel_local', 'rh_vel_global', 'rh_rvel_local', 
        'head_rvel_global', 'lh_rvel_global', 'lh_rvel_local', 'rh_rvel_global', 
        'rh_rvel_local'
    ]
    
    output_names = ['betas_pred', 'global_orient_6d_pred', 'body_pose_6d_pred']
    
    # Dynamic axes for all inputs and outputs
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}
    
    # Export to ONNX
    torch.onnx.export(
        model,
        input_tuple,  # Pass the tuple of inputs
        onnx_path,
        export_params=True,
        opset_version=20,  # Use stable opset
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False  # Enable for debugging
    )
    
    print(f"Model exported to {onnx_path}")
    return onnx_path

class ONNXInference:

    """
    onnx_inference = ONNXInference(onnx_path)
    output = onnx_inference(sample_input)
    """

    def __init__(self, onnx_path:str, providers=None, intra_op_num_threads:int=4):
        """
        Initialize ONNX runtime session
        """
        if providers is None:
            providers = [
                'CUDAExecutionProvider',  # GPU first
                #'CPUExecutionProvider'    # CPU fallback
            ]
        
        # Session options for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = intra_op_num_threads  # Adjust based on your CPU
        
        self.session = ort.InferenceSession(onnx_path, session_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded with providers: {self.session.get_providers()}")
    
    def __call__(self, input_tensor):
        """
        Run inference
        """
        # Convert to numpy if needed
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        
        return torch.from_numpy(outputs[0])
