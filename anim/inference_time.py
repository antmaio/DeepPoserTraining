"""
Script for measuring model inference time.

This script loads a trained model checkpoint, prepares dummy input data 
of the correct shape, and measures average inference latency on GPU.

Key steps:
- Load model from checkpoint
- Move model to GPU (with chosen precision and compile mode)
- Create dummy inputs that match expected model input schema
- Perform warm-up runs to stabilize timings
- Measure and report mean/std inference time over many runs

Intended usage:
$ python -m anim.inference_time <model_dir> --win_len <int> [--checkpoint <int>] ...

Typical use case: benchmark and compare inference speeds of different
checkpoints, architectures, or Torch compile modes.
"""

#Internal
import torch 
torch.set_float32_matmul_precision('high')  # or 'medium' (TF32) | 'highest' (FP32)
import argparse
import logging
import time
#External
import anim.models as models
from anim.data.amass import SmplxJoints, YoloJoints
from anim.models.base import BaseModelInput, BaseModel
from anim.bm_config import _NUM_BETAS_

__COMPILE_MODE__ = [None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

def create_model_input_dict(device,dtype,win_len:int):

    """
    Creates a dictionary of dummy inputs matching the expected model input schema.

    This generates fixed (dummy) positions, rotations, and other body parameters,
    so that inference-time benchmarking can be done without any data loading.
    """
        
    batch_size = 1
    win_len = win_len
    
    eye_3x3 = torch.eye(3, device=device, dtype=dtype)  # shape [3, 3]


    #Dummy pos
    head_pos_global = torch.ones((batch_size, win_len, 3),device=device, dtype=dtype)
    lh_pos_global = torch.ones((batch_size, win_len, 3),device=device, dtype=dtype)
    rh_pos_global = torch.ones((batch_size, win_len, 3),device=device, dtype=dtype)
    #Dummy rot
    head_rot_global = eye_3x3.view(1, 1, 3, 3).expand(batch_size, win_len, -1, -1)
    lh_rot_global = eye_3x3.view(1, 1, 3, 3).expand(batch_size, win_len, -1, -1)
    rh_rot_global = eye_3x3.view(1, 1, 3, 3).expand(batch_size, win_len, -1, -1)
    # Dummy hmr_joints
    hmr_joints = torch.ones((batch_size, win_len, SmplxJoints.NUM_JTS, 3),device=device, dtype=dtype)
    hmr_body_pose = torch.ones((batch_size, win_len, SmplxJoints.NUM_JTS-1, 3),device=device, dtype=dtype)
    hmr_global_orient = eye_3x3.view(1,1,3,3).expand(batch_size, win_len, -1, -1)  # (batch_size, win_len, 3, 3)
    #dummy misc
    betas = torch.zeros((batch_size, win_len, _NUM_BETAS_))
    conf = torch.zeros((batch_size, win_len, YoloJoints.NUM_JTS, 2))
    gender = torch.zeros((batch_size,))

    # Get all local variables
    local_vars = locals()
    
    # Manually specify required keys
    keys = [
        "batch_size", "win_len",
        "head_pos_global", "lh_pos_global", "rh_pos_global",
        "head_rot_global", "lh_rot_global", "rh_rot_global",
        "hmr_joints", "hmr_body_pose", "hmr_global_orient", 
        "betas", "conf", "gender"
    ]
    
    # Return subset of locals()
    return {k: local_vars[k] for k in keys}

def get_inf_time(model:BaseModel, model_input:BaseModelInput, warmup_runs:int=10, num_runs:int=100)->None:
    
    """
    Measures and reports average inference time of the model given dummy input.

    - Runs a number of warm-up passes (to trigger compilation and stabilize GPU kernel timings).
    - Then times multiple runs, synchronizing GPU calls.
    - Prints mean and standard deviation of measured times.
    """
        
    # Warm-up
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(model_input)

    times = []

    with torch.inference_mode():
        for _ in range(num_runs):
            torch.cuda.synchronize()  # Wait for GPU ops to finish
            start_time = time.time()

            _ = model(model_input)

            torch.cuda.synchronize()  # Wait for GPU ops to finish
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()

    print(f"Inference time over {num_runs} runs: mean = {mean_time:.3f} seconds, std = {std_time:.6f} seconds")


def __main():
    
    # Overwrite log file every time the script runs
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    _ = torch.autograd.set_grad_enabled(False)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--win_len', type=int, default=40)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--model_device_str', type=str, default='cuda')
    parser.add_argument('--compile_mode' , type=str, default=None, choices=__COMPILE_MODE__)
    args = parser.parse_args()
    
    model_dir = args.model_dir
    win_len = args.win_len
    checkpoint = args.checkpoint
    model_device = torch.device(args.model_device_str)
    dtype = torch.float32

    # Loading model
    if checkpoint is None:
        checkpoint = models.get_last_model_epoch(model_dir)
    model = models.load_model(model_dir, checkpoint)
    model = model.to(model_device, dtype)
    model.eval()
    if args.compile_mode is not None:
        model = torch.compile(model, mode=args.compile_mode)

    #Create dummy input
    model_input_dict = create_model_input_dict(model_device, dtype, win_len)
    model_input = BaseModelInput(**model_input_dict)

    #Inference time in python measurment
    get_inf_time(model, model_input, warmup_runs=10, num_runs=100)

if __name__ == '__main__':
    __main()