#External
# Register mmpretrain models
from mmpose.apis import MMPoseInferencer
import mmpretrain.models
import torch
import time
import argparse
import numpy as np
import json
import os
import logging
import typing
#Internal
from data.utils_mmpose import _MODEL_STR_
from data.utils_yolo import init_yolo


# Overwrite log file every time the script runs
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def inference_mmpose(model)->typing.List[float]:
    
    inferencer = MMPoseInferencer(model, device='cuda')
    # Warm-up phase (10 runs)
    for _ in range(10):
        result_generator = inferencer('body_image_cam_1.png', show=False)
        _ = next(result_generator)

    # Timed inference phase (100 runs)
    timings = []
    for _ in range(100):
        result_generator = inferencer('body_image_cam_1.png', show=False)

        torch.cuda.synchronize()
        start = time.time()
        _ = next(result_generator)
        torch.cuda.synchronize()
        end = time.time()

        timings.append(end - start)
    
    return timings

def inference_yolo(model_str:str)->typing.List[float]:
    
    model = init_yolo(model_str+'.pt')
    # Predict on a single image
    results = model("body_image_cam_1.png")

    # Warm-up phase (10 runs)
    for _ in range(10):
        results = model("body_image_cam_1.png")

    # Timed inference phase (100 runs)
    timings = []
    for _ in range(100):

        torch.cuda.synchronize()
        start = time.time()
        results = model("body_image_cam_1.png")
        torch.cuda.synchronize()
        end = time.time()

        timings.append(end - start)
    
    return timings
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mm_pose_model', 
        type=str, 
        nargs='*', 
        default=None,
        help='One or more model aliases, e.g.: td-hm_hrnet-w32_8xb64-210e_coco-256x192 or multiple models separated by space'
    )
    parser.add_argument('--yolo_model',
        type=str,
        nargs='*',
        default=None,
        help='One or more model aliases, e.g.: yolov8n-pose or multiple models separated by space'
    )
    parser.add_argument('--json_filename', type=str, default="inference_benchmarks")
    args = parser.parse_args()

    # asserts
    if (args.yolo_model is None and args.mm_pose_model is None) or \
    (args.yolo_model is not None and args.mm_pose_model is not None):
        parser.error('You must specify exactly one of --yolo_model or --mm_pose_model.')
    assert args.json_filename is not None, f"please provide a valid json filename instead of {args.json_filename}"

    # Your default model string if none is given
    if args.mm_pose_model and len(args.mm_pose_model) == 0:
        inference_func = inference_mmpose 
        args.json_filename += "_mm"
        models = [_MODEL_STR_]
    elif args.mm_pose_model and len(args.mm_pose_model) > 0:
        inference_func = inference_mmpose 
        models = args.mm_pose_model    
        args.json_filename += "_mm"
    elif args.yolo_model is not None:
        inference_func = inference_yolo
        models = args.yolo_model
        args.json_filename += "_yolo"
    else:
        raise ValueError('Check carefully args.yolo_model and args.mm_pose_model')

    args.json_filename += '.json'
    
    
    for model in models:
        # Instantiate the inferencer
        try:

            timings = inference_func(model)

            # Compute stats
            avg_time = float(np.mean(timings))
            std_time = float(np.std(timings))

            logging.info(f"✅ Ran 100 inferences. Avg inference time: {avg_time:.4f} seconds")
            logging.info(f"✅ Std inference time: {std_time:.4f} seconds")

            # Load existing JSON
            if os.path.exists(args.json_filename):
                with open(args.json_filename, "r") as f:
                    benchmark_data = json.load(f)
            else:
                benchmark_data = {}

            # Append new data
            benchmark_data[model] = {
                "mean_inference_time": avg_time,
                "std_inference_time": std_time
            }

            # Write back to file
            with open(args.json_filename, "w") as f:
                json.dump(benchmark_data, f, indent=4)

            logging.info(f"✅ Results saved to {args.json_filename}")

        except:
            logging.info(f'Ignore data for {model}')
            continue

"""
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mm_pose_model', type=str, default=None, help='e.g.:td-hm_hrnet-w32_8xb64-210e_coco-256x192')
    args = parser.parse_args()
    

    # instantiate the inferencer using the model alias
    model = args.mm_pose_model if args.mm_pose_model is not None else _MODEL_STR_ 
    inferencer = MMPoseInferencer(model, device='cuda')
    result_generator = inferencer('body_image_cam_1.png', out_dir = "mmpose_output")

    # Warm-up phase (10 runs)
    for _ in range(10):
        result_generator = inferencer('body_image_cam_1.png', show=False)
        _ = next(result_generator)

    
    # Timed inference phase (100 runs)
    timings = []
    for _ in range(100):
        result_generator = inferencer('body_image_cam_1.png', show=False)

        torch.cuda.synchronize()
        start = time.time()
        _ = next(result_generator)
        torch.cuda.synchronize()
        end = time.time()

        timings.append(end - start)

    # Reporting results
    avg_time = sum(timings) / len(timings)
    print(f"✅ Ran 100 inferences. Avg inference time: {avg_time:.4f} seconds")
"""
if __name__ == '__main__':
    main()
