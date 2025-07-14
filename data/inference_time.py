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
#Internal
from data.utils_mmpose import _MODEL_STR_

JSON_PATH = "inference_benchmarks.json"

# Overwrite log file every time the script runs
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mm_pose_model', 
        type=str, 
        nargs='+', 
        default=None,
        help='One or more model aliases, e.g.: td-hm_hrnet-w32_8xb64-210e_coco-256x192 or multiple models separated by space'
    )
    args = parser.parse_args()

    # Your default model string if none is given
    models = args.mm_pose_model if args.mm_pose_model is not None else [_MODEL_STR_]

    for model in models:
        # Instantiate the inferencer
        try:
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

            # Compute stats
            avg_time = float(np.mean(timings))
            std_time = float(np.std(timings))

            logging.info(f"✅ Ran 100 inferences. Avg inference time: {avg_time:.4f} seconds")
            logging.info(f"✅ Std inference time: {std_time:.4f} seconds")

            # Load existing JSON
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r") as f:
                    benchmark_data = json.load(f)
            else:
                benchmark_data = {}

            # Append new data
            benchmark_data[model] = {
                "mean_inference_time": avg_time,
                "std_inference_time": std_time
            }

            # Write back to file
            with open(JSON_PATH, "w") as f:
                json.dump(benchmark_data, f, indent=4)

            logging.info(f"✅ Results saved to {JSON_PATH}")

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
