#External
import argparse
import glob 
import os 
import subprocess

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Optional: controls time format
)
#Internal
# from data.camera_config import CAMERA_PATH
# from data.yolo_data_gen import extract_from_xml
# from body_visualizer.mesh.mesh_viewer import MeshViewer
# from body_visualizer.tools.vis_tools import colors
from data.camera_config import CAMERA_PATH_PROTOCOL_1, CAMERA_PATH_PROTOCOL_2, CAMERA_PATH_PROTOCOL_3


def __main():
    """
    Main execution function for multi-view 3D pose triangulation.

    Processes YOLO keypoints data and performs triangulation using:
    - Camera parameters from XML files
    - 2D keypoints from pickle files
    - Confidence scores to select best camera pairs

    Command-line Arguments
    ---------------------
    --dataset_type : str
        Dataset split identifier (default: 'amass_p1')
    --dataroot : str
        Path to directory containing pickle files (default: './data/keypoints/yolov8n-pose_protocol_1')

    Outputs
    -------
    points3d : np.ndarray
        Array of triangulated 3D points with shape (nframes, njoints, 3)
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_type', default='amass_p1', type=str, choices=('amass_p1', 'amass_p2', 'amass_p3'), help="Dataset split as in AvatarJLM")
    parse.add_argument('--dataroot', default='./data/keypoints/yolov8x-pose_protocol_1', type=str, help='Path to pkl files')
    parse.add_argument('--mpl_path', default='../OpenMPL_private',help="Path to OpenMPL git")
    parse.add_argument('--openmpl_ckpt', default='./pretrained/mpl/yolo11n-pose/model_best_yolo11_p1.pth.tar', help='Path of ckpt .pth.tar file for openmpl')
    parse.add_argument('--output_dir', default='openmpl', type=str, help='relative output path to 3D keypoints')
    args = parse.parse_args()
     
    # Get all camera XML files (any naming pattern)
    # camera_files = glob.glob(os.path.join(CAMERA_PATH, '*.xml'))

    dataset_type = args.dataset_type
    dataroot = args.dataroot
    mpl_path = args.mpl_path
    openmpl_ckpt = args.openmpl_ckpt
    phases = ['train', 'test']

    assert args.dataset_type in ('amass_p1', 'amass_p2', 'amass_p3'), f"{args.dataset_type} not supported for --dataset_type"

    for phase in phases:
        if dataset_type == 'amass_p1':
            protocol = 1
            filename_list = glob.glob(f'{dataroot}/*/{phase}/preprocessed/*.pkl')
            camera_path = CAMERA_PATH_PROTOCOL_1
            assert os.path.exists(camera_path), f"camera dir {camera_path} do not exist!"
        elif dataset_type == 'amass_p2':
            protocol = 2 #TODO change as_testset ! 
            camera_path = CAMERA_PATH_PROTOCOL_1
            if phase == 'train':
                filename_list = glob.glob(f'{dataroot}/MPI_HDM05/*/*.pkl') + glob.glob(f'{dataroot}/BioMotionLab_NTroje/*/*.pkl')
            else:
                filename_list = glob.glob(f'{dataroot}/CMU/*/*.pkl')
        else:
            protocol = 3
            camera_path = CAMERA_PATH_PROTOCOL_3
            if phase == 'train':
                datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLmovi', 'CMU','EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'TotalCapture']
                filename_list = [
                    f for dataset in datasets
                    for f in glob.glob(f'./{dataroot}/{dataset}/*/*/*.pkl')]
            else:
                filename_list = glob.glob(f'./{dataroot}/HumanEva/*/*/*.pkl') + glob.glob(f'./{dataroot}/Transitions_mocap/*/*/*.pkl')
        
        logging.info('-------------------------------number of {} data is {}'.format(phase, len(filename_list)))
        filename_list = sorted(filename_list)

        # create .../openmpl
        if len(filename_list) > 0:
            input_dir = os.path.dirname(filename_list[0])
            output_dir = os.path.join(input_dir, args.output_dir)
            os.makedirs(output_dir, exist_ok=True)

        for filename in filename_list:

            base_name = os.path.splitext(os.path.basename(filename))[0] + ".npz"

            input_dir = os.path.dirname(filename)
            output_dir = os.path.join(input_dir, args.output_dir)
            output_dir = os.path.normpath(output_dir)

            output_dir_parts = output_dir.split(os.sep)
            if 'preprocessed' in output_dir_parts:
                output_dir_parts.remove('preprocessed')
            output_dir = os.path.join(*output_dir_parts)
            output_path = os.path.join(output_dir, base_name)

                
            if os.path.exists(output_path):
                #do not rerun if the file exists
                logging.info(f' File {output_path} already exists!')
                continue
            else:
               
                #TODO change to this config : 
                result = subprocess.run(['python', f'{mpl_path}/RUMPL/run/inference_rumpl.py',
                                        '--cfg',
                                        f'{mpl_path}/RUMPL/configs/openmplposer/rumpl_amass_poser_6_openmplposer_aligned_yolo8_p3/rumpl_601_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml',
                                        #f'{mpl_path}/RUMPL/configs/openmplposer/rumpl_amass_poser/rumpl_204_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml',
                                        #   '/home/ucl/elen/abolfazl/OpenMPL/RUMPL/configs/openmplposer/rumpl_amass_random/rumpl_101_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV3.yaml',
                                        "--data-dir", filename,
                                        "--data-out-dir", output_path,
                                        "--cameras-path", f"{camera_path}",
                                        "--trained-model", f'{openmpl_ckpt}'
                                        ], 
                                        capture_output=True, text=True)
                
                logging.info("--- Error Catched ---")
                logging.info(result.stderr)
                logging.info("--- Result ---")
                logging.info(result.stdout)
                # break



if __name__ == '__main__':
    __main()