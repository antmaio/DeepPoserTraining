# OpenMPLPoser

> Deep Learning-based framework for self-avatar animation from 18DoF sparse inputs extended with multiview 2D pose estimation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
1. Download desired subset of AMASS dataset from https://amass.is.tue.mpg.de/download.php. These folders must be placed into ```ROOT_DIR```
2. Download body models from https://smpl-x.is.tue.mpg.de/download.php. The folders must be placed into ```SUPPORT_DIR```
3. Clone the repository
```bash
git clone https://github.com/antmaio/OpenMPLPoser.git
cd OpenMPLPoser
```
4. Set up virtual environment
```bash
conda create -n YOURENV python=3.9
conda activate YOURENV
```
5. Install the required environment via yaml file. Tested on Ubuntu 20.04, Python 3.9, Pytorch 2.0.1+cu118
```bash
conda env create -f environment.yml
```

6. This project uses [MMPose](https://mmpose.readthedocs.io/en/latest/) and [Ultralytics](https://github.com/ultralytics/ultralytics) for pose estimation.

If you haven’t installed them in step 5, you can do so with the following commands and versions:

7. Install OpenMPL_Private:
```bash 
git clone OpenMPL_Private.git
```
and place OpenMPL_Private into MPL_PATH, (e.g. in ./).

Download pretrained model from here and place into MPL_PRETRAINED_MODEL_PATH (e.g. in ./pretrained/xxx.pth).

#### Ultralytics
Install with pip:
```bash
pip install ultralytics==8.3.163
```
#### MMPose
You can find the official installation instructions in the [MMPose documentation](https://mmpose.readthedocs.io/en/latest/installation.html). We recommand to build MMPose from source.
The tested configuration for this project is:
```bash
mmcv==2.0.1
mmdet==3.3.0
mmengine==0.10.4
mmpose==1.3.1
mmpretrain==1.0.0
```
## Usage

#### Running data prerpocessing and pose estimation
This command preprocesses data from amass, then apply multiviews 2D pose estimation on videos rendering. The virtual cameras parameters for rendering are defined in ./data/virtual_cameras/ 
- Use --yolo_model YOLO_MODEL to specify the YOLO model path for pose estimation.
- Or use --mm_pose_model to use the MMPose model as defined in the configuration file.
```bash
python -m data.prepare_data --root ROOT_DIR --protocol PROTOCOL --support_data SUPPORT_DIR --data_split DATA_SPLIT_DIR [--yolo_model YOLO_MODEL | --mm_pose_model]
```
Example:
```bash
python -m data.prepare_data --root ./amass --protocol 1 --support_data ./support_data --data_split ./data_split --yolo_model yolov8n-pose
```

#### multiviews 2D/3D pose lifter
The methods to extract 3D positions from multiviews 2D keypoints and camera calibration are defined in ./pose_lifter 
- Triangulation
 ```bash
python -m pose_lifter.triang --dataset_type DATASET_TYPE --dataroot KEYPOINTS_DIR
```
- OpenMPL
 ```bash
python -m pose_lifter.triang --dataset_type DATASET_TYPE --dataroot KEYPOINTS_DIR --mpl_path MPL_PATH --openmpl_ckpt MPL_PRETRAINED_MODEL_PATH 
```

KEYPOINTS_DIR is the directory of pickle files built from ```data.prepare_data```

Example:
```bash
python -m pose_lifter.triang --dataset_type amass_p1 --dataroot ./data/keypoints/yolov8n-pose_protocol_1
```
or 
```bash
python -m pose_lifter.triang --dataset_type amass_p1 --dataroot ./data/keypoints/yolov8n-pose_protocol_1 --mpl_path ./OpenMPL_Private --openmpl_ckpt ./pretrained/mpl/td-hm_hrnet-w32_8xb64-210e_coco-384x288/model_best.pth.tar
```


#### Self-avatar animation from sparse trackers and 3D keypoints

- Training
run 
```bash
python -m anim.train ANIM_MODEL --dataset DATASET_TYPE
```
Example:
```bash
python -m anim.train ./anim/data/model_configs/hmd-poser-ext.toml --dataset amass-p1 
```
- Evaluating 
run 
```bash
python -m anim.test ANIM_MODEL_TRAINED DATASET_TYPE SPLIT --checkpoint CKPT --batch_size 1
```
Example:
```bash
python -m anim.test ./saves/hmd-poser-ext_25-6-25-11-20-58 amass-p1 test --checkpoint 400 --batch_size 1  
```
- Rendering 
run 
```bash
python -m anim.render ANIM_MODEL_TRAINED DATASET_TYPE SPLIT --epoch CKPT --rec_idx REC_IDX
```

- ```ANIM_MODEL``` is the path to configuration TOML file for self-avatar animation model. 
- ```ANIM_MODEL_TRAINED``` is the folder path related to the trained model.
- ```SPLIT``` refers to the subset on which the model will be evaluated. 
- ```CKPT``` is the model checkpoint to load
- ```REC_IDX``` indicates which file id to render

## License 

## Acknowledgments
Our implementation is inspired by the work of AvatarJLM . We thanks the authors for sharing their code. 