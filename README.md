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
git clone https://github.com/your-username/your-project.git
cd your-project
4. Set up virtual environment (optional)
```bash
conda create -n YOURENV python=3.9
conda activate YOURENV
```
5. Install the required environment. Tested on Python 3.9, Pytorch 2.3.1
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scikit-learn==1.5.0
pip install trimesh==4.4.0
pip install ultralytics==8.3.92
pip install pyrender==0.1.45
pip install -U numpy==1.26.4
```
or install via the yaml file
```bash
conda env create -f environment.yml
```
## Usage
```bash
python -m data.prepare_data --root ROOT_DIR --protocol PROTOCOL --support_data SUPPORT_DIR --yolo_model YOLO_MODEL --data_split DATA_SPLIT_DIR
```
Then, to get 3d keypoints based on multiview 2d pose estimation, 
 ```bash
python -m data.pose-lifter.triang --dataset_type DATASET_TYPE --dataroot KEYPOINTS_DIR
```
Example:
```bash
python -m data.pose-lifter.triang --dataset_type amass_p1 --dataroot ./data/keypoints/yolov8n-pose_protocol_1
```
KEYPOINTS_DIR is the directory of pickle files built from ```data.prepare_data```
 
## License 

## Acknowledgments
Our implementation is inspired by the work of AvatarJLM . We thanks the authors for sharing their code. 