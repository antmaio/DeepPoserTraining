# MCaVAPoser

## 🚀 Overview

MCaVAPoser provides a pipeline for training and evaluating models that predict full-body human motion from sparse HMD inputs (head and hand positions) and optional external camera views. It features a clean configuration system, integrated Kalman filtering for temporal smoothness, and automated rendering utilities for visualization.


## 📦 Installation

To set up the environment, you can use the provided `requirements.txt` or `environment.yml` files:

### Datasets

1. Please download the datasets `BMLrub`, `CMU`,`HDM05`, `ACCAD`, `BMLmovi`, `EKUT`, `Eyes_Japan_Dataset`, `KIT`, `MPI_Limits`, `MPI_mosh`, `SFU`, `TotalCapture` from [AMASS](https://amass.is.tue.mpg.de/index.html).
2. Download the required body model and placed them in `support_data/body_models` directory of this repository. For SMPL+H body model, download it from http://mano.is.tue.mpg.de/. Please download the AMASS version of the model with DMPL blendshapes. You can obtain dynamic shape blendshapes, e.g. DMPLs, from http://smpl.is.tue.mpg.de
3. Download joints regressor from SMPLX to COCO from [here](https://github.com/kasvii/PMCE/tree/main/data/COCO)

### Dependencies

#### Using pip
```bash
pip install -r requirements.txt
```

#### Using conda
```bash
conda env create -f environment.yml
conda activate mcavaposer
```

This environment has been tested with Python 3.9

#### Other dependencies
1. Download ```./human_body_prior/human_body_prior``` from ```https://github.com/nghorbani/human_body_prior.git```
2. Download ```./body_visualizer/body_visualizer``` from ```https://github.com/nghorbani/body_visualizer.git```
3. Install ```psbody-mesh``` from ```https://github.com/MPI-IS/mesh.git```. Current release has been tested with ```psbody``` from [here](https://drive.google.com/drive/folders/1Lp4E-Kc8sauclMfG6jpMEXisND_k236c?usp=sharing). 



## 🛠 Usage

### Data Preparation
Configure your paths in `configs/prepare_data.toml` and ensure your AMASS dataset is accessible. You can define the virtual cameras parameters for video rendering in `data/virtual_cameras/`.

Then, run:
```bash
python -m anim.prepare_data --config configs/prepare_data.toml
```

### Training
Training configuration is defined in `configs/train.toml`. You can change the training parameters by editing this file. The model configuration is defined in `configs/model_configs/`. 

```bash
python -m anim.train --config configs/train.toml
```
When training starts, the current configuration is "frozen" into a `run_config.json` file inside the save directory. Evaluation and rendering scripts use this file as their source of truth, ensuring that:
1. Input data protocols (YOLO joints vs. ground-truth) are consistent.
2. Model hyperparameters (RNN hidden size, number of transformer heads) match the weights.
3. System parameters (device, dtype) are defaulted correctly.


### Evaluation
To evaluate a trained model, you can use the following command:
```bash
python -m anim.test <PATH_TO_EXPERIMENT_FOLDER>
```
Metrics are saved in `<PATH_TO_EXPERIMENT_FOLDER>/metrics.json`.

### Rendering
To render animation from a trained model, you can use the following command:
```bash
python -m anim.render <PATH_TO_EXPERIMENT_FOLDER> test --rec_idx <RECORDING_INDICES>
```
This renders recordings of indices <RECORDING_INDICES>. If <RECORDING_INDICES> is empty, it renders all recordings. If <RECORDING_INDICES> contains -1, it renders a random recording.

Example:
```bash
python -m anim.render saves/mcavaposer test --rec_idx 0 2 4 5 
```

## 🚀 Key Features

- **Hybrid Input Pipeline**: Supports head/hand trajectories (global/local) combined with external 3D joints (e.g., from YOLOv8-pose).
- **Temporal-Spatial Architecture**: Uses a modular design with RNN/Transformer blocks for robust sequence modeling.
- **Kalman Filtering Suite**: Integrated support for multiple Kalman filter variants (Constant Acceleration, Student's t, etc.) for online smoothing and occlusion handling.
- **Unified Configuration**: Streamlined TOML-based configuration for data preparation, training, and evaluation.
- **Visualization**: Side-by-side rendering of comparison videos (AVI/MP4) with configurable time-scaling.

## 🌊 Temporal Filtering

Enable Kalman filtering by setting `with_kalman_filter = true` in your config. This applies a `ConstantAcc` or `Gaussian` filter to predicted joint positions, significantly reducing high-frequency jitter in the output sequences.

## Citing

Placeholder for citation.