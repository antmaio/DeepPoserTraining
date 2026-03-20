# OpenMPLPoser

## 🚀 Overview

OpenMPLPoser provides a pipeline for training and evaluating models that predict full-body human motion from sparse HMD inputs (head and hand positions) and optional external camera views. It features a clean configuration system, integrated Kalman filtering for temporal smoothness, and automated rendering utilities for visualization.

## 📦 Installation

To set up the environment, you can use the provided `requirements.txt` or `environment.yml` files:

### Using pip
```bash
pip install -r requirements.txt
```

### Using Conda
```bash
conda env create -f environment.yml
conda activate openmplposer
```

## 🚀 Key Features

- **Hybrid Input Pipeline**: Supports head/hand trajectories (global/local) combined with external 3D joints (e.g., from YOLOv8-pose).
- **Temporal-Spatial Architecture**: Uses a modular design with RNN/Transformer blocks for robust sequence modeling.
- **Kalman Filtering Suite**: Integrated support for multiple Kalman filter variants (Constant Acceleration, Student's t, etc.) for online smoothing and occlusion handling.
- **Unified Configuration**: Streamlined TOML-based configuration for data preparation, training, and evaluation.
- **Automated Visualization**: Batch rendering of side-by-side comparison videos (AVI/MP4) with configurable time-scaling.

## 📂 Repository Structure

- **`anim/`**: Core logic and scripts.
  - `train.py`: Main entry point for model training. Handles data loading, optimizer setup, and checkpointing.
  - `test.py`: Fast evaluation script. Automatically loads the latest checkpoint and generates detailed MPJPE/MPJRE metrics.
  - `render.py`: High-quality video generation utility for qualitative analysis.
  - **`models/`**: 
    - `architecture.py`: Definition of RNN/Transformer building blocks.
    - `hmd_poser_ext_hmr_head_centered.py`: The flagship model implementation.
    - `losses.py`: Comprehensive loss suite (FK-space loss, 6D rotation orthonormality, smoothness priors).
  - **`data/`**: AMASS dataset integration and loading logic.
- **`configs/`**: Standard TOML templates.
  - `train.toml`: Global training hyperparameters.
  - `prepare_data.toml`: Data ingestion settings (YOLO model, dataset paths).
- **`filter_model/`**: A library of temporal filters for post-processing and online inference.

## 🛠 Getting Started

### 1. Data Preparation
Configure your paths in `configs/prepare_data.toml` and ensure your AMASS dataset is accessible.

### 2. Training
Run the training script to begin optimization:
```bash
python anim/train.py
```
Settings are pulled from `configs/train.toml`. Logs and model checkpoints are saved to the `./temp` directory by default.

### 3. Evaluation
To evaluate a trained experiment:
```bash
python anim/test.py temp/your_experiment_folder
```
By default, it loads the latest checkpoint and calculates MPJPE, MPJRE, and Jitter metrics on the test split.

### 4. Rendering
Visualize results by rendering AVI videos:
```bash
python anim/render.py temp/your_experiment_folder test --rec_idx 0 10
```
This renders recordings 0 and 10 from the test split using the ground-truth comparison.

## 📊 Configuration System

When training starts, the current configuration is "frozen" into a `run_config.json` file inside the save directory. Evaluation and rendering scripts use this file as their source of truth, ensuring that:
1. Input data protocols (YOLO joints vs. ground-truth) are consistent.
2. Model hyperparameters (RNN hidden size, number of transformer heads) match the weights.
3. System parameters (device, dtype) are defaulted correctly.

## 🌊 Temporal Filtering

Enable Kalman filtering by setting `with_kalman_filter = true` in your config. This applies a `ConstantAcc` (or similar) filter to predicted joint positions, significantly reducing high-frequency jitter in the output sequences.

## Citing

Placeholder for citation.