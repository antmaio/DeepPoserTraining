# OpenMPLPoser

> A brief description or tagline for your project.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Deep Learning-based framework for self-avatar animation from 18DoF sparse inputs extended with multiview 2D pose estimation.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-project.git
cd your-project

# Set up virtual environment (optional)
conda create -n yourenv python=3.9
conda activate yourenv

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python -m data.prepare_data --root ROOT_DIR --protocol PROTOCOL --support_data SUPPORT_DIR --yolo_model YOLO_MODEL --data_split DATA_SPLIT_DIR
```

## Acknowledgments
Our implementation is inspired by the work of AvatarJLM . We thanks the authors for sharing their code.  