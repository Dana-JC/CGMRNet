# CGMRNet

A lightweight image dehazing network for underground coal mine scenes.

## Introduction

This repository provides the implementation of CGMRNet for enhancing underground images degraded by dust and mist.

## Environment and Dependencies

CGMRNet is implemented using PyTorch. The following Python packages are required:

- torch
- torchvision
- numpy
- Pillow
- scipy
- opencv-python
- tensorboardX

### Installation

Create and activate a Python environment. Python 3.10 is suggested as a starting point:

```bash
conda create -n cgmrnet python=3.10 -y
conda activate cgmrnet
```

Install compatible versions of PyTorch and torchvision by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). For GPU execution, select a CUDA-enabled build compatible with your NVIDIA driver.

Install the remaining dependencies:

```bash
python -m pip install numpy Pillow scipy opencv-python tensorboardX
```

To visualize training logs, optionally install TensorBoard:

```bash
python -m pip install tensorboard
```

TensorFlow is not required. A fully tested, version-pinned environment is not yet provided.

### GPU Support

The training scripts use GPU execution by default, and `val.py` requires a CUDA-enabled NVIDIA GPU. Check CUDA availability with:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

For CPU inference with `predict.py`, set `--gpu_id -1`.

## Usage

Run the scripts from the repository root. Update the dataset, checkpoint, input, and output paths to match your local environment before execution.

- `train-CA.py` uses the cosine annealing learning rate scheduler.
- `train-MS.py` uses the MultiStepLR learning rate scheduler.
- `val.py` evaluates model performance using paired degraded and ground-truth images.
- `predict.py` performs image dehazing and saves the restored images.

To visualize training logs with the default log directory, run:

```bash
tensorboard --logdir tblogdir
```

## Dataset

The IMDITE dataset is available upon request from the corresponding author.