# Environment Setup (conda + uv)

This repo contains two upstream-derived codebases with different dependency
stacks. Keep them in separate environments:

- `mm3d` for `mmdetection3d/`
- `fshnet` for `FSHNet/`

Principles:

- Use conda to manage `python` (and isolate the environment).
- Install PyTorch (+ CUDA wheel) manually for your machine.
- Use `uv` to install Python package requirements.

## Prerequisites

- conda (Miniconda/Mambaforge)
- CUDA toolkit/driver matching your chosen PyTorch wheels (if using GPU)

## mm3d (mmdetection3d)

Create the environment:

```bash
conda create -n mm3d python=3.8
conda activate mm3d

python -m pip install -U pip uv
```

Install PyTorch (pick versions that match your CUDA/driver):

```bash
# Example (update as needed):
python -m pip install \
  torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Install OpenMMLab stack (as used by this repo):

```bash
python -m pip install -U openmim 'numpy==1.23.0'
mim install mmengine
mim install 'mmcv==2.0.0rc4'
mim install 'mmdet==3.0.0'
```

Install this repo's `mmdetection3d/` subtree:

```bash
# Install test deps (includes pytest/yapf/isort/flake8 via requirements)
uv pip install -r mmdetection3d/requirements/tests.txt

# Editable install (use mim so that `mim download/search` can find model-index metadata)
cd mmdetection3d
mim install -e . --no-build-isolation
```

Quick sanity check:

```bash
pytest mmdetection3d/tests/test_models/test_voxel_encoders/test_pillar_encoder.py
```

Notes:

- Optional performance deps (e.g. `cumm`, `spconv`) depend on CUDA; install them
  only if your experiment/config requires them.

## fshnet (FSHNet)

Create the environment:

```bash
conda create -n fshnet python=3.8
conda activate fshnet

python -m pip install -U pip uv
```

Install PyTorch (pick versions that match your CUDA/driver):

```bash
# Example (update as needed; repo authors tested torch>=1.10 for this subtree):
python -m pip install \
  torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Install `FSHNet/` dependencies and build extensions (tested recipe):

```bash
cd FSHNet

# CUDA extensions (pick the CUDA-tagged wheels that match your system)
uv pip install \
  cumm-cu113 \
  spconv-cu113 \
  scikit-image \
  pyyaml \
  tensorboardX \
  easydict \
  pyquaternion \
  opencv-python \
  protobuf==3.20.3 \
  triton==2.1.0 \
  numba==0.48.0

# Build and install (editable)
python setup.py develop

# Waymo extras (only needed for Waymo experiments)
python -m pip uninstall -y waymo-open-dataset-tf-2-4-0
uv pip install waymo-open-dataset-tf-2-11-0

# SharedArray: install with plain pip (uv install may fail)
python -m pip install SharedArray==3.1.0

# Install torch-scatter (will resolve a compatible build for your torch/cuda)
uv pip install torch-scatter
```

Quick sanity check:

```bash
python -c "import pcdet; print('pcdet ok')"
```

Waymo-specific extras:

- Waymo and related packages are only needed for Waymo experiments.
- Prefer documenting these as an opt-in block in your experiment notes, since
  they are platform- and CUDA-sensitive.

Optional dependencies:

- `navsim` is optional. If you need `navsim==2.0.0`, it is not available on PyPI
  and must be installed from source.

Note:

- `FSHNet/` does not ship a solver-friendly `requirements.txt` in this repo.
  Use the recipe above (it matches the original working environment).
