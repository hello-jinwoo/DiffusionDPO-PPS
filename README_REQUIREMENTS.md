# Requirements Installation Guide

**Last Updated**: 2025-10-15

## Overview

This project has been updated with comprehensive and up-to-date requirements files:
- `requirements.txt`: Production dependencies (28 packages)
- `requirements-dev.txt`: Development tools (includes all production deps + testing/linting)

## Quick Start

### 1. Basic Installation

```bash
# Install production requirements
pip install -r requirements.txt

# Or install with development tools
pip install -r requirements-dev.txt
```

### 2. PyTorch Installation Notes

**IMPORTANT**: PyTorch installation depends on your CUDA version.

#### Option A: Auto-detect (Recommended)
```bash
# requirements.txt will install the default PyTorch
pip install -r requirements.txt
```

#### Option B: Specific CUDA Version
```bash
# For CUDA 11.8
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cpu

# Then install the rest
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Quick check
python -c "import torch; import diffusers; print('✅ Core packages OK')"

# Full verification (with import_fix)
python -c "import utils.import_fix; import peft; print('✅ All packages OK')"
```

## Package Categories

### Core ML Framework (3 packages)
- `torch>=2.4.0` - PyTorch deep learning framework
- `torchvision>=0.19.0` - Vision utilities
- `numpy>=2.0.0,<3.0.0` - Numerical computing

### Diffusion Models & Training (6 packages)
- `diffusers==0.35.1` - Hugging Face diffusion models
- `accelerate==1.1.0` - Distributed training
- `transformers==4.37.2` - Transformer models
- `peft==0.17.1` - Parameter-efficient fine-tuning
- `safetensors==0.6.2` - Safe model serialization
- `huggingface-hub>=0.35.0` - Model hub integration

### Vision Models (2 packages)
- `timm==1.0.19` - Vision model library (DINOv2, etc.)
- `openai_clip==1.0.1` - CLIP model

### Metrics & Evaluation (3 packages)
- `hpsv2==1.2.0` - Human Preference Score v2
- `lpips==0.1.4` - Perceptual similarity
- `scikit-image==0.25.2` - Image processing & metrics

### Data Processing (8 packages)
- `datasets==2.18.0` - Hugging Face datasets
- `pillow>=11.0.0` - Image I/O
- `opencv-python>=4.12.0` - Computer vision
- `scipy>=1.15.0` - Scientific computing
- `matplotlib>=3.10.0` - Plotting
- `ftfy>=6.0.0` - Text cleaning
- `tqdm>=4.65.0` - Progress bars
- `requests>=2.31.0` - HTTP requests

### Logging & Monitoring (2 packages)
- `wandb==0.22.0` - Experiment tracking
- `tensorboard>=2.20.0` - TensorBoard logging

### Optimization (1 package)
- `bitsandbytes>=0.45.0` - 8-bit optimizers

### Additional (3 packages)
- `Jinja2>=3.1.0` - Templating
- `packaging>=23.0.0` - Version parsing
- `psutil>=5.9.0` - System monitoring

## Development Tools (requirements-dev.txt)

### Testing
- `pytest==7.2.0`
- `pytest-cov>=4.0.0`
- `pytest-xdist>=3.0.0`

### Code Quality
- `black==25.9.0` - Code formatter
- `flake8==7.3.0` - Linter
- `isort==6.0.1` - Import sorter
- `mypy==1.18.2` - Type checker

### Git Hooks
- `pre-commit==4.3.0`

After installation, set up pre-commit hooks:
```bash
pre-commit install
```

## Known Issues

### Issue 1: peft Import Error

**Symptom**: `cannot import name 'EncoderDecoderCache' from 'transformers'`

**Cause**: peft 0.17.1 expects transformers 4.46+, but we use 4.37.2 for stability.

**Solution**: Already handled by `utils/import_fix.py`

The project includes a compatibility patch in `utils/import_fix.py` that adds missing classes. This is automatically imported in `train_modular.py`:

```python
import utils.import_fix  # Applied before other imports
```

**Verification**:
```bash
# This should work
python -c "import utils.import_fix; import peft; print('OK')"
```

### Issue 2: PyTorch Version Mismatch

If you see CUDA-related errors, verify your PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output:
```
2.4.1+cu121  (or cu118, depending on your CUDA)
True
```

If False or version mismatch, reinstall PyTorch for your CUDA version (see section 2 above).

## System Requirements

- **Python**: 3.10+ (tested on 3.10)
- **CUDA**: 11.8+ (for GPU support)
- **VRAM**: 24GB+ recommended (minimum 16GB with optimizations)
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS (CPU only)

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-15 | 2.0 | Complete rewrite with proper versioning |
| Before | 1.0 | Original requirements (outdated) |

## Migration from Old requirements.txt

**Old version** (13 packages, many outdated):
```
accelerate==0.20.2  → 1.1.0 (Major update ✓)
diffusers==0.20.0   → 0.35.1 (Major update ✓)
transformers==4.30.2 → 4.37.2 (Minor update ✓)
xformers==0.0.22    → Removed (not used)
```

**New additions** (15 new packages):
- torch, peft, safetensors, wandb (critical!)
- timm, lpips, scikit-image (vision)
- scipy, matplotlib, opencv-python (utils)
- And 5 more...

**To migrate**:
```bash
# Backup old environment
pip freeze > old_requirements_backup.txt

# Install new requirements
pip install -r requirements.txt

# Verify
python -c "import utils.import_fix; import peft; print('Migration OK')"
```

## Troubleshooting

### Problem: Package conflicts

```bash
# Create fresh environment
conda create -n ppd python=3.10
conda activate ppd
pip install -r requirements.txt
```

### Problem: Out of memory during pip install

```bash
# Install in stages
pip install torch torchvision  # Heavy packages first
pip install -r requirements.txt  # Then the rest
```

### Problem: Pre-commit hooks failing

```bash
# Reinstall hooks
pre-commit clean
pre-commit install
pre-commit run --all-files
```

## Contact

For issues related to requirements, please check:
1. This file (README_REQUIREMENTS.md)
2. Project documentation in `docs/`
3. `.claude/CLAUDE.md` for development guidelines
4. GitHub issues

---

**Status**: ✅ Production Ready (2025-10-15)
