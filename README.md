# Robot Control Policies

A clean, high-iteration codebase structure for replicating robot control policies (like MLP, ACT, or Diffusion Policies) integrated with Hugging Face's `lerobot` datasets.

---

## 📂 Codebase Layout

*   `configs/`: Directory containing model configuration files (e.g. `configs/mlp.yaml`).
*   `src/`: Unified core source code (dataset loaders, environment wrapper, shared CNN encoders).
*   `trainers/`: Architecture-specific training loops (e.g. `trainers/train_mlp.py`) where models are optimized.
*   `pyproject.toml`: Environment configurations managed via the `uv` package manager.

---

## 🚀 Execution Guide

### 1. Installation

Sync your virtual environment and install the codebase in editable mode:
```bash
# Initialize uv virtual environment
uv venv
source .venv/bin/activate

# Install the packages in editable mode
uv pip install -e .
```

Editable installation allows any script in `trainers/` to import from `src/` out-of-the-box:
```python
from src.dataset import RobotDataset
```

### 2. Run MLP Training

Execute training (which defaults to a mock dataset fallback if the Hugging Face `lerobot` library is missing or running offline):
```bash
python trainers/train_mlp.py --config configs/mlp.yaml
```

Checkpoints will be saved in `checkpoints/` and Tensorboard telemetry logs inside `logs/`.

---

## 🛠️ Adding a New Architecture

To build a new policy (e.g. a Transformer or Diffusion Policy):

1.  Write your new policy network class inside `src/models.py`.
2.  Create a custom config file inside `configs/` (e.g. `configs/diffusion.yaml`).
3.  Duplicate `trainers/train_mlp.py` to `trainers/train_diffusion.py` and customize/optimize the training steps and loss functions directly (e.g. introducing EMA updates, noise schedulers, or custom hardware compilation parameters).
