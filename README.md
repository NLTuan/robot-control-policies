# Robot Control Policies

This repo is a small training harness for learning and replicating robot control
policies. The current working path is a state-only MLP baseline trained from a
LeRobot dataset. The structure is intentionally simple so new models and
datasets can be added one at a time.

DreamZero-style world/action models are a later target. For now, the repo is set
up to make the basics reliable: config loading, dataset loading, model creation,
and a minimal training loop.

## Project Layout

```text
robot-control-policies/
  configs/
    dataset/        # data source, keys, horizons, dataloader settings
    model/          # model architecture settings
    train/          # optimizer, runtime, logging, step counts
    experiment/     # named combinations of dataset + model + train configs

  robot_control_policies/
    config.py       # lightweight YAML config composer
    dataset.py      # LeRobot dataloader construction
    models/
      mlp/          # current working MLP policy
      pi0/          # placeholder area for future pi0-style policy work

  trainers/
    train_mlp.py    # current MLP training entrypoint
```

## Config System

Configs are split by responsibility.

- Dataset configs answer: what data am I training on?
- Model configs answer: what network am I building?
- Train configs answer: how do I optimize and run it?
- Experiment configs answer: which dataset/model/train combination is this run?

The config loader in `robot_control_policies/config.py` supports a small
`defaults` list. For example:

```yaml
# configs/experiment/mlp_board_clean.yaml
defaults:
  - ../dataset/board_clean
  - ../model/mlp
  - ../train/local

experiment_name: mlp_board_clean_state_only
```

When loaded, those files are merged into one Python dictionary:

```python
from robot_control_policies.config import load_config

config = load_config("configs/experiment/mlp_board_clean.yaml")
```

Later files override earlier defaults if they define the same field.

## Dataset Configs

Example: `configs/dataset/board_clean.yaml`

```yaml
dataset:
  name: board_clean
  repo_id: NLTuan/board_clean

  keys:
    state: observation.state
    action: action
    images:
      - observation.images.main

  horizons:
    obs: 1
    action: 50

  loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: auto
```

Important fields:

- `repo_id`: LeRobot/Hugging Face dataset id or path.
- `keys.state`: dataset column used as model input.
- `keys.action`: dataset column used as the training target.
- `keys.images`: image columns available for future vision policies.
- `horizons.obs`: number of observation timesteps requested.
- `horizons.action`: number of future action timesteps requested.
- `loader`: passed into the PyTorch `DataLoader`.

The current MLP uses state only. Image keys are kept in the dataset config so
future vision models can request them by setting `model.inputs.use_images: true`.

## Model Configs

Example: `configs/model/mlp.yaml`

```yaml
model:
  name: mlp
  type: mlp_policy

  inputs:
    use_state: true
    use_images: false

  hidden_dims:
    - 256
    - 256
```

Important fields:

- `name`: short model name used in experiment naming.
- `type`: model family. The MLP trainer currently expects `mlp_policy`.
- `inputs`: which data modalities the model consumes.
- `hidden_dims`: layer widths passed into `MLPPolicy`.

Keep model YAMLs descriptive. The Python class itself should not load YAML. The
trainer or a future model builder should read config and instantiate the model.

## Train Configs

Example: `configs/train/overfit.yaml`

```yaml
train:
  name: overfit
  seed: 42
  device: auto
  precision: fp32

  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.0

  steps:
    max_steps: 1000
    log_every: 1
    save_every: 250
    eval_every: 250

  overfit_one_batch: true
  output_dir: runs/overfit
```

Important fields:

- `device: auto`: uses CUDA if available, otherwise CPU.
- `optimizer`: currently supports `adamw`.
- `steps.max_steps`: number of optimizer steps.
- `steps.log_every`: print loss every N steps.
- `overfit_one_batch`: repeatedly trains on the first batch. This is useful for
  checking whether the model and loss can learn at all.
- `output_dir`: reserved for checkpoints/logs; checkpointing is not wired yet.

## Current Training Flow

The current entrypoint is:

```bash
uv run python trainers/train_mlp.py --config configs/experiment/mlp_board_clean.yaml
```

For an overfit test:

```bash
uv run python trainers/train_mlp.py --config configs/experiment/overfit_mlp_board_clean.yaml
```

`train_mlp.py` does this:

1. Loads and composes the experiment config.
2. Builds a LeRobot dataloader from `robot_control_policies/dataset.py`.
3. Reads the first batch to infer `input_dim` and `output_dim`.
4. Builds `MLPPolicy` from `robot_control_policies/models/mlp/mlp_policy.py`.
5. Creates an AdamW optimizer from the train config.
6. Trains with MSE loss.

The current MLP flattens:

```text
observation.state -> model input
action            -> model target
```

So a state sequence of shape `[B, obs_horizon, state_dim]` becomes
`[B, obs_horizon * state_dim]`, and an action chunk becomes
`[B, action_horizon * action_dim]`.

## Adding a New Dataset

1. Create a file under `configs/dataset/`, for example:

```text
configs/dataset/my_robot.yaml
```

2. Fill in the LeRobot repo/path and correct column keys.
3. Create a new experiment config:

```yaml
defaults:
  - ../dataset/my_robot
  - ../model/mlp
  - ../train/overfit

experiment_name: overfit_mlp_my_robot
```

4. Run the overfit experiment first.

## Adding a New Model

Add model code under:

```text
robot_control_policies/models/<model_name>/
```

Then add a matching config under:

```text
configs/model/<model_name>.yaml
```

Do not make the model class load YAML. Keep the split:

```text
YAML config     -> describes the model
trainer/builder -> reads config and creates the model
model class     -> pure PyTorch module
```

Once there are two real models, add a small model builder instead of putting all
model selection logic directly into each trainer.

## Development Notes

- Start new work with `configs/train/overfit.yaml`. If a model cannot overfit
  one batch, the full training run is not worth debugging yet.
- Keep config files small and honest. Avoid adding placeholder model configs
  before the matching model code exists.
- Dataset keys should come from YAML, not hardcoded trainer strings.
- Normalization is listed in dataset configs but not implemented yet.
- Checkpointing, evaluation, scheduler behavior, and mixed precision are planned
  but not fully wired into the MLP trainer yet.
