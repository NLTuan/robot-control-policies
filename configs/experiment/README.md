# Experiment Configs

Experiment configs compose one dataset config, one model config, and one training
config. Use these files for named runs you expect to repeat.

Use `overrides` for run-specific changes that should not live in the shared
dataset, model, or train config:

```yaml
defaults:
  - ../dataset/board_clean
  - ../model/mlp
  - ../train/default

experiment_name: mlp_board_clean

overrides:
  dataset:
    loader:
      batch_size: 64
  train:
    optimizer:
      lr: 0.0001
    output_dir: runs/mlp_board_clean
```

The loader composes defaults first, then applies the experiment file, then
applies `overrides`.

For quick one-off changes, pass dotlist overrides on the command line:

```bash
uv run python trainers/train_mlp.py \
  --config configs/experiment/mlp_board_clean.yaml \
  train.optimizer.lr=0.0001 \
  dataset.loader.batch_size=64 \
  train.steps.max_steps=100
```

Command-line overrides are applied last.
