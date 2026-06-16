# Config Layout

Configs are split by what you want to swap independently.

- `dataset/`: dataset location, keys, horizons, loader settings, normalization.
- `model/`: architecture and model input choices.
- `train/`: optimizer, schedule, precision, logging, checkpoint cadence.
- `experiment/`: a concrete combination of dataset, model, and train configs.

Start with one working experiment, then add new YAML files only when you add the
matching code.
