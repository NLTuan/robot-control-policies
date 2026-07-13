from lerobot.datasets.lerobot_dataset import LeRobotDataset

import torch
from torch.utils.data import DataLoader


def make_dataloader(config):
    dataset_cfg = config["dataset"]

    repo_id = dataset_cfg["repo_id"]
    batch_size = dataset_cfg["loader"]["batch_size"]
    obs_horizon = dataset_cfg["horizons"]["obs"]
    action_horizon = dataset_cfg["horizons"]["action"]

    state_key = dataset_cfg["keys"]["state"]
    action_key = dataset_cfg["keys"]["action"]
    image_keys = dataset_cfg["keys"]["images"]

    temp_dataset = LeRobotDataset(repo_id)
    dt = 1.0 / temp_dataset.fps

    delta_timestamps = {
        state_key: [i * dt for i in range(-obs_horizon + 1, 1)],
        action_key: [i * dt for i in range(action_horizon)],
    }

    for image_key in image_keys:
        delta_timestamps[image_key] = [i * dt for i in range(-obs_horizon + 1, 1)]

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    pin_memory = dataset_cfg["loader"].get("pin_memory", "auto")
    if pin_memory == "auto":
        pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset_cfg["loader"].get("shuffle", True),
        num_workers=dataset_cfg["loader"].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    return dataloader, dataset.features
