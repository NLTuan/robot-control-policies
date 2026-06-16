from lerobot.datasets.lerobot_dataset import LeRobotDataset

import torch
from torch.utils.data import DataLoader


def make_dataloader(config):

    repo_id = config['dataset']['repo_id']
    batch_size = config['dataset']['batch_size']
    obs_horizon = config['dataset']['obs_horizon']
    pred_horizon = config['dataset']['pred_horizon']
    
    image_key = config['model']['image_key']

    temp_dataset = LeRobotDataset(repo_id)
    fps = temp_dataset.fps
    dt = 1.0 / fps

    delta_timestamps = {
        'observation.state': [i * dt for i in range(-obs_horizon + 1, 1)],
        'action': [i * dt for i in range(pred_horizon)],
    }

    delta_timestamps[image_key] = [i * dt for i in range(-obs_horizon + 1, 1)]

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=config['dataset'].get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    return dataloader, dataset.features

