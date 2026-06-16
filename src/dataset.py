from lerobot.datasets.lerobot_dataset import LeRobotDataset

import torch
from torch.utils.data import DataLoader


def make_dataloader(config):

    repo_id = config['dataset']['repo_id']
    batch_size = config['dataset']['loader']['batch_size']
    obs_horizon = config['dataset']['horizons']['obs']
    action_horizon = config['dataset']['horizons']['action']

    state_key = config['dataset']['keys']['state']
    action_key = config['dataset']['keys']['action']
    image_keys = config['dataset']['keys'].get('images', [])
    use_images = config['model']['inputs'].get('use_images', False)

    temp_dataset = LeRobotDataset(repo_id)
    fps = temp_dataset.fps
    dt = 1.0 / fps

    delta_timestamps = {
        state_key: [i * dt for i in range(-obs_horizon + 1, 1)],
        action_key: [i * dt for i in range(action_horizon)],
    }

    if use_images:
        for image_key in image_keys:
            delta_timestamps[image_key] = [i * dt for i in range(-obs_horizon + 1, 1)]

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    pin_memory = config['dataset']['loader'].get('pin_memory', 'auto')
    if pin_memory == 'auto':
        pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=config['dataset']['loader'].get('shuffle', True),
            num_workers=config['dataset']['loader'].get('num_workers', 0),
            pin_memory=pin_memory
        )
    
    return dataloader, dataset.features
