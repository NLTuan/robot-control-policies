import os
import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

class RobotDataset(Dataset):
    """
    Wrapper around LeRobotDataset.
    If 'lerobot' is missing, it falls back to generating synthetic sequence data.
    """
    def __init__(self, repo_id, obs_horizon=2, pred_horizon=16, use_visual=True, image_key="observation.image"):
        super().__init__()
        self.repo_id = repo_id
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_visual = use_visual
        self.image_key = image_key
        
        self.is_mock = True
        if LEROBOT_AVAILABLE:
            print(f"[DATASET] Loading LeRobot dataset: {repo_id}...")
            try:
                # Initialize once to query properties
                self.lerobot_dataset = LeRobotDataset(repo_id=repo_id)
                self.fps = self.lerobot_dataset.fps if hasattr(self.lerobot_dataset, 'fps') else 10
                dt = 1.0 / self.fps
                
                # Setup delta timestamps
                delta_timestamps = {
                    "observation.state": [t * dt for t in range(-obs_horizon + 1, 1)],
                    "action": [t * dt for t in range(pred_horizon)]
                }
                if self.use_visual:
                    delta_timestamps[self.image_key] = [t * dt for t in range(-obs_horizon + 1, 1)]
                
                # Load LeRobotDataset with delta timestamps configuration
                self.lerobot_dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)
                self.is_mock = False
                print("[DATASET] LeRobot dataset successfully loaded.")
            except Exception as e:
                print(f"[DATASET] Warning: Failed to load LeRobot dataset ({e}). Falling back to mock dataset.")
        else:
            print("[DATASET] Warning: 'lerobot' not found. Falling back to mock dataset.")
            
        if self.is_mock:
            self.num_samples = 100
            self.state_dim = 2
            self.action_dim = 2
            self.image_shape = (3, 96, 96)
        else:
            self.num_samples = len(self.lerobot_dataset)
            sample0 = self.lerobot_dataset[0]
            self.state_dim = sample0["observation.state"].shape[-1]
            self.action_dim = sample0["action"].shape[-1]
            if self.use_visual and self.image_key in sample0:
                self.image_shape = sample0[self.image_key].shape[1:] 
            else:
                self.image_shape = (3, 96, 96)
                
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.is_mock:
            states = torch.randn(self.obs_horizon, self.state_dim)
            actions = torch.randn(self.pred_horizon, self.action_dim)
            sample = {
                "observation.state": states,
                "action": actions
            }
            if self.use_visual:
                images = torch.sigmoid(torch.randn(self.obs_horizon, *self.image_shape))
                sample[self.image_key] = images
            return sample
            
        raw_sample = self.lerobot_dataset[idx]
        sample = {
            "observation.state": raw_sample["observation.state"],
            "action": raw_sample["action"]
        }
        if self.use_visual and self.image_key in raw_sample:
            img = raw_sample[self.image_key]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            sample[self.image_key] = img
            
        return sample
