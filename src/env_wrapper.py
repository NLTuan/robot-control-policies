import torch
import numpy as np

class EnvWrapper:
    """
    Standard interface wrapping simulation or real robot environment loop.
    """
    def __init__(self, env_name, config):
        self.env_name = env_name
        self.config = config
        self.state_dim = config["model"]["state_dim"]
        self.use_visual = config["model"]["use_visual"]
        self.image_key = config["model"].get("image_key", "observation.image")
        
    def reset(self):
        """
        Resets environment.
        returns: initial observation dict containing state (and image).
        """
        obs = {
            "observation.state": torch.zeros(self.state_dim, dtype=torch.float32)
        }
        if self.use_visual:
            obs[self.image_key] = torch.zeros((3, 96, 96), dtype=torch.float32)
        return obs
        
    def step(self, action):
        """
        Applies action in the environment.
        returns: obs (dict), reward (float), done (bool), info (dict)
        """
        obs = {
            "observation.state": torch.zeros(self.state_dim, dtype=torch.float32)
        }
        if self.use_visual:
            obs[self.image_key] = torch.zeros((3, 96, 96), dtype=torch.float32)
            
        reward = 0.0
        done = False
        info = {"success": True}
        return obs, reward, done, info
