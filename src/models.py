import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    Lightweight CNN visual backbone that compresses raw camera inputs into feature vectors.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # -> 48x48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 6x6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # -> 1x1
            nn.Flatten() # -> 128
        )
        self.fc = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        # Input shape: (B, C, H, W)
        return self.fc(self.conv(x))

class MLP(nn.Module):
    """
    Standard fully-connected Multi-Layer Perceptron layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class BasePolicy(nn.Module):
    """
    Base class for robot control policies.
    """
    def forward(self, batch):
        raise NotImplementedError
        
    def get_action(self, obs):
        raise NotImplementedError

class MLPPolicy(BasePolicy):
    """
    MLP policy mapping historical observations (states + visual features) to action sequences.
    """
    def __init__(self, config):
        super().__init__()
        self.obs_horizon = config["dataset"]["obs_horizon"]
        self.pred_horizon = config["dataset"]["pred_horizon"]
        self.state_dim = config["model"]["state_dim"]
        self.action_dim = config["model"]["action_dim"]
        self.use_visual = config["model"]["use_visual"]
        self.image_key = config["model"].get("image_key", "observation.image")
        
        # Hidden dims & layers
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        
        # Calculate context size
        if self.use_visual:
            self.visual_feature_dim = config["model"]["visual_feature_dim"]
            self.visual_encoder = VisualEncoder(feature_dim=self.visual_feature_dim)
            obs_dim = (self.state_dim + self.visual_feature_dim) * self.obs_horizon
        else:
            obs_dim = self.state_dim * self.obs_horizon
            
        action_dim_total = self.action_dim * self.pred_horizon
        
        self.mlp = MLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim_total,
            num_layers=num_layers
        )
        
    def _process_obs(self, batch):
        from einops import rearrange
        states = batch["observation.state"] # (B, obs_horizon, state_dim)
        
        features = [rearrange(states, 'b t d -> b (t d)')]
        
        if self.use_visual:
            images = batch[self.image_key] # (B, obs_horizon, C, H, W)
            images_reshaped = rearrange(images, 'b t c h w -> (b t) c h w')
            img_features = self.visual_encoder(images_reshaped) # (B * T, visual_feature_dim)
            img_features = rearrange(img_features, '(b t) d -> b (t d)', t=self.obs_horizon)
            features.append(img_features)
            
        obs_features = torch.cat(features, dim=-1)
        return obs_features
        
    def forward(self, batch):
        """
        Computes MSE loss for training.
        """
        from einops import rearrange
        obs_features = self._process_obs(batch)
        pred_actions_flat = self.mlp(obs_features)
        pred_actions = rearrange(pred_actions_flat, 'b (t d) -> b t d', t=self.pred_horizon)
        
        gt_actions = batch["action"] # (B, pred_horizon, action_dim)
        loss = nn.functional.mse_loss(pred_actions, gt_actions)
        return loss
        
    def get_action(self, obs):
        """
        Predicts action horizon sequence.
        """
        from einops import rearrange
        self.eval()
        with torch.no_grad():
            batch = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    # Add batch dimension if missing using unsqueeze
                    if k == "observation.state" and len(v.shape) == 2:
                        batch[k] = v.unsqueeze(0)
                    elif k == self.image_key and len(v.shape) == 4:
                        batch[k] = v.unsqueeze(0)
                    else:
                        batch[k] = v
                else:
                    batch[k] = v
                    
            obs_features = self._process_obs(batch)
            pred_actions_flat = self.mlp(obs_features)
            pred_actions = rearrange(pred_actions_flat, '1 (t d) -> t d', t=self.pred_horizon)
        return pred_actions
