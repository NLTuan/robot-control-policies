import os
import yaml
import torch
import collections
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import RobotDataset
from src.models import MLPPolicy
from src.env_wrapper import EnvWrapper

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(policy, config, device, epoch):
    """
    Evaluates policy by rolling out trajectories in the wrapped environment.
    """
    env_name = config["eval"]["env_name"]
    num_episodes = config["eval"]["num_episodes"]
    obs_horizon = config["dataset"]["obs_horizon"]
    action_horizon = config["dataset"]["action_horizon"]
    image_key = config["model"]["image_key"]
    
    print(f"\n[EVAL] Starting rollout evaluation for epoch {epoch}...")
    env = EnvWrapper(env_name, config)
    success_count = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        state_buffer = collections.deque(maxlen=obs_horizon)
        image_buffer = collections.deque(maxlen=obs_horizon)
        
        # Initialize temporal context buffer by repeating initial frame
        for _ in range(obs_horizon):
            state_buffer.append(obs["observation.state"])
            if config["model"]["use_visual"]:
                image_buffer.append(obs[image_key])
                
        episode_reward = 0
        done = False
        step = 0
        info = {}
        
        while not done and step < 200:
            policy_input = {
                "observation.state": torch.stack(list(state_buffer)).to(device)
            }
            if config["model"]["use_visual"]:
                policy_input[image_key] = torch.stack(list(image_buffer)).to(device)
                
            # Predict action chunk
            actions = policy.get_action(policy_input).cpu().numpy()
            
            # Execute action horizon slice
            for i in range(action_horizon):
                action = actions[i]
                obs, reward, done, info = env.step(action)
                
                state_buffer.append(obs["observation.state"])
                if config["model"]["use_visual"]:
                    image_buffer.append(obs[image_key])
                    
                episode_reward += reward
                step += 1
                if done:
                    break
                    
        if info.get("success", False) or episode_reward > 0:
            success_count += 1
            
    success_rate = success_count / num_episodes
    print(f"[EVAL] Success Rate: {success_rate * 100:.1f}%\n")
    return success_rate

def train(config_path):
    config = load_config(config_path)
    
    # Setup directories
    os.makedirs(config["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    
    writer = SummaryWriter(log_dir=config["train"]["log_dir"])
    print("[TRAIN] TensorBoard initialized.")
    
    # Configure Device
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")
    
    # Setup Dataset & Loader
    dataset = RobotDataset(
        repo_id=config["dataset"]["repo_id"],
        obs_horizon=config["dataset"]["obs_horizon"],
        pred_horizon=config["dataset"]["pred_horizon"],
        use_visual=config["model"]["use_visual"],
        image_key=config["model"]["image_key"]
    )
    
    config["model"]["state_dim"] = dataset.state_dim
    config["model"]["action_dim"] = dataset.action_dim
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"] if not dataset.is_mock else 0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize MLP Model & Optimizer
    policy = MLPPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"])
    )
    
    epochs = config["train"]["epochs"]
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        sample_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress_bar:
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v
            
            optimizer.zero_grad()
            loss = policy(device_batch)
            loss.backward()
            optimizer.step()
            
            bs = len(device_batch["action"])
            epoch_loss += loss.item() * bs
            sample_count += bs
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("Loss/step", loss.item(), global_step)
            global_step += 1
            
        avg_loss = epoch_loss / sample_count
        print(f"[TRAIN] Epoch {epoch} Loss: {avg_loss:.5f}")
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        
        # Run Evaluation Rollouts
        if epoch % config["train"]["eval_freq"] == 0 or epoch == epochs:
            success_rate = run_evaluation(policy, config, device, epoch)
            writer.add_scalar("Eval/SuccessRate", success_rate, epoch)
            
        # Save Model Checkpoint
        if epoch % config["train"]["save_freq"] == 0 or epoch == epochs:
            checkpoint_path = os.path.join(
                config["train"]["checkpoint_dir"],
                f"mlp_policy_epoch_{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"[TRAIN] Checkpoint saved: {checkpoint_path}")
            
    print("[TRAIN] Training process completed successfully!")
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mlp.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
