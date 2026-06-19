import argparse

import torch
from torch import nn

from robot_control_policies.config import load_config
from robot_control_policies.dataset import make_dataloader
from robot_control_policies.models.mlp import MLPPolicy


def get_device(train_cfg):
    if train_cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(train_cfg["device"])


def make_optimizer(model, optimizer_cfg):
    if optimizer_cfg["name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def flatten_batch(batch, config):
    state_key = config["dataset"]["keys"]["state"]
    action_key = config["dataset"]["keys"]["action"]

    state = batch[state_key].float().flatten(start_dim=1)
    action = batch[action_key].float().flatten(start_dim=1)
    return state, action


def train(config_path="configs/experiment/mlp_board_clean.yaml"):
    config = load_config(config_path)
    dataloader, features = make_dataloader(config)

    first_batch = next(iter(dataloader))
    x, y = flatten_batch(first_batch, config)
    device = get_device(config["train"])

    model = MLPPolicy(
        input_dim=x.shape[-1],
        output_dim=y.shape[-1],
        hidden_dims=config["model"]["hidden_dims"],
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    optimizer = make_optimizer(model, config["train"]["optimizer"])
    loss_fn = nn.MSELoss()

    max_steps = config["train"]["steps"]["max_steps"]
    log_every = config["train"]["steps"]["log_every"]
    overfit_one_batch = config["train"].get("overfit_one_batch", False)

    print(f"features: {features}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"target shape: {tuple(y.shape)}")
    print(f"device: {device}")
    print(f"parameters: total={total_params:,} trainable={trainable_params:,}")

    step = 0
    while step < max_steps:
        for batch in dataloader:
            if overfit_one_batch:
                batch = first_batch

            x, y = flatten_batch(batch, config)
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"step={step} loss={loss.item():.6f}")

            step += 1
            if step >= max_steps:
                break

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/mlp_board_clean.yaml")
    args = parser.parse_args()
    train(args.config)
