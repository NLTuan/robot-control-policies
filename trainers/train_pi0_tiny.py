import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from robot_control_policies.config import load_config
from robot_control_policies.dataset import make_dataloader
from robot_control_policies.models.pi0 import Pi0Tiny


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


def _flatten_state(state):
    return state.float().flatten(start_dim=1)


def _action_chunk(action):
    action = action.float()
    if action.ndim == 2:
        action = action[:, None, :]
    if action.ndim != 3:
        raise ValueError(f"action must have shape [B, H, A], got {tuple(action.shape)}")
    return action


def _image_sequence(batch, image_keys, image_size):
    if not image_keys:
        return None

    images = []
    for key in image_keys:
        image = batch[key]
        if image.ndim == 4:
            image = image[:, None, :, :, :]
        if image.ndim != 5:
            raise ValueError(
                f"image key {key!r} must have shape [B, T, C, H, W], "
                f"got {tuple(image.shape)}"
            )

        if image.shape[-1] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
            image = image.permute(0, 1, 4, 2, 3)

        image = image.float()
        if image.max() > 2.0:
            image = image / 255.0

        batch_size, num_images, channels, height, width = image.shape
        if height != image_size or width != image_size:
            image = image.flatten(0, 1)
            image = F.interpolate(
                image,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            image = image.reshape(batch_size, num_images, channels, image_size, image_size)

        images.append(image)

    return torch.cat(images, dim=1)


def prepare_batch(batch, config):
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    state_key = dataset_cfg["keys"]["state"]
    action_key = dataset_cfg["keys"]["action"]
    image_keys = dataset_cfg["keys"]["images"]

    state = _flatten_state(batch[state_key])
    actions = _action_chunk(batch[action_key])
    images = _image_sequence(batch, image_keys, model_cfg["image_size"])
    return state, actions, images


def train(config_path="configs/experiment/overfit_pi0_tiny_board_clean.yaml", cli_overrides=None):
    config = load_config(config_path, cli_overrides=cli_overrides)
    torch.manual_seed(config["train"].get("seed", 0))
    dataloader, features = make_dataloader(config)
    first_batch = next(iter(dataloader))

    state, actions, images = prepare_batch(first_batch, config)
    device = get_device(config["train"])
    model_cfg = config["model"]

    model = Pi0Tiny(
        state_dim=state.shape[-1],
        action_dim=actions.shape[-1],
        action_horizon=actions.shape[1],
        hidden_dim=model_cfg["hidden_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        vocab_size=model_cfg["vocab_size"],
        image_channels=model_cfg["image_channels"],
        image_size=model_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        image_encoder_depth=model_cfg["image_encoder_depth"],
        max_action_horizon=model_cfg["max_action_horizon"],
    ).to(device)

    optimizer = make_optimizer(model, config["train"]["optimizer"])
    total_params, trainable_params = count_parameters(model)

    max_steps = config["train"]["steps"]["max_steps"]
    log_every = config["train"]["steps"]["log_every"]
    save_every = config["train"]["steps"].get("save_every", 0)
    overfit_one_batch = config["train"].get("overfit_one_batch", False)
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"features: {features}")
    print(f"state shape: {tuple(state.shape)}")
    print(f"action shape: {tuple(actions.shape)}")
    print(f"image shape: {None if images is None else tuple(images.shape)}")
    print(f"device: {device}")
    print(f"parameters: total={total_params:,} trainable={trainable_params:,}")

    step = 0
    while step < max_steps:
        for batch in dataloader:
            if overfit_one_batch:
                batch = first_batch

            state, actions, images = prepare_batch(batch, config)
            state = state.to(device)
            actions = actions.to(device)
            images = images.to(device) if images is not None else None

            loss = model.compute_loss(state, actions, images=images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"step={step} loss={loss.item():.6f}")

            step += 1
            if save_every and step % save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                    },
                    checkpoint_path,
                )
                print(f"saved checkpoint: {checkpoint_path}")

            if step >= max_steps:
                break

    final_path = output_dir / "final.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        final_path,
    )
    print(f"saved checkpoint: {final_path}")

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/overfit_pi0_tiny_board_clean.yaml")
    args, cli_overrides = parser.parse_known_args()
    train(args.config, cli_overrides=cli_overrides)
