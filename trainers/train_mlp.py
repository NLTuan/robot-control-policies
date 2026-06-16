from src.dataset import make_dataloader
from src.config import load_config

import argparse

def train(config_path="configs/experiment/mlp_board_clean.yaml"):
    config = load_config(config_path)
    dataloader, features = make_dataloader(config)
    
    for batch in dataloader:
        print(batch)
        break

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/mlp_board_clean.yaml")
    args = parser.parse_args()
    train(args.config)
