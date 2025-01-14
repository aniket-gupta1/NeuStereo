import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from NeuStereo.neustereo import NeuStereo
from utils import prepare_logger

def setup_dataloaders():
    pass

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print the entire config
    print(OmegaConf.to_yaml(cfg))

    # Create necessary directories
    logdir = Path(cfg.logdir) / cfg.exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    # Prepare logger
    logger, log_path = prepare_logger(cfg)
    logger.info(f"Logging to {log_path}")

    # Save config for reproducibility
    config_out_fname = logdir / "config.yaml"
    with open(config_out_fname, "w") as out_fid:
        out_fid.write(OmegaConf.to_yaml(cfg))

    # Dataset loading
    train_dataset = build_train_dataset(cfg.dataset.path, cfg.dataset.train_split)
    val_dataset = build_train_dataset(cfg.dataset.path, cfg.dataset.val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Model setup
    model = NeuStereo(cfg)
    logger.info("Model initialized.")

    # Training process (placeholder)
    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        # Training loop
        # Validation loop

if __name__ == "__main__":
    main()
