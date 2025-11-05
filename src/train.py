import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Robust config import: use src.config when available and provide fallbacks
try:
    import src.config as cfg
except Exception:
    import config as cfg  # fallback if running from within src
# fallback-safe getters
PHONE_DATASET_PATH = getattr(cfg, "PHONE_DATASET_PATH", None)
MERGED_MEASUREMENT_TRAINING = getattr(cfg, "MERGED_MEASUREMENT_TRAINING", None)
MERGED_MEASUREMENT_TEST = getattr(cfg, "MERGED_MEASUREMENT_TEST", None)
SAVE_DIR = getattr(cfg, "SAVE_DIR", "./results/checkpoints")
ACCUMULATION_STEPS = getattr(cfg, "ACCUMULATION_STEPS", 4)
CACHE_SIZE = getattr(cfg, "CACHE_SIZE", 30000)
BATCH_SIZE = getattr(cfg, "BATCH_SIZE", 16)
NUM_WORKERS = getattr(cfg, "NUM_WORKERS", 2)

# Local imports (data, model, training loop)
from src.preprocess import (
    load_and_preprocess_data,
    load_phone_dimensions,
    ConcatenatedImageDataset,
    custom_collate,
)
from src.model import EnhancedResNet101
from src.training import train_and_evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data CSVs (must exist)
    if MERGED_MEASUREMENT_TRAINING is None or MERGED_MEASUREMENT_TEST is None:
        raise ValueError(
            "Please set MERGED_MEASUREMENT_TRAINING and MERGED_MEASUREMENT_TEST in config.py"
        )

    train_measurement_df, measurement_float = load_and_preprocess_data(
        MERGED_MEASUREMENT_TRAINING
    )
    test_measurement_df, _ = load_and_preprocess_data(MERGED_MEASUREMENT_TEST)

    # Transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Load phone dimensions (optional file)
    phone_dimensions = None
    if PHONE_DATASET_PATH:
        phone_dimensions = load_phone_dimensions(PHONE_DATASET_PATH)
    if not phone_dimensions:
        print("Warning: No valid phone dimensions loaded. Using default placeholder.")
        phone_dimensions = [("Default", 150.0, 75.0, 8.0)]

    # Image directories: prefer config values, otherwise fall back to old defaults
    train_front_img_dir = getattr(
        cfg, "TRAIN_FRONT_IMG_DIR", "/content/drive/MyDrive/bodymeasurement/train/mask"
    )
    train_side_img_dir = getattr(
        cfg,
        "TRAIN_SIDE_IMG_DIR",
        "/content/drive/MyDrive/bodymeasurement/train/mask_left",
    )
    test_front_img_dir = getattr(
        cfg, "TEST_FRONT_IMG_DIR", "/content/drive/MyDrive/bodymeasurement/testA/mask"
    )
    test_side_img_dir = getattr(
        cfg,
        "TEST_SIDE_IMG_DIR",
        "/content/drive/MyDrive/bodymeasurement/testA/mask_left",
    )

    print("Creating datasets...")
    train_dataset = ConcatenatedImageDataset(
        train_front_img_dir,
        train_side_img_dir,
        train_measurement_df,
        measurement_float,
        transform=transform,
        cache_size=CACHE_SIZE,
    )
    test_dataset = ConcatenatedImageDataset(
        test_front_img_dir,
        test_side_img_dir,
        test_measurement_df,
        measurement_float,
        transform=transform,
        cache_size=CACHE_SIZE,
    )

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
    )

    # Initialize model
    print("Initializing model...")
    enhanced_resnet101 = EnhancedResNet101(measurement_float).to(device)

    # Ensure save dir exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Train
    best_mae = train_and_evaluate(
        model=enhanced_resnet101,
        train_loader=train_loader,
        test_loader=test_loader,
        measurement_float=measurement_float,
        train_measurement_df=train_measurement_df,
        save_dir=SAVE_DIR,
        accumulation_steps=ACCUMULATION_STEPS,
        resume=False,
        epochs=getattr(cfg, "EPOCHS", 200),
    )

    print(f"Training finished. Best test MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main()
