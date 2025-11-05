import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from .model import WeightedMeasurementLoss, MeasurementNormalizer, get_optimizer
import csv
import time
from tqdm import tqdm
from functools import lru_cache
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    measurement_float,
    train_measurement_df,
    save_dir="/content/drive/MyDrive/bodymeasurement/train/checkpoints_adv2",
    accumulation_steps=4,
    resume=False,
    epochs=200,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize optimizer, loss functions, and normalizer
    optimizer = get_optimizer(model)
    weighted_loss = WeightedMeasurementLoss(
        [m for m in measurement_float if m not in ["height_cm", "weight_kg"]]
    )
    normalizer = MeasurementNormalizer(train_measurement_df, measurement_float)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = os.path.join(save_dir, "training_log.csv")
    log_exists = os.path.exists(log_file)

    best_mae = float("inf")
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
    if resume and os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    print("Starting training...")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["Epoch", "Train Loss", "Train MAE", "Test MAE", "Time"])

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            optimizer.zero_grad()

            for batch_idx, (inputs, targets, gender, phone_info) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            ):
                inputs = inputs.to(device)
                targets = targets.to(device)
                gender = gender.to(device)
                phone_info = phone_info.to(device)

                # Normalize targets
                normalized_targets = normalizer.normalize(
                    targets,
                    [
                        m
                        for m in measurement_float
                        if m not in ["height_cm", "weight_kg"]
                    ],
                )

                outputs = model(inputs, phone_info, gender)
                loss = weighted_loss(outputs, normalized_targets)
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(
                    train_loader
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                # Denormalize for MAE calculation
                denormalized_outputs = normalizer.denormalize(
                    outputs,
                    [
                        m
                        for m in measurement_float
                        if m not in ["height_cm", "weight_kg"]
                    ],
                )
                train_loss += loss.item() * accumulation_steps
                train_mae += (
                    F.l1_loss(denormalized_outputs, targets).item() * accumulation_steps
                )

            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)

            model.eval()
            test_mae = 0.0
            with torch.no_grad():
                for inputs, targets, gender, phone_info in tqdm(
                    test_loader, desc="Evaluating"
                ):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    gender = gender.to(device)
                    phone_info = phone_info.to(device)

                    outputs = model(inputs, phone_info, gender)
                    denormalized_outputs = normalizer.denormalize(
                        outputs,
                        [
                            m
                            for m in measurement_float
                            if m not in ["height_cm", "weight_kg"]
                        ],
                    )
                    test_mae += F.l1_loss(denormalized_outputs, targets).item()

            current_test_mae = test_mae / len(test_loader)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Test MAE: {current_test_mae:.4f}"
            )
            print(f"Time: {epoch_time:.2f}s")
            # Log results
            writer.writerow(
                [epoch + 1, avg_train_loss, avg_train_mae, current_test_mae, epoch_time]
            )
            f.flush()
            if current_test_mae < best_mae:
                best_mae = current_test_mae
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mae": best_mae,
                    },
                    os.path.join(save_dir, "best_model.pth"),
                )
                print(f"New best model saved with MAE: {best_mae:.4f}")
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mae": best_mae,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at epoch {epoch + 1}")
    return best_mae
