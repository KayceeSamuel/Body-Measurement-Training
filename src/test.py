import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from model import EnhancedResNet101, MeasurementNormalizer
from preprocess import (
    load_and_preprocess_data,
    ConcatenatedImageDataset,
    custom_collate,
)
from torch.utils.data import DataLoader
from torchvision import transforms


def test_model(model, test_loader, measurement_float, train_measurement_df):
    """
    Evaluate the model on test data and compute per-measurement MAE.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        measurement_float: List of measurement column names
        train_measurement_df: Training DataFrame used for normalization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initialize the normalizer with training data
    normalizer = MeasurementNormalizer(train_measurement_df, measurement_float)

    test_mae = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, gender, phone_info in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            gender = gender.to(device)
            phone_info = phone_info.to(device)

            # Model inference
            outputs = model(inputs, phone_info, gender)

            # Denormalize for real-world MAE computation
            denormalized_outputs = normalizer.denormalize(
                outputs,
                [m for m in measurement_float if m not in ["height_cm", "weight_kg"]],
            )

            test_mae += F.l1_loss(denormalized_outputs, targets).item()

            all_predictions.extend(denormalized_outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_test_mae = test_mae / len(test_loader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    measurement_names = [
        m for m in measurement_float if m not in ["height_cm", "weight_kg"]
    ]
    per_measurement_mae = {
        name: np.mean(np.abs(all_predictions[:, i] - all_targets[:, i]))
        for i, name in enumerate(measurement_names)
    }

    print(f"\n Overall Test MAE: {avg_test_mae:.4f}")
    print("\n Per-Measurement MAE:")
    for name, mae in per_measurement_mae.items():
        print(f"{name}: {mae:.4f}")

    return avg_test_mae, per_measurement_mae, all_predictions, all_targets


if __name__ == "__main__":
    # Paths
    save_dir = "/content/drive/MyDrive/bodymeasurement/train/checkpoints_adv2"
    checkpoint_path = os.path.join(save_dir, "best_model.pth")

    # Load data
    train_measurement_df, measurement_float = load_and_preprocess_data(
        "/content/drive/MyDrive/bodymeasurement/train/merged_measurements.csv"
    )
    test_measurement_df, _ = load_and_preprocess_data(
        "/content/drive/MyDrive/bodymeasurement/testA/merged_measurements.csv"
    )

    transform = transforms.Compose([transforms.ToTensor()])

    test_front_img_dir = "/content/drive/MyDrive/bodymeasurement/testA/mask"
    test_side_img_dir = "/content/drive/MyDrive/bodymeasurement/testA/mask_left"

    test_dataset = ConcatenatedImageDataset(
        test_front_img_dir,
        test_side_img_dir,
        test_measurement_df,
        measurement_float,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate,
    )

    # Load best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedResNet101(measurement_float).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run test
    mae, per_measurement_mae, predictions, targets = test_model(
        model=model,
        test_loader=test_loader,
        measurement_float=measurement_float,
        train_measurement_df=train_measurement_df,
    )
