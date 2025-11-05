import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


# Model definition
class EnhancedResNet101(nn.Module):
    """
    Model definitions for body measurement prediction.
    Includes:
    - EnhancedResNet101 (main model)
    - WeightedMeasurementLoss (custom loss)
    - MeasurementNormalizer (scaling utility)
    """

    def __init__(self, measurement_float):
        super().__init__()
        self.base_model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        )
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Explicitly define the number of additional features
        self.num_phone_dims = 3  # length, width, depth
        self.num_front_rect = 4  # x1, y1, x2, y2
        self.num_side_rect = 4  # x1, y1, x2, y2
        self.num_gender = 1

        total_additional_features = (
            self.num_phone_dims
            + self.num_front_rect
            + self.num_side_rect
            + self.num_gender
        )

        # Add intermediate layers for better feature extraction
        self.fc1 = nn.Linear(
            self.base_model.fc.in_features + total_additional_features, 1024
        )
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        # Final layer for predictions
        self.fc3 = nn.Linear(512, len(measurement_float))

        # Store measurement names for potential use
        self.measurement_names = measurement_float

    def forward(self, x, phone_info, gender):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, phone_info, gender.unsqueeze(1)], dim=1)

        # Enhanced feature processing
        x = self.dropout1(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)

        return x


def get_optimizer(model):
    # Group parameters by layer type
    backbone_params = []
    head_params = []

    # Backbone (ResNet) parameters
    for name, param in model.features.named_parameters():
        backbone_params.append(param)

    # Head (FC layers) parameters
    head_params = (
        list(model.fc1.parameters())
        + list(model.bn1.parameters())
        + list(model.fc2.parameters())
        + list(model.bn2.parameters())
        + list(model.fc3.parameters())
    )

    # Create optimizer with different learning rates
    optimizer = optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": 1e-5,
            },  # Slower learning for pretrained layers
            {"params": head_params, "lr": 5e-4},  # Faster learning for new layers
        ],
        weight_decay=0.0001,
    )
    return optimizer


class WeightedMeasurementLoss(nn.Module):
    def __init__(self, measurement_names):
        super().__init__()
        # Define weights based on typical measurement ranges
        self.weights = {
            "height": 0.1,
            "shoulder-breadth": 0.5,
            "chest": 0.3,
            "waist": 0.3,
            "hip": 0.3,
            "leg-length": 0.2,
            "arm-length": 0.5,
            "shoulder-to-crotch": 0.5,
            "thigh": 0.8,
            "bicep": 1.0,
            "forearm": 1.0,
            "wrist": 1.2,
            "calf": 1.0,
            "ankle": 1.2,
        }
        self.measurement_names = measurement_names

    def forward(self, predictions, targets):
        loss = 0
        for i, name in enumerate(self.measurement_names):
            if name in self.weights:
                weight = self.weights.get(
                    name, 1.0
                )  # Default weight of 1.0 if not specified
                measure_loss = F.l1_loss(predictions[:, i], targets[:, i])
                loss += weight * measure_loss
        return loss / len(self.measurement_names)


class MeasurementNormalizer:
    def __init__(self, measurement_df, measurement_names):
        self.means = {}
        self.stds = {}

        # Calculate mean and std for each measurement
        for name in measurement_names:
            if name not in ["height_cm", "weight_kg"]:
                self.means[name] = measurement_df[name].mean()
                self.stds[name] = measurement_df[name].std()

    def normalize(self, measurements, measurement_names):
        normalized = []
        for i, name in enumerate(measurement_names):
            if name not in ["height_cm", "weight_kg"]:
                normalized.append(
                    (measurements[:, i] - self.means[name]) / self.stds[name]
                )
        return torch.stack(normalized, dim=1)

    def denormalize(self, normalized_measurements, measurement_names):
        denormalized = []
        for i, name in enumerate(measurement_names):
            if name not in ["height_cm", "weight_kg"]:
                denormalized.append(
                    normalized_measurements[:, i] * self.stds[name] + self.means[name]
                )
        return torch.stack(denormalized, dim=1)
