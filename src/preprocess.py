import torch
import numpy as np
from PIL import Image
import random
import cv2
import os
from PIL import Image, ImageDraw
from functools import lru_cache
from torch.utils.data import Dataset
import pandas as pd
from .config import PHONE_DATASET_PATH, PREPROCESS_DIR
from tqdm import tqdm
from torchvision import transforms


# Load and preprocess data
def load_and_preprocess_data(csv_path):
    merged_measures = pd.read_csv(csv_path)
    print("Missing entries:")
    print(merged_measures.isnull().sum())

    measurement_float = [
        "ankle",
        "arm-length",
        "bicep",
        "calf",
        "chest",
        "forearm",
        "height",
        "hip",
        "leg-length",
        "shoulder-breadth",
        "shoulder-to-crotch",
        "thigh",
        "waist",
        "wrist",
        "height_cm",
        "weight_kg",
    ]

    for col in measurement_float:
        merged_measures[col] = pd.to_numeric(merged_measures[col], errors="coerce")

    print("NaN values after conversion:")
    print(merged_measures.isna().sum())

    merged_measures["gender"] = (
        merged_measures["gender"].map({"male": 0, "female": 1}).astype(int)
    )

    return merged_measures, measurement_float


def load_phone_dimensions(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(df.head())

        # Ensure all required columns are present
        required_columns = ["Brand", "Model", "Height"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in the CSV file.")

        # Combine 'Brand' and 'Model' columns
        df["Brand Model"] = df["Brand"] + " " + df["Model"]

        # Convert dimensions to float, just in case
        df["Height"] = df["Height"].astype(float)

        # If 'Width' and 'Depth' are not present, use 'Height' as a substitute
        if "Width" not in df.columns:
            df["Width"] = df["Height"]
        if "Depth" not in df.columns:
            df["Depth"] = df["Height"]

        # Create a list of tuples with the required information
        phone_dimensions = list(
            zip(df["Brand Model"], df["Height"], df["Width"], df["Depth"])
        )

        print(f"Loaded {len(phone_dimensions)} valid phone dimensions.")
        print("First few entries:")
        for dim in phone_dimensions[:5]:
            print(dim)

        return phone_dimensions

    except Exception as e:
        print(f"Error loading or processing {csv_file}: {str(e)}")
        return []


# Global variable for phone dimensions
phone_dimensions = load_phone_dimensions(PHONE_DATASET_PATH)
if not phone_dimensions:
    print("Warning: No valid phone dimensions loaded. Using default values.")
    phone_dimensions = [("Default", 150.0, 75.0, 8.0)]


class ConcatenatedImageDataset(Dataset):
    def __init__(
        self,
        front_img_dir,
        side_img_dir,
        measurement_df,
        measurement_float,
        transform=None,
        cache_size=1000,
    ):
        self.front_img_dir = front_img_dir
        self.side_img_dir = side_img_dir
        self.measurements = measurement_df
        self.measurement_float = measurement_float
        self.transform = transform
        self.cache = lru_cache(maxsize=cache_size)(self.process_image)

        # Save sample images during initialization - allows for visualization and debugging
        self.save_sample_images(num_samples=20)

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                photo_id = self.measurements.iloc[idx]["photo_id"]
                concatenated_image, phone_info = self.cache(idx, photo_id)

                if concatenated_image is None or phone_info is None:
                    idx = (idx + 1) % len(self)
                    continue

                if self.transform:
                    concatenated_image = self.transform(concatenated_image)

                targets = torch.tensor(
                    [
                        self.measurements.iloc[idx][col]
                        for col in self.measurement_float
                        if col not in ["height_cm", "weight_kg"]
                    ],
                    dtype=torch.float32,
                )
                gender = torch.tensor(
                    self.measurements.iloc[idx]["gender"], dtype=torch.float32
                )

                return concatenated_image, targets, gender, phone_info
            except Exception as e:
                idx = (idx + 1) % len(self)

        raise RuntimeError("Failed to fetch valid data after multiple attempts")

    def process_image(self, idx, photo_id):
        try:
            # 1. Load original images
            front_image = Image.open(
                os.path.join(self.front_img_dir, f"{photo_id}.png")
            ).convert("RGB")
            side_image = Image.open(
                os.path.join(self.side_img_dir, f"{photo_id}.png")
            ).convert("RGB")

            if front_image.size != side_image.size:
                return None, None

            # Get person height for scaling
            person_height_cm = self.measurements.iloc[idx]["height_cm"]

            # 2. Generate and add phone rectangles to original images
            phone_params = self.generate_phone_params(
                front_image.size[1], person_height_cm
            )
            if phone_params is None:
                return None, None

            (
                _,
                phone_length_mm,
                phone_width_mm,
                phone_depth_mm,
                front_rect,
                side_rect,
            ) = phone_params

            # Add rectangles to original images (rectangle represents the phone)
            front_image = self.add_rectangle_to_image(front_image, front_rect)
            side_image = self.add_rectangle_to_image(side_image, side_rect)

            # 3. Crop images to person bounds (including phone)
            front_bounds = self.find_person_bounds(front_image)
            side_bounds = self.find_person_bounds(side_image)
            if None in [front_bounds, side_bounds]:
                return None, None

            front_image = front_image.crop(front_bounds)
            side_image = side_image.crop(side_bounds)

            # 4. Resize to (480, 640) maintaining aspect ratio and add padding
            front_image = self.resize_with_padding(front_image, (480, 640))
            side_image = self.resize_with_padding(side_image, (480, 640))

            # 5. Create final concatenated image
            concatenated_image = Image.new("RGB", (960, 640))
            concatenated_image.paste(front_image, (0, 0))
            concatenated_image.paste(side_image, (480, 0))

            # Calculate scaled rectangle coordinates
            scaled_front_rect = self.scale_coordinates(
                front_rect, front_bounds, (480, 640)
            )
            scaled_side_rect = self.scale_coordinates(
                side_rect, side_bounds, (480, 640)
            )
            scaled_side_rect = [
                x + 480 if i % 2 == 0 else x for i, x in enumerate(scaled_side_rect)
            ]

            # Create phone info tensor
            phone_info = torch.tensor(
                [
                    phone_length_mm,
                    phone_width_mm,
                    phone_depth_mm,
                    *scaled_front_rect,
                    *scaled_side_rect,
                ],
                dtype=torch.float32,
            )

            return concatenated_image, phone_info

        except Exception as e:
            return None, None

    def generate_phone_params(self, image_height, person_height_cm):
        try:
            phone_model, phone_length_mm, phone_width_mm, phone_depth_mm = (
                random.choice(phone_dimensions)
            )

            # Convert person height to mm for consistent units
            person_height_mm = person_height_cm * 10

            # Calculate pixel-to-mm ratio based on person height
            pixels_per_mm = image_height / person_height_mm

            # Calculate phone dimensions in pixels
            phone_height_pixels = int(phone_length_mm * pixels_per_mm)
            phone_width_pixels = int(phone_width_mm * pixels_per_mm)
            phone_depth_pixels = int(phone_depth_mm * pixels_per_mm)

            # Generate positions for phone rectangles in lower half of image
            min_y = int(image_height * 0.5)  # Start from middle of image
            max_y = int(image_height * 0.8)  # Don't place too low

            # Front phone rectangle
            front_y = random.randint(min_y, max_y - phone_height_pixels)
            front_x = random.randint(0, 480 - phone_width_pixels)

            # Side phone rectangle
            side_y = random.randint(min_y, max_y - phone_height_pixels)
            side_x = random.randint(0, 480 - phone_depth_pixels)

            return (
                phone_model,
                phone_length_mm,
                phone_width_mm,
                phone_depth_mm,
                (
                    front_x,
                    front_y,
                    front_x + phone_width_pixels,
                    front_y + phone_height_pixels,
                ),
                (
                    side_x,
                    side_y,
                    side_x + phone_depth_pixels,
                    side_y + phone_height_pixels,
                ),
            )

        except Exception as e:
            return None

    def save_sample_images(self, num_samples=20):
        print(f"Saving {num_samples} sample images...")
        save_dir = PREPROCESS_DIR
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        idx = 0
        saved_count = 0
        while saved_count < num_samples and idx < len(self):
            try:
                photo_id = self.measurements.iloc[idx]["photo_id"]
                concatenated_image, phone_info = self.process_image(idx, photo_id)

                if concatenated_image is not None:
                    # Save the image
                    save_path = os.path.join(
                        save_dir, f"sample_{saved_count}_photo_{photo_id}.png"
                    )
                    concatenated_image.save(save_path)

                    # Save the phone info for verification
                    info_path = os.path.join(
                        save_dir, f"sample_{saved_count}_photo_{photo_id}_info.txt"
                    )
                    with open(info_path, "w") as f:
                        f.write(
                            f"Phone dimensions (mm): length={phone_info[0]}, width={phone_info[1]}, depth={phone_info[2]}\n"
                        )
                        f.write(f"Front rectangle: {phone_info[3:7].tolist()}\n")
                        f.write(f"Side rectangle: {phone_info[7:].tolist()}\n")

                    saved_count += 1
                    print(f"Saved sample {saved_count}/{num_samples}")

                idx += 1
            except Exception as e:
                print(f"Error saving sample {idx}: {str(e)}")
                idx += 1

        print(f"Successfully saved {saved_count} sample images to {save_dir}")

    @staticmethod
    def find_person_bounds(image):
        # Convert to numpy array for processing
        img_array = np.array(image)

        # Find non-black pixels
        non_black = np.any(img_array != 0, axis=2)
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Add padding to bounds
        padding = 10
        ymin = max(0, ymin - padding)
        ymax = min(image.size[1], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(image.size[0], xmax + padding)

        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def add_rectangle_to_image(image, rectangle_coords):
        draw = ImageDraw.Draw(image)
        draw.rectangle(rectangle_coords, outline=(0, 0, 255), width=2)
        return image

    @staticmethod
    def resize_with_padding(image, target_size):
        # Calculate aspect ratios
        target_ratio = target_size[0] / target_size[1]
        img_ratio = image.size[0] / image.size[1]

        if img_ratio > target_ratio:
            # Image is wider than target
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than target
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)

        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new black image and paste resized image
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_image.paste(resized, (paste_x, paste_y))

        return new_image

    @staticmethod
    def scale_coordinates(rect, bounds, target_size):
        # Adjust coordinates relative to crop bounds
        rel_coords = [
            rect[0] - bounds[0],
            rect[1] - bounds[1],
            rect[2] - bounds[0],
            rect[3] - bounds[1],
        ]

        # Calculate scale factors
        source_width = bounds[2] - bounds[0]
        source_height = bounds[3] - bounds[1]
        scale_x = target_size[0] / source_width
        scale_y = target_size[1] / source_height

        # Scale coordinates
        scaled_coords = [
            rel_coords[0] * scale_x,
            rel_coords[1] * scale_y,
            rel_coords[2] * scale_x,
            rel_coords[3] * scale_y,
        ]

        return scaled_coords


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Prepare dataset
def prepare_dataset(
    front_img_dir, side_img_dir, measurement_df, measurement_float, processed_dir
):
    dataset = ConcatenatedImageDataset(
        front_img_dir,
        side_img_dir,
        measurement_df,
        measurement_float,
        processed_dir=processed_dir,
    )
    print(f"Preparing dataset with {len(dataset)} images")
    for i in tqdm(range(len(dataset)), desc="Processing images"):
        _ = dataset[i]  # This will process and save all images
    print("Dataset preparation complete")
    return dataset
