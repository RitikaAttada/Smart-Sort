import os
import shutil
import random

# Define dataset paths
trashnet_path = "dataset-resized"
realwaste_path = "realwaste/realwaste-main/RealWaste"
output_dataset = "dataset"

# Define categories based on source folders
category_mapping = {
    "dry": ["cardboard", "paper", "plastic", "metal"],
    "wet": ["Food Organics", "Vegetation"],
    "other": ["glass", "trash", "Miscellaneous Trash", "Textile Trash"]
}

# Create train/test directories if not exist
for split in ["train", "test"]:
    for category in category_mapping.keys():
        os.makedirs(os.path.join(output_dataset, split, category), exist_ok=True)

# Function to move images to train/test folders
def split_and_move_images(source_path, category_map):
    for category, folders in category_map.items():
        for folder in folders:
            folder_path = os.path.join(source_path, folder)

            if not os.path.exists(folder_path):
                print(f"Skipping {folder} (not found in {source_path})")
                continue

            images = os.listdir(folder_path)
            random.shuffle(images)

            split_index = int(len(images) * 0.8)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # Move images to train/test folders
            for img in train_images:
                shutil.move(os.path.join(folder_path, img), os.path.join(output_dataset, "train", category, img))
            for img in test_images:
                shutil.move(os.path.join(folder_path, img), os.path.join(output_dataset, "test", category, img))

# Process both datasets (TrashNet and RealWaste)
split_and_move_images(trashnet_path, category_mapping)
split_and_move_images(realwaste_path, category_mapping)

print("✅ Dataset has been successfully split into train (80%) and test (20%)!")
