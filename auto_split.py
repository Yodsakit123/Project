import os
import shutil
import random

# Define paths
base_dir = "data"  # Main dataset directory
source_domain = "front_view"  # Name of source domain folder
target_domain = "side_view"   # Name of target domain folder
images_dir = "C:/Users/First/Desktop/Project/image/torch"  # Directory where uploaded images are stored
image_class = "torch"

# Define split ratio (e.g., 50% for source, 50% for target)
split_ratio = 0.5  # Adjust if needed

# Ensure reproducibility
random.seed(42)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle images randomly
random.shuffle(image_files)

# Split images into source and target
split_index = int(len(image_files) * split_ratio)
source_images = image_files[:split_index]
target_images = image_files[split_index:]

# Create domain folders
source_path = os.path.join(base_dir, source_domain, image_class)
target_path = os.path.join(base_dir, target_domain, image_class)
os.makedirs(source_path, exist_ok=True)
os.makedirs(target_path, exist_ok=True)

# Move images into respective folders
for img in source_images:
    shutil.copy(os.path.join(images_dir, img), os.path.join(source_path, img))

for img in target_images:
    shutil.copy(os.path.join(images_dir, img), os.path.join(target_path, img))

print(f"Successfully split images randomly!")
print(f"Source Domain: {source_path} ({len(source_images)} images)")
print(f"Target Domain: {target_path} ({len(target_images)} images)")
