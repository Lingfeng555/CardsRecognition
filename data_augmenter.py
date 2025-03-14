import torch
from torchvision.transforms import v2
import pandas as pd
import os
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F
import csv
from tqdm import tqdm
import concurrent.futures
import threading
from functools import partial

from utils.Perspectiver import Perspectiver
from utils.Loader import CardsDataset

def add_random_mask(img, mask_probability=0.5):
    """Add random center dot masks to the image to prevent model bias
    
    Args:
        img: Input tensor image
        mask_probability: Probability of applying a mask
    
    Returns:
        Masked tensor image
    """
    if torch.rand(1).item() > mask_probability:
        return img
    
    # Clone the image to avoid modifying the original
    masked_img = img.clone()
    
    # Get image dimensions
    _, height, width = masked_img.shape
    
    # Create a circular mask in the center
    center_h, center_w = height // 2, width // 2
    
    # Control the size of the center dot - adjust these values to change size
    # Smaller divisor = larger dot
    min_radius = min(height, width) // 6  # Minimum size
    max_radius = min(height, width) // 3  # Maximum size
    mask_radius = torch.randint(min_radius, max_radius + 1, (1,)).item()
    
    # Create the circular mask
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    dist_from_center = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
    mask = dist_from_center <= mask_radius
    
    # Apply mask by setting the region to black (0)
    for c in range(masked_img.shape[0]):
        masked_img[c][mask] = 0.0
    
    return masked_img

# Thread-safe counters
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

# Initialize main variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale_2 = 0.60

# Define CSV file path
csv_file = "data/cards_dataset.csv"
csv_exists = os.path.exists(csv_file)

# Load existing CSV to get class indices if available
class_indices = {}
csv_rows = []
if csv_exists:
    df = pd.read_csv(csv_file)
    for idx, label in zip(df['class index'], df['labels']):
        if label not in class_indices:
            class_indices[label] = idx
    max_index = max(class_indices.values()) if class_indices else -1
else:
    max_index = -1
    csv_rows.append(["class index", "filepaths", "labels", "card type", "data set"])

# Thread-safe access to class_indices and max_index
class_indices_lock = threading.Lock()
csv_rows_lock = threading.Lock()
label_counters_lock = threading.Lock()

# Create rotation transforms
rotation_transforms = [
    v2.RandomRotation(degrees=(90, 90)),
    v2.RandomRotation(degrees=(180, 180)),
    v2.RandomRotation(degrees=(270, 270))
]

# Process a single image with all augmentations
def process_image(args):
    idx, dataset, split, output_split_dir, label_counters = args
    
    img, label = dataset.__getitem__(idx)
    text_label = dataset.decode_label(label)
    
    # Get or create class index - thread safe
    with class_indices_lock:
        if text_label not in class_indices:
            global max_index
            max_index += 1
            class_indices[text_label] = max_index
        class_idx = class_indices[text_label]
    
    # Extract card type
    card_type = text_label.split(' ')[0] if ' ' in text_label else text_label
    
    # Create label directory - thread safe file operations
    label_dir = os.path.join(output_split_dir, text_label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Initialize counter for this label
    with label_counters_lock:
        if text_label not in label_counters:
            label_counters[text_label] = Counter()
    
    # Local results collection
    local_csv_rows = []
    augmentation_count = 0
    
    # Create a fresh copy of the original image
    current_img = img.clone()
    
    # 1. Apply a random rotation (just one)
    rotation_idx = torch.randint(0, len(rotation_transforms), (1,)).item()
    rot_transform = rotation_transforms[rotation_idx]
    rotation_angle = "90" if rotation_idx == 0 else "180" if rotation_idx == 1 else "270"
    
    # 2. Apply Gaussian noise with 50% probability
    apply_noise = torch.rand(1).item() < 0.5
    if apply_noise:
        sigma = 0.155
        current_img = v2.GaussianNoise(sigma=sigma)(current_img)
    
    # 3. Apply random mask (already has 50% probability internally)
    current_img = add_random_mask(current_img)
    
    # Get a thread-safe counter value
    counter_value = label_counters[text_label].increment()
    augmentation_count += 1
    
    final_img = rot_transform(current_img)
    
    denorm_img = final_img.clone()
    denorm_img = torch.clamp(denorm_img, 0, 1)
    
    pil_img = F.to_pil_image(denorm_img)
    
    # Create descriptive filename
    noise_text = "noisy" if apply_noise else "clean"
    filename = f"{counter_value:03d}_rot{rotation_angle}_{noise_text}.png"
    file_path = os.path.join(label_dir, filename)
    
    # Path for CSV - use the relative path format
    relative_path = os.path.join(f"augmented/{split}", text_label, filename)
    
    pil_img.save(file_path)
    
    # Add entry to CSV with the correct split
    local_csv_rows.append([
        class_idx,
        relative_path,
        text_label,
        card_type,
        f"augmented_{split}"  # Mark the data as augmented with the split
    ])
    
    # Append local results to global CSV rows - thread safe
    with csv_rows_lock:
        csv_rows.extend(local_csv_rows)
    
    return augmentation_count

# Process all three splits
splits = ["train", "test", "valid"]

for split in splits:
    print(f"\nProcessing {split} dataset...")
    
    # Load dataset for current split
    dataset = CardsDataset(scale=scale_2, split=split)
    
    # Create output directory for this split
    output_split_dir = os.path.join("data/augmented", split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Track labels per split
    label_counters = {}
    
    # Create argument list for all images
    args_list = [(idx, dataset, split, output_split_dir, label_counters) 
                for idx in range(len(dataset))]
    
    # Method 2: Using as_completed for better resource utilization
    max_workers = os.cpu_count() * 4  # Use more threads since this is I/O bound

    total_augmentations = 0
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_image, args) for args in args_list]
        
        # Process as they complete (better resource utilization)
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures),
                          desc=f"Augmenting {split} images"):
            aug_count = future.result()
            total_augmentations += aug_count
    
    print(f"Completed {split} augmentation. Created {total_augmentations} images across {len(label_counters)} classes.")

# Save the CSV
with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
    writer = csv.writer(f)
    for row in csv_rows:
        writer.writerow(row)

print(f"\nAugmentation complete for all splits.")
print(f"Images saved to data/augmented/[train|test|valid]")
print(f"CSV updated at {csv_file}")