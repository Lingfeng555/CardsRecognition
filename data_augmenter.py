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
    
    # Apply all augmentations
    for h_flip in [False, True]:
        h_flipped_img = v2.RandomHorizontalFlip(p=1.0)(img) if h_flip else img
        
        for v_flip in [False, True]:
            v_flipped_img = v2.RandomVerticalFlip(p=1.0)(h_flipped_img) if v_flip else h_flipped_img
            
            for rot_transform in rotation_transforms:
                for noise in [True, False]:
                    # Create a fresh copy
                    current_img = v_flipped_img.clone()
                    
                    if noise:
                        sigma = 0.155
                        current_img = v2.GaussianNoise(sigma=sigma)(current_img)
                    
                    # Get a thread-safe counter value
                    counter_value = label_counters[text_label].increment()
                    augmentation_count += 1
                    
                    final_img = rot_transform(current_img)

                    denorm_img = final_img.clone()
                    denorm_img = torch.clamp(denorm_img, 0, 1)

                    pil_img = F.to_pil_image(denorm_img)

                    h_flip_text = "h-flipped" if h_flip else "original"
                    v_flip_text = "v-flipped" if v_flip else "original"
                    rotation = "90" if "90" in str(rot_transform) else "180" if "180" in str(rot_transform) else "270"
                    noise_text = "noisy" if noise else "clean"

                    filename = f"{counter_value:03d}_{h_flip_text}_{v_flip_text}_rot{rotation}_{noise_text}.png"
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