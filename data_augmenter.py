import torch
from torchvision.transforms import v2
import pandas as pd
import os
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F
import skimage

from utils.Perspectiver import Perspectiver
from utils.Loader import CardsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scale_2 = 0.60
dataset_scale_050 = CardsDataset(scale=scale_2)
dataset_scale_050_test = CardsDataset(scale=scale_2, split="test")

rotation_transforms = [
    v2.RandomRotation(degrees=(90, 90)),   # Exactly 90 degrees
    v2.RandomRotation(degrees=(180, 180)), # Exactly 180 degrees  
    v2.RandomRotation(degrees=(270, 270))  # Exactly 270 degrees
]

output_base_dir = "data/augmented"
os.makedirs(output_base_dir, exist_ok=True)

label_counters = {}

for idx in range(len(dataset_scale_050_test)):
    img, label = dataset_scale_050_test.__getitem__(idx)
    
    text_label = dataset_scale_050_test.decode_label(label)
    
    label_dir = os.path.join(output_base_dir, text_label)
    os.makedirs(label_dir, exist_ok=True)
    
    if text_label not in label_counters:
        label_counters[text_label] = 0
    
    for h_flip in [False, True]:
        h_flipped_img = v2.RandomHorizontalFlip(p=1.0)(img) if h_flip else img
        
        for v_flip in [False, True]:
            v_flipped_img = v2.RandomVerticalFlip(p=1.0)(h_flipped_img) if v_flip else h_flipped_img
            
            for rot_transform in rotation_transforms:
                
                for noise in [True, False]:
                    # Create a fresh copy to prevent modification carry-over
                    current_img = v_flipped_img.clone()
                    
                    if noise:
                        sigma = 0.155
                        current_img = v2.GaussianNoise(sigma=sigma)(current_img)
                    
                    label_counters[text_label] += 1

                    final_img = rot_transform(current_img)

                    denorm_img = final_img.clone()
                    denorm_img = torch.clamp(denorm_img, 0, 1)

                    pil_img = F.to_pil_image(denorm_img)

                    h_flip_text = "h-flipped" if h_flip else "original"
                    v_flip_text = "v-flipped" if v_flip else "original"
                    rotation = "90" if "90" in str(rot_transform) else "180" if "180" in str(rot_transform) else "270"
                    noise_text = "noisy" if noise else "clean"

                    filename = f"{label_counters[text_label]:03d}_{h_flip_text}_{v_flip_text}_rot{rotation}_{noise_text}.png"

                    pil_img.save(os.path.join(label_dir, filename))

print(f"Augmentation complete. Created {sum(label_counters.values())} images across {len(label_counters)} classes.")
print(f"Images saved to {output_base_dir}")