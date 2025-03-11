import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CardsDataset(Dataset):

    def __init__(self, path: str = "data/", transform = None, seed: int = 55, scale: float = 1, split: str = "train", convert: str = "L"):
        self.data = pd.read_csv(os.path.join(path, "cards.csv"))
        self.data = self.data[self.data["data set"]==split].drop(columns=["data set", "card type"])
        
        if transform is not None: self.transform = transform
        else: self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.scale = scale
        self.path = path

        if seed != None :
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        self.data["filepaths_full"] = self.data["filepaths"].apply(lambda x: os.path.join(self.path, x))
        self.labels = pd.get_dummies(self.data, columns=['labels'], drop_first=True, dtype=int).drop(columns=["class index", "filepaths", "filepaths_full"])
        self.convert = convert

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index]["filepaths_full"]
        
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")

        image = Image.open(img_path).convert(self.convert)

        original_width, original_height = image.size
        image = image.resize((max(1, int(original_width * self.scale)), max(1, int(original_height * self.scale))))

        image = self.transform(image)
        
        label = torch.tensor(self.labels.iloc[index].values.astype(int), dtype=torch.int8)
        return image, label
    
    def decode_label(self, encoed_label):
        return np.array(self.labels.columns.to_list())[np.argmax(encoed_label)].removeprefix("labels_")
