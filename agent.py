from typing import Literal
from utils.Loader import CardsDataset
from arquitecture.CardsClassifier import CardClassifier
import torch
import torch.nn as nn
import random
import os
import json
import time

class CardRecognizer:
    device: str
    category_classifier: CardClassifier
    suit_classifier: CardClassifier
    category_dataset: CardsDataset
    suit_dataset: CardsDataset
    
    def __init__(self, csv_file: str = "cards.csv", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        
        csv_file = csv_file
        self.device = device
        
        self.set_category_classifier(csv_file)
        self.set_suit_classifier(csv_file)

    def size (self) -> int:
        return self.category_classifier.n_parameters() + self.suit_classifier.n_parameters()

    def rebuild_classifier(self, target: Literal["suit", "category"]) -> nn.Module:
        with open(f"result/{target}_config.json", 'rb') as f:
            config = json.load(f)
            classifier = CardClassifier(image_size=torch.Size((config["image_height"], config["image_width"],)), 
                            convolution_structure=config["convolution_structure"],
                            expert_output_len=config["expert_output_len"],
                            output_len=config["output_len"],
                            pool_depth=config["pool_depth"],
                            device=self.device
                            )
        return classifier
    
    # @todo Do not let this hardcoded bitch
    def set_category_classifier(self, csv_file):
        self.category_dataset = CardsDataset(scale=0.6, split="test", csv_file=csv_file, target="category")
        self.category_classifier = self.rebuild_classifier(target="category")
        
        category_checkpoint = torch.load("result/category_classifier.pth")
        self.category_classifier.load_state_dict(category_checkpoint['model_state_dict'])
        
        pruned_expert_path = os.path.join("result/category_pruned_experts.json")
        if os.path.exists(pruned_expert_path):
            with open(pruned_expert_path, 'rb') as f:
                pruned_expert = json.load(f)
                self.category_classifier.prune_experts(list_of_experts=pruned_expert)
                
        self.category_classifier.eval()
        self.category_classifier.to(self.device)

    # @todo Do not let this hardcoded bitch
    def set_suit_classifier(self, csv_file):
        self.suit_dataset = CardsDataset(scale=0.6, split="test", csv_file=csv_file, target="suit")
        self.suit_classifier = self.rebuild_classifier(target="suit")
        
        suit_checkpoint = torch.load("result/suit_classifier.pth")
        self.suit_classifier.load_state_dict(suit_checkpoint['model_state_dict'])
        
        pruned_expert_path = os.path.join("result/suit_pruned_experts.json")
        if os.path.exists(pruned_expert_path):
            with open(pruned_expert_path, 'rb') as f:
                pruned_expert = json.load(f)
                self.suit_classifier.prune_experts(list_of_experts=pruned_expert)
                
        self.suit_classifier.eval()
        self.suit_classifier.to(self.device)
    
    def classify_card(self, image: torch.Tensor) -> tuple: 
        '''
        Classify a card image into its category and suit.
        Args:
            image (torch.Tensor): The input card image tensor. ("ace", "diamonds")
        '''
        image = image.unsqueeze(0).to(self.device)
        category = self.category_classifier(image)
        suit = self.suit_classifier(image)
        if self.device == "cuda":
            return self.category_dataset.decode_label(category.detach().cpu()), self.suit_dataset.decode_label(suit.detach().cpu())
        else:
            return self.category_dataset.decode_label(category.detach().numpy()[0]), self.suit_dataset.decode_label(suit.detach().numpy()[0])
    
if __name__ == "__main__":
    agent = CardRecognizer(csv_file="cards.csv", device="cpu")
    
    dataset = CardsDataset(scale=0.6, split="test", csv_file="cards.csv", target="labels")
    image, label = dataset.__getitem__(random.randint(0, len(dataset)))
    
    category, suit = agent.classify_card(image)
    print(f"Category: {category}, Suit: {suit}") # Category: ace, Suit: diamonds
    print(f"True Label: {dataset.decode_label(label)}")
    print(f"Agent_size: {agent.size()}")
    
    start_time = time.time()
    for i in range(len(dataset)):
        image, _ = dataset.__getitem__(i)
        agent.classify_card(image)
    elapsed_time = time.time() - start_time
    print(f"Time taken to process the dataset one by one: {elapsed_time:.2f} seconds, {elapsed_time/len(dataset):.2f} seconds per image")