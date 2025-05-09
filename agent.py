from utils.Loader import CardsDataset
from arquitecture.CardsClassifier import CardClassifier
import torch
import random
class Agent:
    device: str
    category_classifier: CardClassifier
    suit_classifier: CardClassifier
    category_dataset: CardsDataset
    suit_dataset: CardsDataset
    
    def __init__(self, csv_file: str = "cards.csv",):
        
        
        csv_file = csv_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.set_category_classifier(csv_file)
        self.set_suit_classifier(csv_file)

    # @todo Do not let this hardcoded bitch
    def set_category_classifier(self, csv_file):
        self.category_dataset = CardsDataset(scale=0.6, split="test", csv_file=csv_file, target="category")
        _, label = self.category_dataset.__getitem__(1)
        self.category_classifier = CardClassifier(image_size=torch.Size((134, 134)), 
                            convolution_structure=[1,8,8,16,16,24,24,32,32],
                            expert_output_len=4,
                            expert_depth=4,
                            output_len=len(label),
                            pool_depth=2
                            )
        category_checkpoint = torch.load("result/category_classifier.pth")
        self.category_classifier.load_state_dict(category_checkpoint['model_state_dict'])
        self.category_classifier.eval()
        self.category_classifier.to(self.device)

    # @todo Do not let this hardcoded bitch
    def set_suit_classifier(self, csv_file):
        self.suit_dataset = CardsDataset(scale=0.6, split="test", csv_file=csv_file, target="suit")
        _, label = self.suit_dataset.__getitem__(1)
        self.suit_classifier = CardClassifier(image_size=torch.Size((134, 134)), 
                            convolution_structure=[1,8,8,16,16,24,24,32,32],
                            expert_output_len=2,
                            expert_depth=4,
                            output_len=len(label),
                            pool_depth=2
                            ).to(self.device)
        suit_checkpoint = torch.load("result/suit_classifier.pth")
        self.suit_classifier.load_state_dict(suit_checkpoint['model_state_dict'])
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
        return self.category_dataset.decode_label(category.detach().cpu()), self.suit_dataset.decode_label(suit.detach().cpu())
    
if __name__ == "__main__":
    agent = Agent(csv_file="cards.csv")
    
    suit_dataset = CardsDataset(scale=0.6, split="test", csv_file="cards.csv", target="labels")
    image, label = suit_dataset.__getitem__(random.randint(0, len(suit_dataset)))
    
    category, suit = agent.classify_card(image)
    print(f"Category: {category}, Suit: {suit}") # Category: ace, Suit: diamonds
    print(f"True Label: {suit_dataset.decode_label(label)}")



