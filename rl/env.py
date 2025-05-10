import numpy as np
import gym
from gym import spaces
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import CardRecognizer
from utils.Loader import CardsDataset
import pandas as pd
import torch
from torchvision import transforms

class BlackjackEnv(gym.Env):
    """
    Modified Blackjack environment with a realistic poker deck, usable Ace logic, and suit-based rewards.
    """
    def __init__(self, csv_path = "cards.csv"):
        super(BlackjackEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0 = Stand, 1 = Hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's hand total (0–31)
            spaces.Discrete(11),  # Dealer's visible card (1–10)
            spaces.Discrete(2)    # Usable Ace: 1 = yes, 0 = no
        ))

        self.csv_path = csv_path
        self.agent = CardRecognizer(csv_file=csv_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.deck = self._init_deck()
        self.reset()

    def _init_deck(self):
        """
        Initialize the deck by sampling one image per unique card from the dataset
        and using CardRecognizer to identify its category and suit.
        """
        # Load dataset
        card_data = pd.read_csv(self.csv_path)
        card_data = card_data[card_data['data set'] == 'test']
        card_data = card_data[card_data['labels'].str.lower() != 'joker']
        card_data = card_data[['filepaths', 'labels', 'card type']].reset_index(drop=True)

        dataset = CardsDataset(
            path='./data/',
            csv_file='cards.csv',
            split='test',
            target='labels',
            scale=0.6,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        unique_cards = card_data['labels'].unique()

        deck = []
        for card_label in unique_cards:
            card_of_type = card_data[card_data['labels'] == card_label]
            select_card = card_of_type.sample(1).iloc[0]
            card_path = select_card['filepaths']
            
            try:
                img_idx = card_data[card_data['filepaths'] == card_path].index[0]
                image, _ = dataset.__getitem__(img_idx)

                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)

                # Classify the card
                category, suit = self.agent.classify_card(image)

                category_map = {
                    'ace': 'A',
                    'two': '2',
                    'three': '3',
                    'four': '4',
                    'five': '5',
                    'six': '6',
                    'seven': '7',
                    'eight': '8',
                    'nine': '9',
                    'ten': '10',
                    'jack': 'J',
                    'queen': 'Q',
                    'king': 'K'
                }
                value = category_map.get(category.lower(), 'A')  # Default to A if unrecognized
                suit = suit.capitalize()

                deck.append({'value': value, 'suit': suit})
            except Exception as e:
                print(f"Error processing card {card_label} at {card_path}: {e}")
                continue

        random.shuffle(deck)
        return deck

    def draw_card(self):
        if len(self.deck) == 0:
            self.deck = self._init_deck()
        return self.deck.pop()

    def card_value(self, card):
        if card['suit'] == 'Hearts':
            added_val = 2
        elif card['suit'] == 'Diamonds':
            added_val = 1
        elif card['suit'] == 'Clubs':
            added_val = 0
        else:
            added_val = -2

        if card['value'] == 'A':
            return 1 + added_val
        elif card['value'] in ['J', 'Q', 'K']:
            return 10 + added_val
        
        return int(card['value']) + added_val

    def card_suit(self, card):
        return card['suit']

    def hand_value(self, hand):
        total = sum(self.card_value(card) for card in hand)
        suits = [self.card_suit(card) for card in hand]
        ace_count = sum(1 for card in hand if card['value'] == 'A')
        usable_ace = 0

        while ace_count > 0:
            if total + 10 <= 21:
                total += 10
                usable_ace = 1
                break
            ace_count -= 1

        return total, usable_ace, suits

    def _get_obs(self):
        total, usable_ace, _ = self.hand_value(self.player)
        dealer_card_value = self.card_value(self.dealer[0])
        return (total, dealer_card_value, usable_ace)

    def reset(self):
        self.deck = self._init_deck()
        self.player = [self.draw_card()]
        self.dealer = [self.draw_card()]
        self.done = False
        self.player_stick = False
        self.dealer_stick = False
        return self._get_obs()

    def dealer_policy(self, dealer_hand):
        total, _, _ = self.hand_value(dealer_hand)
        return 0 if total >= 17 else 1

    def step(self, player_action):
        assert self.action_space.contains(player_action), "Invalid action (0 = stick, 1 = hit)"

        dealer_action = self.dealer_policy(self.dealer)

        if not self.player_stick and player_action == 1:
            self.player.append(self.draw_card())
        else:
            self.player_stick = True

        if not self.dealer_stick and dealer_action == 1:
            self.dealer.append(self.draw_card())
        else:
            self.dealer_stick = True

        player_total, _, player_suits = self.hand_value(self.player)
        dealer_total, _, _ = self.hand_value(self.dealer)

        if player_total > 21 or dealer_total > 21 or (self.player_stick and self.dealer_stick):
            self.done = True
            reward = self._get_reward(player_total, dealer_total, player_suits)
        else:
            reward = 0

        return self._get_obs(), reward, self.done, {}

    def _get_reward(self, player_total, dealer_total, player_suits):
        if player_total > 21:
            return -1
        elif dealer_total > 21:
            base_reward = 1
        elif player_total > dealer_total:
            base_reward = 1.5
        elif player_total < dealer_total:
            return -1
        elif player_total == dealer_total and player_total <= 21:
            base_reward = 0.3
        else:
            return 0

        return base_reward