import gym
from gym import spaces
from collections import defaultdict
import pickle
import random
import torch
import numpy as np

class BlackjackEnv(gym.Env):
    """
    Modified Blackjack environment with a realistic poker deck, usable Ace logic, and suit-based rewards.
    """
    def __init__(self, csv_path = "cards.csv", q_table_path="q_table_dealer.pkl"):
        super(BlackjackEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0 = Stand, 1 = Hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's hand total (0–31)
            spaces.Discrete(11),  # Dealer's visible card (1–10)
            spaces.Discrete(2)    # Usable Ace: 1 = yes, 0 = no
        ))

        try:
            with open(q_table_path, 'rb') as f:
                self.Q = defaultdict(lambda: np.zeros(self.action_space.n), pickle.load(f))
            print(f"Loaded Q-table from {q_table_path}")
        except FileNotFoundError:
            print(f"Q-table file {q_table_path} not found. Initializing empty Q-table.")
            self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        
        self.deck = self._init_deck()
        self.reset()

    def _init_deck(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

        values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        deck = [{'value': v, 'suit': s} for s in suits for v in values]

        random.shuffle(deck)

        return deck

    def draw_card(self):
        if len(self.deck) == 0:
            self.deck = self._init_deck()
        return self.deck.pop()

    def card_value(self, card, ace_as_one=True):
        """
        Calculate the value of a card. For aces, ace_as_one determines if the ace is counted as 1 or 11
        before adding the suit adjustment.
        """
        if card['suit'] == 'Hearts':
            added_val = 2
        elif card['suit'] == 'Diamonds':
            added_val = 1
        elif card['suit'] == 'Clubs':
            added_val = 0
        else:
            added_val = -2

        if card['value'] == 'A':
            base_value = 1 if ace_as_one else 11
            return base_value + added_val
        elif card['value'] in ['J', 'Q', 'K']:
            return 10 + added_val
        return int(card['value']) + added_val

    def card_suit(self, card):
        return card['suit']

    def hand_value(self, hand):
        """
        Calculate the total value of the hand, considering multiple aces with suit adjustments.
        Returns the best total (≤ 21 if possible), the number of usable aces, and the suits.
        """
        # Calculate base total from non-ace cards
        base_total = 0
        aces = []
        for card in hand:
            if card['value'] == 'A':
                aces.append(card)
            else:
                base_total += self.card_value(card)

        suits = [self.card_suit(card) for card in hand]
        if not aces:
            return base_total, 0, suits

        # For each ace, compute its possible values (1 or 11, adjusted by suit)
        ace_values = []
        for ace in aces:
            value_as_one = self.card_value(ace, ace_as_one=True)
            value_as_eleven = self.card_value(ace, ace_as_one=False)
            ace_values.append((value_as_one, value_as_eleven))

        # Try all combinations of ace values
        best_total = base_total
        best_usable_aces = 0
        best_bust_total = float('inf')

        def evaluate_combinations(index, current_total, usable_aces):
            nonlocal best_total, best_usable_aces, best_bust_total
            if index == len(ace_values):
                if current_total <= 21:
                    if current_total > best_total:
                        best_total = current_total
                        best_usable_aces = usable_aces
                else:
                    if current_total < best_bust_total:
                        best_bust_total = current_total
                return

            # Ace as 1 (or adjusted value)
            evaluate_combinations(index + 1, current_total + ace_values[index][0], usable_aces)

            # Ace as 11 (or adjusted value)
            evaluate_combinations(index + 1, current_total + ace_values[index][1], usable_aces + 1)

        evaluate_combinations(0, base_total, 0)

        # If all combinations bust, use the smallest total
        if best_total == base_total and best_bust_total != float('inf'):
            best_total = best_bust_total
            best_usable_aces = 0  # Reset usable aces if we bust

        return best_total, best_usable_aces, suits

    def _get_obs(self):
        total, usable_ace, _ = self.hand_value(self.player)
        dealer_card_value = self.card_value(self.dealer[0])
        return (total, dealer_card_value, usable_ace)
    
    def _get_dealer_state(self):
        dealer_total, dealer_usable_ace, _ = self.hand_value(self.dealer)
        player_first_card_value = self.card_value(self.player[0])
        return (dealer_total, player_first_card_value, dealer_usable_ace)

    def reset(self):
        self.deck = self._init_deck()
        self.player = [self.draw_card()]
        self.dealer = [self.draw_card()]
        self.done = False
        self.player_stick = False
        self.dealer_stick = False
        return self._get_obs()

    def dealer_policy(self, dealer_hand):
        state = self._get_dealer_state()
        state = tuple(state)  # Q-table keys are tuples
        return np.argmax(self.Q[state])  # Choose the action with the highest Q-value

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