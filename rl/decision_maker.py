import pickle
import numpy as np
from collections import defaultdict

class DecisionMaker:
    """
    A class that uses a trained Q-table to make decisions based on a given state in a Blackjack environment.
    State format: (player_total, dealer_visible_card, usable_ace)
    Action: 0 = stand, 1 = hit
    """
    def __init__(self, q_table_path="rl/q_table_final.pkl"):
        """
        Initialize the DecisionMaker with a Q-table.

        Args:
            q_table_path (str): Path to the pickled Q-table file.
        """
        try:
            with open(q_table_path, 'rb') as f:
                self.Q = defaultdict(lambda: np.zeros(2), pickle.load(f))
            print(f"Loaded Q-table from {q_table_path}")
        except FileNotFoundError:
            print(f"Q-table file {q_table_path} not found. Initializing empty Q-table.")
            self.Q = defaultdict(lambda: np.zeros(2))  # 2 actions: 0 = stand, 1 = hit

    def decide(self, state):
        """
        Make a decision based on the given state using the Q-table.

        Args:
            state (tuple): State in the format (player_total, dealer_visible_card, usable_ace).
                           - player_total: int (0-31)
                           - dealer_visible_card: int (1-10)
                           - usable_ace: int (0-2)

        Returns:
            int: Action (0 = stand, 1 = hit).
        """
        # Validate state format
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("State must be a tuple or list of length 3: (player_total, dealer_visible_card, usable_ace)")

        player_total, dealer_visible_card, usable_ace = state
        if not (0 <= player_total <= 31):
            raise ValueError("player_total must be between 0 and 31")
        if not (1 <= dealer_visible_card <= 10):
            raise ValueError("dealer_visible_card must be between 1 and 10")
        if usable_ace not in (0, 1, 2):
            raise ValueError("usable_ace must be 0, 1 or 2")

        # Convert state to tuple for Q-table lookup
        state = tuple(state)
        
        # Return the action with the highest Q-value (exploit)
        return int(np.argmax(self.Q[state]))