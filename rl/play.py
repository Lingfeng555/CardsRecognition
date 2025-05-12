from env import BlackjackEnv
from utils.logger import save_game_to_parquet
import numpy as np
import pickle
from datetime import datetime

env = BlackjackEnv()
n_episodes = 1000

# Load Q-table
with open("q_table_final.pkl", "rb") as f:
    Q = pickle.load(f)

def choose_action(state):
    state = tuple(state)
    return np.argmax(Q.get(state, np.zeros(env.action_space.n)))

games = []

for episode in range(n_episodes):
    state = env.reset()
    state = tuple(state)
    done = False
    history = []

    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        # Store hand as (player_total, dealer_total)
        player_total, _, _ = env.hand_value(env.player)
        dealer_total, _, _ = env.hand_value(env.dealer)
        history.append((player_total, dealer_total))

        state = next_state

    games.append({
        "shown_cards": (env.card_value(env.player[0]), env.card_value(env.dealer[0])),
        "hands": history,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Save games
save_game_to_parquet(games, path_prefix="data/played_blackjack")
print("Games saved to data/played_blackjack.parquet")