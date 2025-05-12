from env import BlackjackEnv
from utils.logger import save_game_to_parquet
import numpy as np
import random
from collections import defaultdict
import pickle
from datetime import datetime

env = BlackjackEnv()
n_episodes = 100_000
alpha = 0.1           # learning rate
gamma = 1.0           # discount factor
epsilon = 1.0         # exploration rate
epsilon_decay = 0.99995
min_epsilon = 0.05

Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Q-table

def choose_action(state):
    state = tuple(state) if isinstance(state, (list, tuple)) else state
    if random.random() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(Q[state])        # exploit

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

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta

        state = next_state

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if episode % 1000 == 0:
        print(f"Episode {episode} - Epsilon: {epsilon:.4f}")
    
    if episode % 10000 == 0:
        with open(f"q_table_{episode}.pkl", "wb") as f:
            pickle.dump(dict(Q), f)
        print("Q-table saved to q_table.pkl")
    
    if episode == 99999:
        with open("q_table_final.pkl", "wb") as f:
            pickle.dump(dict(Q), f)
        print("Final Q-table saved to q_table_final.pkl")

    if episode % 50 == 0:
        games.append({
            "shown_cards": (env.card_value(env.player[0]), env.card_value(env.dealer[0])),
            "hands": history,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Save games
save_game_to_parquet(games, path_prefix="data/trained_blackjack")