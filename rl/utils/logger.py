import pandas as pd
from datetime import datetime

def save_game_to_parquet(games, path_prefix="blackjack_session"):
    """
    Guarda una lista de partidas como un archivo .parquet con el formato especificado.
    
    games: lista de diccionarios:
        {
            "shown_cards": (player_card_value, dealer_card_value),
            "hands": [ (player_total, dealer_total), ... ],
            "timestamp": str (e.g., "2025-05-07 18:46:00")
        }
    """
    records = []

    for game in games:
        row = {
            "Timestamp": game["timestamp"],
            "Shown_cards": game["shown_cards"]
        }
        for i, hand in enumerate(game["hands"]):
            row[f"Hand {i}"] = hand
        records.append(row)
    
    df = pd.DataFrame(records)
    filename = f"{path_prefix}.parquet"
    df.to_parquet(filename, engine="pyarrow", index=False)
    print(f"Saved to {filename}")