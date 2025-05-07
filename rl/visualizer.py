import pandas as pd
import os

def list_parquet_files(directory="data"):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".parquet")]

def load_full_hands_table(file_path):
    print(f"Attempting to load file: {file_path}")
    try:
        df = pd.read_parquet(file_path)

        # Select only desired columns
        hand_cols = [col for col in df.columns if col.startswith("Hand")]
        hand_cols.sort(key=lambda x: int(x.split(" ")[1]))
        display_cols = ["Timestamp", "Shown_cards"] + hand_cols
        
        return df[display_cols]
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return None

if __name__ == "__main__":
    parquet_files = list_parquet_files()

    if not parquet_files:
        print("No .parquet files found in the 'data' directory.")
    else:
        print("Found .parquet files:")
        for idx, f in enumerate(parquet_files):
            print(f"{idx}: {os.path.basename(f)}")

        choice = input("Select file number: ")
        try:
            idx = int(choice)
            if 0 <= idx < len(parquet_files):
                df = load_full_hands_table(parquet_files[idx])
                if df is not None:
                    print("\nHands played per game:\n")
                    print(df.fillna("").to_string(index=False))
            else:
                print("Invalid file number.")
        except ValueError:
            print("Please enter a valid number.")