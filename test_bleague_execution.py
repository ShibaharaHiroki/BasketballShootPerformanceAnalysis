
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from core.bleague_data_loader import load_bleague_data, get_bleague_players, make_bleague_tensor

def test_loading():
    print("Testing B.League Data Loading...")
    try:
        df = load_bleague_data()
        print(f"Successfully loaded data. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Sample data:\n", df.head())
        
        print("\nTesting Player List...")
        players = get_bleague_players()
        print(f"Found {len(players)} players.")
        if players:
            print("Top player:", players[0])
            
        print("\nTest passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
