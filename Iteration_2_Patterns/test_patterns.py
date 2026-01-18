"""
Test script to verify pattern detection is working correctly
"""
import numpy as np
from connect4_adp import Connect4Game, ADPAgent

def test_pattern_detection():
    """Test that pattern counting works correctly"""
    
    # Create a test board
    board = np.zeros((6, 7), dtype=int)
    agent = ADPAgent(player_id=1)
    
    print("Testing Pattern Detection")
    print("=" * 60)
    
    # Test 1: Horizontal 2-in-a-row (extendable)
    board[:] = 0
    board[5, 0] = 1  # Player 1
    board[5, 1] = 1  # Player 1
    # Pattern: [1, 1, 0, 0] - should count as 1 extendable 2-in-a-row
    
    features = agent.get_features(board)
    print("\nTest 1: Horizontal 2-in-a-row")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [>=1, 0, 0, 0] - at least one 2-in-a-row")
    
    # Test 2: Horizontal 3-in-a-row (extendable)
    board[:] = 0
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1
    # Pattern: [1, 1, 1, 0] - should count as 1 extendable 3-in-a-row
    
    features = agent.get_features(board)
    print("\nTest 2: Horizontal 3-in-a-row")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [0, >=1, 0, 0] - at least one 3-in-a-row")
    
    # Test 3: Blocked pattern (not extendable)
    board[:] = 0
    board[5, 0] = 2  # Opponent blocks one end
    board[5, 1] = 1
    board[5, 2] = 1
    board[5, 3] = 2  # Opponent blocks other end
    # Pattern: [2, 1, 1, 2] - blocked, should NOT count
    
    features = agent.get_features(board)
    print("\nTest 3: Blocked pattern")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [0, 0, 0, 0] - blocked patterns don't count")
    
    # Test 4: Opponent patterns
    board[:] = 0
    board[5, 0] = 2
    board[5, 1] = 2
    board[5, 2] = 2
    # Pattern: [2, 2, 2, 0] - opponent's 3-in-a-row
    
    features = agent.get_features(board)
    print("\nTest 4: Opponent's 3-in-a-row")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [0, 0, 0, >=1] - opponent has 3-in-a-row")
    
    # Test 5: Vertical pattern
    board[:] = 0
    board[5, 0] = 1
    board[4, 0] = 1
    board[3, 0] = 1
    # Vertical: [1, 1, 1, 0] - should count
    
    features = agent.get_features(board)
    print("\nTest 5: Vertical 3-in-a-row")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [0, >=1, 0, 0] - vertical 3-in-a-row")
    
    # Test 6: Diagonal pattern
    board[:] = 0
    board[5, 0] = 1
    board[4, 1] = 1
    board[3, 2] = 1
    # Diagonal: [1, 1, 1, 0] - should count
    
    features = agent.get_features(board)
    print("\nTest 6: Diagonal 3-in-a-row")
    print(board)
    print(f"Features: {features}")
    print(f"Expected: [0, >=1, 0, 0] - diagonal 3-in-a-row")
    
    print("\n" + "=" * 60)
    print("Pattern detection tests complete!")

if __name__ == "__main__":
    test_pattern_detection()
