"""
Connect-4 Approximate Dynamic Programming Implementation
This implements TD learning with post-decision state value functions

ITERATION 2: Pattern-based features
- 4 features: my 2s, my 3s, opponent's 2s, opponent's 3s
- All patterns must be extendable (not blocked)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for Connect-4 game"""
    rows: int = 6
    cols: int = 7
    connect: int = 4
    

class Connect4Game:
    """
    Connect-4 game environment
    Player 1 uses value 1, Player 2 uses value 2
    Empty cells are 0
    """
    
    def __init__(self, config: GameConfig = GameConfig()):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.config.rows, self.config.cols), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def get_valid_actions(self) -> List[int]:
        """Return list of valid column indices (0-6)"""
        return [col for col in range(self.config.cols) if self.board[0, col] == 0]
    
    def make_move(self, column: int) -> Tuple[np.ndarray, float, bool]:
        """
        Make a move in the specified column
        Returns: (new_board, reward, done)
        """
        if column not in self.get_valid_actions():
            raise ValueError(f"Invalid move: column {column} is full")
        
        # Find the lowest empty row in this column
        for row in range(self.config.rows - 1, -1, -1):
            if self.board[row, column] == 0:
                self.board[row, column] = self.current_player
                break
        
        # Check for win
        reward = 0
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0  # Winner gets +1
        elif len(self.get_valid_actions()) == 0:
            # Draw
            self.done = True
            self.winner = 0
            reward = 0.0
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2
        
        return self.board.copy(), reward, self.done
    
    def _check_win(self, player: int) -> bool:
        """Check if the specified player has won"""
        rows, cols = self.config.rows, self.config.cols
        connect = self.config.connect
        
        # Check horizontal
        for r in range(rows):
            for c in range(cols - connect + 1):
                if all(self.board[r, c + i] == player for i in range(connect)):
                    return True
        
        # Check vertical
        for r in range(rows - connect + 1):
            for c in range(cols):
                if all(self.board[r + i, c] == player for i in range(connect)):
                    return True
        
        # Check diagonal (down-right)
        for r in range(rows - connect + 1):
            for c in range(cols - connect + 1):
                if all(self.board[r + i, c + i] == player for i in range(connect)):
                    return True
        
        # Check diagonal (down-left)
        for r in range(rows - connect + 1):
            for c in range(connect - 1, cols):
                if all(self.board[r + i, c - i] == player for i in range(connect)):
                    return True
        
        return False
    
    def render(self):
        """Print the board"""
        print("\n" + "=" * (self.config.cols * 4 + 1))
        for row in self.board:
            print("|", end="")
            for cell in row:
                symbol = " . " if cell == 0 else (" X " if cell == 1 else " O ")
                print(symbol + "|", end="")
            print()
        print("=" * (self.config.cols * 4 + 1))
        print("  " + "   ".join(str(i) for i in range(self.config.cols)))
        print()


class ADPAgent:
    """
    Agent that learns using Approximate Dynamic Programming
    Uses linear value function approximation with TD learning
    """
    
    def __init__(self, player_id: int, n_features: int = 4, 
                 alpha: float = 0.1, initial_epsilon: float = 1.0):
        """
        Args:
            player_id: 1 or 2
            n_features: number of features (4 for pattern features)
            alpha: learning rate
            initial_epsilon: initial exploration rate
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.n_features = n_features
        self.alpha = alpha
        self.initial_epsilon = initial_epsilon
        
        # Initialize weights randomly (small values)
        self.theta = np.random.randn(n_features) * 0.01
        
    def count_patterns(self, board: np.ndarray, player: int, length: int) -> int:
        """
        Count extendable patterns of given length for a player
        
        Args:
            board: game board
            player: player id (1 or 2)
            length: pattern length (2 or 3)
        
        Returns:
            count of extendable patterns
        """
        rows, cols = board.shape
        count = 0
        connect = 4  # Need to reach 4 to win
        
        # Check all possible windows of size 4 (the winning length)
        # A pattern is extendable if it has 'length' pieces and can reach 4
        
        # Horizontal
        for r in range(rows):
            for c in range(cols - connect + 1):
                window = board[r, c:c + connect]
                if self._is_extendable_pattern(window, player, length):
                    count += 1
        
        # Vertical
        for r in range(rows - connect + 1):
            for c in range(cols):
                window = board[r:r + connect, c]
                if self._is_extendable_pattern(window, player, length):
                    count += 1
        
        # Diagonal (down-right)
        for r in range(rows - connect + 1):
            for c in range(cols - connect + 1):
                window = np.array([board[r + i, c + i] for i in range(connect)])
                if self._is_extendable_pattern(window, player, length):
                    count += 1
        
        # Diagonal (down-left)
        for r in range(rows - connect + 1):
            for c in range(connect - 1, cols):
                window = np.array([board[r + i, c - i] for i in range(connect)])
                if self._is_extendable_pattern(window, player, length):
                    count += 1
        
        return count
    
    def _is_extendable_pattern(self, window: np.ndarray, player: int, length: int) -> bool:
        """
        Check if a window of 4 cells contains an extendable pattern
        
        Args:
            window: array of 4 cells
            player: player id
            length: target pattern length (2 or 3)
        
        Returns:
            True if window has exactly 'length' player pieces, rest empty (no opponent)
        """
        player_count = np.sum(window == player)
        opponent_count = np.sum(window == (3 - player))
        empty_count = np.sum(window == 0)
        
        # Pattern is extendable if:
        # - Has exactly 'length' of our pieces
        # - Has no opponent pieces (not blocked)
        # - Has empty spaces to extend
        return (player_count == length and 
                opponent_count == 0 and 
                empty_count == (4 - length))
        
    def get_features(self, board: np.ndarray) -> np.ndarray:
        """
        Extract pattern features from the board state
        
        Features (4 total):
        0: My extendable 2-in-a-rows
        1: My extendable 3-in-a-rows
        2: Opponent's extendable 2-in-a-rows
        3: Opponent's extendable 3-in-a-rows
        """
        features = np.zeros(self.n_features)
        
        features[0] = self.count_patterns(board, self.player_id, length=2)
        features[1] = self.count_patterns(board, self.player_id, length=3)
        features[2] = self.count_patterns(board, self.opponent_id, length=2)
        features[3] = self.count_patterns(board, self.opponent_id, length=3)
        
        return features
    
    def get_value(self, board: np.ndarray) -> float:
        """Compute V(S) = theta^T * phi(S)"""
        features = self.get_features(board)
        return np.dot(self.theta, features)
    
    def select_action(self, game: Connect4Game, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy
        """
        valid_actions = game.get_valid_actions()
        
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose action with highest post-decision value
            best_action = None
            best_value = -np.inf
            
            for action in valid_actions:
                # Simulate the action to get post-decision state
                temp_game = Connect4Game(game.config)
                temp_game.board = game.board.copy()
                temp_game.current_player = game.current_player
                temp_game.done = game.done
                
                temp_game.make_move(action)
                post_decision_value = self.get_value(temp_game.board)
                
                if post_decision_value > best_value:
                    best_value = post_decision_value
                    best_action = action
            
            return best_action
    
    def update(self, post_decision_board: np.ndarray, target: float):
        """
        Update weights using TD learning
        theta <- theta + alpha * delta * phi(S^a)
        where delta = target - V(S^a)
        """
        features = self.get_features(post_decision_board)
        current_value = np.dot(self.theta, features)
        
        # TD error
        delta = target - current_value
        
        # Clip delta to prevent overflow (conservative update)
        delta = np.clip(delta, -10.0, 10.0)
        
        # Update weights
        gradient = delta * features
        self.theta += self.alpha * gradient
        
        # Clip weights to prevent explosion
        self.theta = np.clip(self.theta, -10.0, 10.0)


class Connect4Trainer:
    """Trainer for Connect-4 ADP agents"""
    
    def __init__(self, config: GameConfig = GameConfig()):
        self.config = config
        # Note: Pattern features need smaller alpha than cell features
        # Pattern counts are larger numbers (0-20) vs binary (0-1)
        self.agent1 = ADPAgent(player_id=1, n_features=4, alpha=0.005, initial_epsilon=1.0)
        self.agent2 = ADPAgent(player_id=2, n_features=4, alpha=0.005, initial_epsilon=1.0)
        
        # Training statistics
        self.training_stats = {
            'player1_wins': [],
            'player2_wins': [],
            'draws': [],
            'epsilon_values': []
        }
    
    def get_epsilon(self, episode: int, total_episodes: int, 
                   epsilon_start: float = 1.0, epsilon_end: float = 0.1) -> float:
        """Linear epsilon decay"""
        return epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / total_episodes)
    
    def train_episode(self, epsilon: float) -> int:
        """
        Play one training game
        Returns: winner (1, 2, or 0 for draw)
        """
        game = Connect4Game(self.config)
        game.reset()
        
        # Separate histories for each player (only their own post-decision states)
        history_p1 = []  # List of post-decision boards for player 1
        history_p2 = []  # List of post-decision boards for player 2
        
        while not game.done:
            current_agent = self.agent1 if game.current_player == 1 else self.agent2
            
            # Select action
            action = current_agent.select_action(game, epsilon)
            
            # Create post-decision state (before opponent responds)
            # Find where the piece will land
            row = None
            for r in range(self.config.rows - 1, -1, -1):
                if game.board[r, action] == 0:
                    row = r
                    break
            
            # Create post-decision board
            post_decision_board = game.board.copy()
            post_decision_board[row, action] = current_agent.player_id
            
            # Store in appropriate history
            if current_agent.player_id == 1:
                history_p1.append(post_decision_board.copy())
            else:
                history_p2.append(post_decision_board.copy())
            
            # Execute move in game
            board, reward, done = game.make_move(action)
        
        # Game ended - update both agents
        # Determine terminal rewards from each player's perspective
        if game.winner == 0:
            reward_p1 = 0.0
            reward_p2 = 0.0
        elif game.winner == 1:
            reward_p1 = 1.0
            reward_p2 = -1.0
        else:  # winner == 2
            reward_p1 = -1.0
            reward_p2 = 1.0
        
        # Update Player 1's value function
        for i, post_dec_board in enumerate(history_p1):
            if i < len(history_p1) - 1:
                # Bootstrap from next post-decision state
                target = self.agent1.get_value(history_p1[i + 1])
            else:
                # Terminal state - use actual reward
                target = reward_p1
            
            self.agent1.update(post_dec_board, target)
        
        # Update Player 2's value function
        for i, post_dec_board in enumerate(history_p2):
            if i < len(history_p2) - 1:
                # Bootstrap from next post-decision state
                target = self.agent2.get_value(history_p2[i + 1])
            else:
                # Terminal state - use actual reward
                target = reward_p2
            
            self.agent2.update(post_dec_board, target)
        
        return game.winner
    
    def train(self, n_episodes: int = 10000, eval_interval: int = 1000):
        """
        Train both agents for n_episodes
        """
        print(f"Training for {n_episodes} episodes...")
        
        wins = {1: 0, 2: 0, 0: 0}  # Track wins in current window
        
        for episode in range(n_episodes):
            epsilon = self.get_epsilon(episode, n_episodes)
            winner = self.train_episode(epsilon)
            wins[winner] += 1
            
            # Evaluate and print progress
            if (episode + 1) % eval_interval == 0:
                total = eval_interval
                p1_pct = wins[1] / total * 100
                p2_pct = wins[2] / total * 100
                draw_pct = wins[0] / total * 100
                
                print(f"Episode {episode + 1}/{n_episodes}, ε={epsilon:.3f}")
                print(f"  Last {eval_interval} games: P1: {p1_pct:.1f}%, P2: {p2_pct:.1f}%, Draws: {draw_pct:.1f}%")
                
                self.training_stats['player1_wins'].append(p1_pct)
                self.training_stats['player2_wins'].append(p2_pct)
                self.training_stats['draws'].append(draw_pct)
                self.training_stats['epsilon_values'].append(epsilon)
                
                # Reset counters
                wins = {1: 0, 2: 0, 0: 0}
        
        print("\nTraining complete!")
    
    def evaluate(self, n_games: int = 1000):
        """
        Evaluate trained agents with greedy policy (no exploration)
        """
        print(f"\nEvaluating over {n_games} games with greedy policy...")
        wins = {1: 0, 2: 0, 0: 0}
        
        for _ in range(n_games):
            winner = self.train_episode(epsilon=0.0)  # Greedy
            wins[winner] += 1
        
        total = n_games
        print(f"\nEvaluation Results:")
        print(f"  Player 1 wins: {wins[1]} ({wins[1]/total*100:.2f}%)")
        print(f"  Player 2 wins: {wins[2]} ({wins[2]/total*100:.2f}%)")
        print(f"  Draws: {wins[0]} ({wins[0]/total*100:.2f}%)")
        
        # Print learned weights for analysis
        print(f"\nLearned Weights (what the agents value):")
        print(f"Player 1 weights: {self.agent1.theta}")
        print(f"  [0] My 2-in-a-rows: {self.agent1.theta[0]:.4f}")
        print(f"  [1] My 3-in-a-rows: {self.agent1.theta[1]:.4f}")
        print(f"  [2] Opp 2-in-a-rows: {self.agent1.theta[2]:.4f}")
        print(f"  [3] Opp 3-in-a-rows: {self.agent1.theta[3]:.4f}")
        print(f"\nPlayer 2 weights: {self.agent2.theta}")
        print(f"  [0] My 2-in-a-rows: {self.agent2.theta[0]:.4f}")
        print(f"  [1] My 3-in-a-rows: {self.agent2.theta[1]:.4f}")
        print(f"  [2] Opp 2-in-a-rows: {self.agent2.theta[2]:.4f}")
        print(f"  [3] Opp 3-in-a-rows: {self.agent2.theta[3]:.4f}")
        
        return wins
    
    def plot_training_progress(self):
        """Plot training statistics"""
        if not self.training_stats['player1_wins']:
            print("No training statistics to plot")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot win rates
        plt.subplot(1, 2, 1)
        x = range(len(self.training_stats['player1_wins']))
        plt.plot(x, self.training_stats['player1_wins'], label='Player 1', marker='o')
        plt.plot(x, self.training_stats['player2_wins'], label='Player 2', marker='s')
        plt.plot(x, self.training_stats['draws'], label='Draws', marker='^')
        plt.xlabel('Evaluation Interval')
        plt.ylabel('Win Rate (%)')
        plt.title('Training Progress: Win Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(x, self.training_stats['epsilon_values'], marker='o', color='red')
        plt.xlabel('Evaluation Interval')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (ε) Decay')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("Training progress plot saved to 'training_progress.png'")


def main():
    """Main training script"""
    print("=" * 60)
    print("Connect-4 Approximate Dynamic Programming")
    print("=" * 60)
    
    # Create trainer
    trainer = Connect4Trainer()
    
    # Train agents
    trainer.train(n_episodes=50000, eval_interval=5000)
    
    # Evaluate
    trainer.evaluate(n_games=1000)
    
    # Plot results
    trainer.plot_training_progress()
    
    print("\nTraining complete! You can now play against the trained agents.")


if __name__ == "__main__":
    main()