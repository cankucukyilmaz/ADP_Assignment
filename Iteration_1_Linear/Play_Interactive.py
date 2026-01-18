"""
Interactive Connect-4 game against trained ADP agent
"""

from Baseline_Connect4_adp import Connect4Game, ADPAgent, GameConfig, Connect4Trainer
import numpy as np


def play_interactive(trainer: Connect4Trainer, human_player: int = 1):
    """
    Play interactively against a trained agent
    
    Args:
        trainer: Trained Connect4Trainer with learned agents
        human_player: 1 or 2 (which player is human)
    """
    game = Connect4Game()
    game.reset()
    
    computer_agent = trainer.agent2 if human_player == 1 else trainer.agent1
    
    print("\n" + "=" * 60)
    print(f"You are Player {human_player} ({'X' if human_player == 1 else 'O'})")
    print(f"Computer is Player {3 - human_player} ({'O' if human_player == 1 else 'X'})")
    print("=" * 60)
    
    game.render()
    
    while not game.done:
        if game.current_player == human_player:
            # Human turn
            valid_actions = game.get_valid_actions()
            print(f"Your turn! Valid columns: {valid_actions}")
            
            while True:
                try:
                    action = int(input("Enter column (0-6): "))
                    if action in valid_actions:
                        break
                    else:
                        print(f"Invalid! Column {action} is full. Try again.")
                except (ValueError, KeyboardInterrupt):
                    print("Please enter a valid column number (0-6)")
            
            game.make_move(action)
        else:
            # Computer turn
            print("Computer is thinking...")
            action = computer_agent.select_action(game, epsilon=0.0)  # Greedy
            print(f"Computer plays column {action}")
            game.make_move(action)
        
        game.render()
    
    # Game ended
    if game.winner == 0:
        print("It's a draw!")
    elif game.winner == human_player:
        print("🎉 Congratulations! You won!")
    else:
        print("Computer wins! Better luck next time.")
    
    print()


def main():
    """Main interactive play script"""
    print("Loading trained agents...")
    
    # Option 1: Train new agents
    trainer = Connect4Trainer()
    trainer.train(n_episodes=50000, eval_interval=10000)
    
    # Option 2: If you've saved weights, you could load them here
    # trainer.agent1.theta = np.load('agent1_weights.npy')
    # trainer.agent2.theta = np.load('agent2_weights.npy')
    
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("  1. Play as Player 1 (goes first)")
        print("  2. Play as Player 2 (goes second)")
        print("  3. Watch computer vs computer")
        print("  4. Quit")
        print("=" * 60)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            play_interactive(trainer, human_player=1)
        elif choice == '2':
            play_interactive(trainer, human_player=2)
        elif choice == '3':
            print("\nComputer vs Computer demo:")
            game = Connect4Game()
            game.reset()
            game.render()
            
            while not game.done:
                agent = trainer.agent1 if game.current_player == 1 else trainer.agent2
                action = agent.select_action(game, epsilon=0.0)
                print(f"Player {game.current_player} plays column {action}")
                game.make_move(action)
                game.render()
                input("Press Enter to continue...")
            
            if game.winner == 0:
                print("Draw!")
            else:
                print(f"Player {game.winner} wins!")
        elif choice == '4':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()