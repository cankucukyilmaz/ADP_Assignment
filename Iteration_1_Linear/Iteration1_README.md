# Connect-4 Approximate Dynamic Programming Implementation

## Overview
This implementation solves Connect-4 using Approximate Dynamic Programming (ADP) with Temporal Difference (TD) learning and linear value function approximation.

## File Structure

### `Baseline_Connect4_adp.py` - Main Implementation
Contains all core components:

#### 1. **GameConfig**
- Configuration dataclass for game parameters (6x7 board, connect-4)

#### 2. **Connect4Game**
The game environment that handles:
- Board state management (6x7 grid)
- Valid action checking (which columns are not full)
- Move execution (drops chip with gravity)
- Win detection (horizontal, vertical, diagonal)
- Drawing and game termination

#### 3. **ADPAgent**
The learning agent that:
- **Features**: Extracts 42 binary indicators (1 if agent's chip in that cell, 0 otherwise)
- **Value Function**: V(S) = θᵀφ(S) (linear approximation)
- **Action Selection**: Epsilon-greedy policy
  - Exploration: random action with probability ε
  - Exploitation: action with highest post-decision value
- **Learning**: TD update rule
  ```
  θ ← θ + α × δ × φ(S^a)
  where δ = target - V(S^a)
  ```

#### 4. **Connect4Trainer**
Manages training process:
- Creates two agents (one for each player)
- Runs training episodes with epsilon decay
- Tracks statistics (win rates, epsilon values)
- Evaluates trained policies
- Visualizes training progress

## Key Concepts Implemented

### 1. Post-Decision States
- After you make a move but before opponent moves
- Allows you to evaluate YOUR action before seeing opponent's response
- Critical for 2-player games with alternating turns

### 2. TD Learning with Value Function Approximation
```
Episode flow:
  1. Agent 1 selects action a₁ → Post-decision state S₁^a₁
  2. Agent 2 selects action a₂ → Post-decision state S₂^a₂  
  3. Agent 1 selects action a₁' → Post-decision state S₁^a₁' (or terminal)
  ...
  
Update for Agent 1's first move:
  Target = V(S₁^a₁') if game continues, or R if game ended
  θ₁ ← θ₁ + α × (Target - V(S₁^a₁)) × φ(S₁^a₁)
```

### 3. Epsilon-Greedy with Decay
- Start: ε = 1.0 (pure exploration)
- End: ε = 0.1 (mostly exploitation)
- Decay: Linear over training episodes
- Evaluation: ε = 0.0 (pure exploitation)

### 4. Separate Value Functions
- Agent 1 has its own θ₁ and sees rewards from its perspective
  - Win: +1, Loss: -1, Draw: 0
- Agent 2 has its own θ₂ and sees rewards from its perspective
  - Win: +1, Loss: -1, Draw: 0

## Usage

### Training and Evaluation
```bash
python connect4_adp.py
```
This will:
1. Train both agents for 50,000 episodes
2. Print progress every 5,000 episodes
3. Evaluate on 1,000 test games with greedy policy
4. Save training progress plot

### Interactive Play
```bash
python play_interactive.py
```
This lets you:
- Play as Player 1 or Player 2
- Watch computer vs computer
- Test the trained agents

## Parameters You Can Adjust

### In ADPAgent:
- `alpha`: Learning rate (default: 0.01)
  - Higher: faster learning but less stable
  - Lower: more stable but slower
  
- `initial_epsilon`: Starting exploration rate (default: 1.0)

### In Connect4Trainer:
- `n_episodes`: Number of training games (default: 50,000)
- `eval_interval`: How often to print progress (default: 5,000)
- `epsilon_start`: Initial ε (default: 1.0)
- `epsilon_end`: Final ε (default: 0.1)

## For Your Assignment

### Iteration 1 (Current Implementation)
- ✅ 42 binary features (simplest approach)
- ✅ Linear VFA
- ✅ TD learning with post-decision states
- ✅ Epsilon-greedy with linear decay
- ✅ Training and evaluation
- ✅ Visualization

### Iteration 2 (Improvements to Try)
Ideas for improvement cycles:
1. **Better Features**:
   - Count 2-in-a-rows (extendable)
   - Count 3-in-a-rows (extendable)
   - Opponent's threatening positions
   - Control of center columns

2. **Exploration Strategy**:
   - Exponential epsilon decay
   - Different decay schedules
   - Self-play with different exploration rates

3. **Learning Parameters**:
   - Different learning rates
   - Discount factor γ (if using n-step returns)

4. **Architecture**:
   - Non-linear features (products of base features)
   - Different initialization strategies

### Reporting Requirements
The code already tracks:
- ✅ Win percentage for Player 1
- ✅ Win percentage for Player 2
- ✅ Draw percentage
- ✅ Training progress visualization

For your report, you'll want to:
1. Run evaluation multiple times and report confidence intervals
2. Compare before/after each improvement
3. Analyze which features matter most (look at learned weights)

## Understanding the Code

### Key Questions to Ask Yourself:
1. **Why post-decision states?**
   - Because opponent's move is uncertain when you decide
   - Separates your decision from opponent's randomness

2. **Why separate agents?**
   - Each player needs to learn from their own perspective
   - Rewards are opposite (zero-sum game)

3. **Why epsilon decay?**
   - Early: need exploration to find good states
   - Late: know enough to exploit learned values

4. **Why linear VFA?**
   - Simple, interpretable, fast
   - Good baseline before trying complex features

### Debugging Tips:
- If one player always wins: check if rewards are assigned correctly
- If both players play randomly: epsilon might not be decaying, or alpha too high
- If training is slow: reduce n_episodes for testing, increase for final run
- Check learned weights: `print(trainer.agent1.theta)` - which features have high magnitudes?

## Next Steps
1. **Run the code** and observe results
2. **Understand the flow**: trace through one episode manually
3. **Experiment**: change alpha, epsilon, n_episodes
4. **Design better features** for iteration 2
5. **Compare results** quantitatively

Good luck with your assignment!