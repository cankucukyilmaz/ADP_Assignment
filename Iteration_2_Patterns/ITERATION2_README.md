# Iteration 2: Pattern-Based Features - Summary

## What Changed from Iteration 1?

### Features (Most Important!)
**Before (Iteration 1):**
- 42 binary cell indicators
- φ(S) = [1 if my piece in cell 0, 1 if my piece in cell 1, ..., 1 if my piece in cell 41]
- 42 weights to learn

**After (Iteration 2):**
- 4 pattern counts
- φ(S) = [my 2-in-a-rows, my 3-in-a-rows, opponent 2-in-a-rows, opponent 3-in-a-rows]
- Only 4 weights to learn!

### What Makes a Pattern "Extendable"?
A pattern within a window of 4 cells is extendable if:
- Has exactly 2 or 3 of our pieces
- Has NO opponent pieces (not blocked)
- Has empty spaces to extend to 4-in-a-row

Examples:
- [X, X, _, _] → Extendable 2-in-a-row ✓
- [X, X, X, _] → Extendable 3-in-a-row ✓
- [O, X, X, _] → NOT extendable (blocked by opponent) ✗
- [X, X, O, O] → NOT extendable (blocked both ends) ✗

### Other Changes
- Learning rate: α = 0.1 (was 0.01 in first iteration)
- Added weight display after evaluation to see what agents learned

## Expected Improvements

### Why Pattern Features Should Work Better:

1. **Recognizes Threats**: Agent can now "see" when opponent has 3-in-a-row
2. **Recognizes Opportunities**: Agent knows when it has 3-in-a-row (can win!)
3. **Fewer Parameters**: 4 weights vs 42 → faster learning, less overfitting
4. **More Interpretable**: Can understand what agent values by looking at weights

### What to Expect:

**Player 2 should win more often!** Why?
- With 42 cell features, Player 2 couldn't learn to defend properly
- With pattern features, Player 2 can recognize "opponent has 3-in-a-row → BLOCK!"

**Possible outcomes:**
- Player 1: 60-70% wins (first-move advantage)
- Player 2: 25-35% wins (much better than 9%!)
- Draws: 0-5% (still rare in Connect-4)

### Learned Weights - What to Look For:

After training, check the printed weights:

**Expected weight signs:**
- θ₁ (my 2-in-a-rows): **Positive** (building toward win)
- θ₂ (my 3-in-a-rows): **Strongly positive** (one move from winning!)
- θ₃ (opponent 2-in-a-rows): **Negative** (opponent building threat)
- θ₄ (opponent 3-in-a-rows): **Strongly negative** (must block!)

**Example of good learning:**
```
Player 1 weights:
  [0] My 2-in-a-rows: 0.5234
  [1] My 3-in-a-rows: 2.8901  ← Very high! Winning is valuable
  [2] Opp 2-in-a-rows: -0.3421
  [3] Opp 3-in-a-rows: -2.1234  ← Very negative! Must defend
```

## How to Run

### Option 1: Train and Evaluate
```bash
python connect4_adp.py
```
This will:
- Train for 50,000 episodes
- Print progress every 5,000 episodes
- Evaluate on 1,000 games
- **Show learned weights** (NEW!)
- Save training plot

### Option 2: Test Pattern Detection First
```bash
python test_patterns.py
```
This verifies that pattern counting works correctly with various test cases.

### Option 3: Interactive Play
```bash
python play_interactive.py
```
Play against the trained agent to see if it's smarter!

## For Your Report

### What to Compare:
**Iteration 1 vs Iteration 2:**
- Win rates (Player 1, Player 2, Draws)
- Training plots (did agents learn faster?)
- Learned weights (are they interpretable?)
- Playing strength (try playing both versions!)

### Questions to Answer:
1. Did Player 2 improve? By how much?
2. Are the learned weights intuitive? (positive for offensive, negative for defensive)
3. Does the agent now block 3-in-a-rows?
4. Does the agent prioritize its own 3-in-a-rows?
5. Is training faster (fewer episodes needed)?

### Expected Observations:
- "With pattern features, Player 2's win rate increased from 9% to 30%"
- "The learned weights show that both agents value 3-in-a-rows highly (θ₂ ≈ 2.5) and recognize opponent threats (θ₄ ≈ -2.0)"
- "During interactive play, agents now consistently block immediate threats"

## Next Steps (Iteration 3)

After running this iteration, consider:

1. **Better exploration**: Exponential epsilon decay instead of linear
2. **More features**: Add center control, directional patterns
3. **Combined features**: Use both cell indicators AND patterns
4. **Different learning rate**: Try α = 0.05 or adaptive learning rate

## Notes

- Pattern detection checks all 4 directions: horizontal, vertical, both diagonals
- Total possible patterns per board: much fewer than 3^42 states!
- Faster training expected due to smaller parameter space

Good luck with Iteration 2! Run it and let me know the results.
