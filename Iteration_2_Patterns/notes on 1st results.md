1. Is Iteration 2 actually an improvement?
Yes and No.

The Improvement (Defense): In Iteration 1, Player 2 was helpless (1% win rate). In Iteration 2, Player 2 has successfully learned to not lose. A 94% draw rate means Player 2 is now consistently blocking Player 1's threats. This is a massive step up in "intelligence" compared to the random play of Iteration 1.

The Problem (Offense): The agents have likely "broken" their ability to win. A 94% draw rate in Connect-4 is extremely unnatural. It implies that both agents are actively avoiding winning lines or are stuck in a "defensive loop" where they prioritize blocking over building.

The "Smoking Gun" in your weights: Look closely at the weights in your screenshot (Iter_2_2nd.png):

Player 1 "My 3-in-a-rows": -0.0094 (Negative!)

Player 2 "My 3-in-a-rows": -0.0610 (Negative!)

This explains the draws. A negative weight means the agent dislikes this state. Both agents have learned to avoid creating 3-in-a-rows. If they avoid 3-in-a-rows, they can never get 4-in-a-row. They are playing "Don't Lose" instead of "Win"


2. Why did this happen?
This often happens in TD learning with simple features:

Correlation vs Causality: In early training (high exploration), having "3-in-a-row" might have often been followed by a loss (because the opponent blocked it and countered), so the agent associated "3-in-a-row" with "pain".

Blocked Patterns: If your feature counts any 3-in-a-row (even blocked ones like X X X O), the agent sees no value in them because they can never become a win.

Lack of "Killer Instinct": The reward for winning (+1) might be too distant or rare compared to the noise of random exploration.