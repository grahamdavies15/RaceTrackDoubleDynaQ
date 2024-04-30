# DynaQ Learning and Double Q Learning

I implemented Double Q-Learning and the DynaQ algorithm. DynaQ combines model-based planning with real experience. By simulating transitions, it learns faster as seen in the comparative graphs. Double Q-learning mitigates the maximization bias by having two separate q-tables, which one randomly chooses a table to select an action from. By minimising bias, it improves the overall model performance. 

After 30 episodes, the agent achieved returns similar to what Q-learning reached in 150 episodes, indicating significantly faster learning. Additionally, it achieved higher average returns, showing improved overall performance. The increase in learning speed can be predominantly attributed to increased exploration in model planning from DynaQ, while double q-learning maintained efficiency through avoiding maximization bias.
