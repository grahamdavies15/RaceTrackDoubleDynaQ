# DynaQ Learning and Double Q Learning

I implemented Double Q-Learning and the DynaQ algorithm. DynaQ combines model-based planning with real experience. By simulating transitions, it learns faster as seen in the comparative graphs. Double Q-learning mitigates the maximization bias by having two separate q-tables, which one randomly chooses a table to select an action from. By minimising bias, it improves the overall model performance. 

![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/blob/main/images/returns.png)


After 30 episodes, the agent achieved returns similar to what Q-learning reached in 150 episodes, indicating significantly faster learning. Additionally, it achieved higher average returns, showing improved overall performance. The increase in learning speed can be predominantly attributed to increased exploration in model planning from DynaQ, while double q-learning maintained efficiency by avoiding maximization bias.

![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/blob/main/images/returns_cropped.png)


The environment was the racetrack environment provided by Joshua Evans:

![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/blob/main/images/environment.png)

# References:

Racetrack environment code by Dr Joshua Evans (racetrack_env.py)

Basic Q-learning returns plot by Dr Joshua Evans (correct_returns_q.json)

Off-policy TD Control Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.5 p.131)

Double Q-learning Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.7 p.136)

Tabular Dyna-Q Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 8.2 p.164)

