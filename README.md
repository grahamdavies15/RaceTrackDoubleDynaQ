# DynaQ Learning and Double Q Learning

I implemented Double Q-Learning and the DynaQ algorithm. DynaQ combines model-based planning with real experience. By simulating transitions, it learns faster as seen in the comparative graphs. Double Q-learning mitigates the maximization bias by having two separate q-tables, which one randomly chooses a table to select an action from. By minimising bias, it improves the overall model performance. 
![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/assets/86721524/36768c79-a4e7-46f2-8d31-1ab4a1e1b92a)


After 30 episodes, the agent achieved returns similar to what Q-learning reached in 150 episodes, indicating significantly faster learning. Additionally, it achieved higher average returns, showing improved overall performance. The increase in learning speed can be predominantly attributed to increased exploration in model planning from DynaQ, while double q-learning maintained efficiency through avoiding maximization bias.
![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/assets/86721524/ded6f66d-b44c-4103-b37d-9b37fc8ef03e)


The environment was a the racetrack environment provided by Joshua Evans:
![image](https://github.com/grahamdavies15/RaceTrackDoubleDynaQ/assets/86721524/73491e9f-146e-4648-b97d-d6db0a3b35ef)

# References:
Racetrack environment code by Dr Joshua Evans (racetrack_env.py)
Basic Q-learning returns plot by Dr Joshua Evans (correct_returns_q.json)
Off-policy TD Control Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.5 p.131)
Double Q-learning Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.7 p.136)
Tabular Dyna-Q Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 8.2 p.164)

