import numpy as np
import random
seed = 5
random.seed(seed)
np.random.seed(seed)

from racetrack_env import RacetrackEnv

# Instantiate environment object.
env = RacetrackEnv()

# Initialise/reset environment.
state = env.reset()
env.render()
print("Initial State: {}".format(state))


# Please write your code for Exercise 2a in this cell or in as many cells as you want ABOVE this cell.
# You should implement your modified TD learning agent here.
# Do NOT delete or duplicate this cell.

# YOUR CODE HERE

class DoubleDynaQAgent:
    # initialising function
    def __init__(self, env, alpha=0.15, gamma=0.9, epsilon=0.1, n_planning=15):  # , kappa = 0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration
        self.n_planning = n_planning  # Number of planning steps. Set to 10 to balance computation with reward. Around 20 would be another whole episode planned
        self.Q1 = {}  # Action-value function
        self.Q2 = {}  # Double Q
        self.model = {}  # Transition and reward model
        # self.kappa = kappa  # Small positive scaling factor for bonus reward -  Dyna Q plus method removed as environment is static
        # self.model_visits = {}  # Model visits counter -   Dyna Q plus method removed as environment is static

    # Function implementing the epsilon greedy policy.
    def policy(self, state):
        # Some code adapted from Joshua Evans
        # Get the list of actions
        available_actions = self.env.get_actions().copy()

        # Shuffle the list of actions to act as a random tie-breaker, as
        random.shuffle(available_actions)

        # initialise new
        if state not in self.Q1:
            self.Q1[state] = {action: 0.0 for action in available_actions}
        if state not in self.Q2:
            self.Q2[state] = {action: 0.0 for action in available_actions}

        # epsilon greedy policy:
        if random.random() < self.epsilon:
            return random.choice(available_actions)  # random move
        else:
            return max((self.Q1[state][a] + self.Q2[state][a], a) for a in available_actions)[1]  # greedy move

    def update_Q(self, state, action, reward, next_state):
        # initialise new
        if next_state not in self.Q1:
            self.Q1[next_state] = {a: 0.0 for a in self.env.get_actions()}
        if next_state not in self.Q2:  # Ensure Q2 is also initialized
            self.Q2[next_state] = {a: 0.0 for a in self.env.get_actions()}

        # Double q-learning updates
        if random.random() < 0.5:
            next_action = max((self.Q1[next_state][a], a) for a in self.env.get_actions())[1]
            td_target = reward + self.gamma * self.Q2[next_state][next_action]
            td_error = td_target - self.Q1[state][action]  # get difference
            self.Q1[state][action] += self.alpha * td_error  # add to q table
        else:
            next_action = max((self.Q2[next_state][a], a) for a in self.env.get_actions())[1]
            td_target = reward + self.gamma * self.Q1[next_state][next_action]
            td_error = td_target - self.Q2[state][action]  # get difference
            self.Q2[state][action] += self.alpha * td_error  # add to q table

        # get bonus with tau and kappa
        # bonus_reward = self.kappa * np.sqrt(self.model_visits.get((state, action), 0)) Dyna Q plus method removed as environment is static
        # self.Q[state][action] += self.alpha * (td_error + bonus_reward) Dyna Q plus method removed as environment is static

    def update_model(self, state, action, reward, next_state):
        # initialise new
        if state not in self.model:
            self.model[state] = {}
        if action not in self.model[state]:
            self.model[state][action] = []

        # add observed transition to model
        self.model[state][action].append((reward, next_state))

        # update visits for tau rewards
        # self.model_visits[(state, action)] = self.model_visits.get((state, action), 0) + 1  Dyna Q plus method removed as environment is static

    def planning_step(self):
        # randomly sample a state-action pair from model
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))

        # randomly sample a transition
        reward, next_state = random.choice(self.model[state][action])

        # update
        self.update_Q(state, action, reward, next_state)

    def generate_episode(self, num_episodes):
        # empty list of rewards
        rewards = []

        # iterate for every episode
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            # until terminal node
            while True:
                action = self.policy(state)  # select action from epsilon greedy
                next_state, reward, terminal = self.env.step(action)  # execute action
                self.update_Q(state, action, reward, next_state)  # update Q-values
                self.update_model(state, action, reward, next_state)  # update the model
                episode_reward += reward  # update reward for episode

                # condition if terminal ended
                if terminal:
                    break

                # set next_state to current state
                state = next_state

                # perform planning steps # (f) in Page 164 Chapter 8, Planning and Learning with Tabular Methods (Sutton and Barto)
                for _ in range(self.n_planning):
                    self.planning_step()

            # add total reward to list of rewards
            rewards.append(episode_reward)
        return rewards

    # reset Q values and model
    def reset_Q(self):
        self.Q1 = {}
        self.Q2 = {}
        self.model = {}


# Run 20 agents for 150 episodes, recording per-episode reward.
num_agents = 20
num_episodes = 150

modified_agent_rewards = []

# iterate over each agent
for _ in range(num_agents):
    # new instance
    agent = DoubleDynaQAgent(env)

    # reset Q values and models
    agent.reset_Q()

    # train agent for 150 episodes
    rewards = agent.generate_episode(num_episodes)

    # add the rewards obtained by agent
    modified_agent_rewards.append(rewards)


from racetrack_env import plot_modified_agent_results
from racetrack_env import simple_issue_checking

# Checking Modified Agent Results for Obvious Issues.
simple_issue_checking(modified_agent_rewards, modified_agent = True)
plot_modified_agent_results(modified_agent_rewards)

# References:
# Racetrack environment code by Dr Joshua Evans MC Control lab (racetrack_env.py)
# Off-policy TD Control Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.5 p.131)
# Double Q-learning Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 6.7 p.136)
# Tabular Dyna-Q Algorithm (Reinforcement Learning, Sutton & Barto, 2018, Section 8.2 p.164)