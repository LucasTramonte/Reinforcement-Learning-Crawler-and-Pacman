# Reinforcement-Learning-Crawler-and-Pacman
Fourth project of the Artificial Intelligence course at CentraleSupélec. Implementing value iteration and Q-learning to a simulated robot controller (Crawler) and Pacman.

http://ai.berkeley.edu.

## Contents
- [Overview](#Overview)
- [ValueIteration](#Value_Iteration)
- [Policies](#Policies)
- [Q-Learning](#Q-Learning)
- [EpsilonGreedy](#Epsilon_Greedy)
- [Q-LearningandPacman](#Q-Learning_and_Pacman)
- [ApproximateQ-Learning](#Approximate_Q-Learning)
- [DeepQ-Learning](#Deep_Q-Learning)

## Overview


## Value Iteration


**File:** valueIterationAgents.py 
**class:** `ValueIterationAgent`

To run:
```bash
python gridworld.py -a value -i 5
```

## Policies

**File:** analysis.py
**Function:** question2a() through question2e()

To run:

```bash
python autograder.py -q q2
```

- Discount Factor (γ):

The discount factor determines the importance of future rewards. A value close to 1 means the agent will consider future rewards nearly as important as immediate rewards, while a value close to 0 means the agent will prioritize immediate rewards.

Choice: A higher discount factor (0.9) was chosen for policies aiming for the distant exit (+10), as these policies should prioritize the long-term reward. Conversely, a lower discount factor (0.3) was used for policies focusing on the close exit (+1) to make the agent prefer immediate rewards.

- Noise:

Noise represents the randomness in the agent's actions. A noise value of 0 means the agent's actions are deterministic, while higher values introduce more randomness, simulating an uncertain environment.

Choice: For policies that should avoid the cliff, a moderate noise level (e.g., 0.2 or 0.3) was introduced. This encourages the agent to avoid high-risk paths, as the possibility of unintended actions makes risky routes less appealing. For riskier policies, the noise was set to 0, ensuring the agent follows the shortest path without deviation.

- Living Reward:

The living reward is a reward (or penalty) the agent receives at each time step. Positive values encourage the agent to continue exploring, while negative values push the agent to reach a terminal state quickly.

Choice: Negative living rewards were used to incentivize the agent to reach terminal states quickly. For example, a living reward of -2 was used when aiming for the distant exit with a risk, making the agent prefer reaching the +10 reward faster. Zero or slightly positive living rewards were applied to make the agent more cautious, promoting safer routes and discouraging quick termination unless it ensures safety.

## Q-Learning

**File:** qlearningAgents.py
**Functions:** computeValueFromQValues(self, state) ; update(self, state, action, nextState, reward) ; getQValue(self, state, action) and computeActionFromQValues(self, state)

To run:

```bash
python gridworld.py -a q -k 5 -m
```

- getQValue: Returns the Q-value for a given state-action pair. If the pair has not been encountered before, it returns a default value of 0.0.

- computeValueFromQValues: Computes the maximum Q-value for all legal actions in a given state. If there are no legal actions, it returns 0.0, indicating a terminal state.

- computeActionFromQValues: Determines the best action to take in a given state by selecting the action with the highest Q-value. In the case of a tie, it randomly chooses among the best actions to ensure better exploration.

- update: Updates the Q-value for a state-action pair using the reward received and the discounted value of the next state's optimal action. This update rule is based on the Q-learning formula, which incorporates the learning rate (alpha) and the discount factor (gamma).

These functions collectively enable the Q-Learning agent to iteratively improve its policy by learning from the rewards and transitions it experiences in the environment. Through this process, the agent adjusts its Q-values to better estimate the expected future rewards for state-action pairs, leading to increasingly optimal decision-making.

## Epsilon Greedy

**File:** qlearningAgents.py
**Function:** getAction(self, state)

To run

```bash
python crawler.py
```

getAction: Decides the action to take based on the current state. With a probability defined by epsilon, it selects a random action to explore; otherwise, it chooses the best action according to the Q-values. This balance between exploration and exploitation is crucial for effective learning.

## Q-Learning and Pacman

**File:** qlearningAgents.py
**Class:** PacmanQAgent(QLearningAgent)

To run:

```bash
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

## Approximate Q-Learning

**File:** qlearningAgents.py
**Class:** ApproximateQAgent

To run:

```bash
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

```

An Approximate Q-Learning agent was implemented, with the getQValue and update methods modified to use feature-based state representations and update weights accordingly. Handling of transitions with no legal actions was ensured, and the agent was tested with various feature extractors to validate its learning effectiveness. Finally, a mechanism to display the learned weights after training was included.

## Deep Q-Learning

A Deep Q-Network (DQN) for reinforcement learning was enhanced to optimize Pacman's performance. The model uses four neural network layers with sizes [256, 128, 64, action_dim] to predict Q-values for state-action pairs. Key hyperparameters were set: learning rate (1), training games (4000), and batch size (128). By computing the squared loss between predicted and target Q-values and applying gradient descent, the agent's game performance was significantly improved.

**File:** model.py
**Class:** DeepQNetwork

To run:

```bash
python pacman.py -p PacmanDeepQAgent -x [numGames] -n [numGames + 10] -l testClassic
```

