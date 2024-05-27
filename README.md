# Reinforcement-Learning-Crawler-and-Pacman
Fourth project of the Artificial Intelligence course at CentraleSupélec. Implementing value iteration and Q-learning to a simulated robot controller (Crawler) and Pacman.

Trimmed down reinforcement project, and added Deep Q learning Pacman

https://centralesupelec.edunao.com/pluginfile.php/423959/course/section/60097/p2-search.html?time=1714677059278

http://ai.berkeley.edu.

## Contents
- [Overview](#Overview)
- [Question1](#Question1)
- [Question2](#Question2)
- [Question3](#Question3)
- [Question4](#Question4)
- [Question5](#Question5)
- [Question6](#Question6)
- [Question7](#Question7)


## Overview



## Question1

Value Iteration

**File:** `valueIterationAgents.py`  
**class:** `ValueIterationAgent`

To run:
```bash
python gridworld.py -a value -i 100 -k 10
```

## Question2

Policies

**File:** analysis.py
**Function:** question2a() through question2e()

To run:

```bash
python autograder.py -q q2
```

## Question3

Q-Learning

**File:** qlearningAgents.py
**Functions:** computeValueFromQValues(self, state) ; update(self, state, action, nextState, reward) ; getQValue(self, state, action) and computeActionFromQValues(self, state)

To run:

```bash
python gridworld.py -a q -k 5 -m
```

## Question4

Epsilon Greedy

**File:** qlearningAgents.py
**Function:** getAction(self, state)

To run

```bash
python crawler.py
```

## Question5
Q-Learning and Pacman

**File:** qlearningAgents.py
**Class:** PacmanQAgent(QLearningAgent)

To run:

```bash
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

## Question6
Approximate Q-Learning

**File:** qlearningAgents.py
**Class:** ApproximateQAgent

To run:

```bash
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

```

## Question7
Deep Q-Learning

**File:** model.py
**Class:** DeepQNetwork

To run:

```bash
python pacman.py -p PacmanDeepQAgent -x [numGames] -n [numGames + 10] -l testClassic
```

