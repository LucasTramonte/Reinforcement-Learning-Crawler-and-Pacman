# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

# Prefer the close exit (+1), risking the cliff (-10)

def question2a():
    #The agent prioritizes immediate rewards without worrying about randomness, thus risking the cliff for the close exit.
    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    #The agent still prefers the close exit but with a slight randomness that makes the cliff risky, thus promoting safer paths.
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    #The agent values future rewards highly and takes the risk to achieve the distant exit quickly.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -2
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    #The agent aims for the distant exit but with caution due to the added randomness, avoiding the cliff.
    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    # Avoid both exits and the cliff (so an episode should never terminate)
    #The agent receives a positive reward for each step, encouraging continuous exploration without termination.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
