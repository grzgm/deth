import copy

from assignment1_2 import MDP

# # print the states
# for y in range(3):
#     for x in range(5):
#         for z in [True, False]:
#             print((x, y, z), end=", ")
#     print("")

# grid world (x, y)
# if can start in the key state, but not obtaining the key at the same time
# states = [(2, 0, True), (2, 0, False),
states = [(2, 0, True),
          (0, 1, True), (0, 1, False), (1, 1, True), (1, 1, False), (2, 1, True), (2, 1, False), (3, 1, True),
          (3, 1, False), (4, 1, True)]
# right, left, up, down
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

transition_probabilities = {}

# generate transition_probabilities
for state in states:
    transition_probabilities[state] = {}
    for action in actions:
        if action[1] == 0 and state[1] == 1 and (state[0] + action[0], state[1] + action[1], state[2]) in states:
            transition_probabilities[state][action] = {(state[0] + action[0], state[1] + action[1], state[2]): 1}

# this depends if we say that key is obtained by going to this state, or being in it
# transition_probabilities[(2, 0, False)][(0, 1)] = {(2, 1, True): 1}

# without the key to the key state
transition_probabilities[(2, 1, False)][(0, -1)] = {(2, 0, True): 1}
# with the key to the key state
transition_probabilities[(2, 1, True)][(0, -1)] = {(2, 0, True): 1}

# from the key state to state lower
transition_probabilities[(2, 0, True)][(0, 1)] = {(2, 1, True): 1}

# transition to the green state
transition_probabilities[(3, 1, True)][(1, 0)] = {(4, 1, True): 1}

# terminal states
transition_probabilities[(0, 1, True)] = {}
transition_probabilities[(0, 1, False)] = {}
transition_probabilities[(4, 1, True)] = {}

# Rewards
rewards = {}

for state in transition_probabilities:
    rewards[state] = {}
    for action in transition_probabilities[state]:
        for next_state in transition_probabilities[state][action]:
            if next_state in [(0, 1, True), (0, 1, False)]:
                rewards[state][action] = {next_state: -1}
            if next_state == (4, 1, True):
                rewards[state][action] = {next_state: 1}

# start
start_state = (0, 2)

mdp = MDP(states, actions, transition_probabilities, rewards, start_state, random_termination=0.3,
          cost_of_living=-1.5)
