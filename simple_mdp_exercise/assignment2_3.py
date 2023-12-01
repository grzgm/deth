import copy

from assignment1_2 import MDP

# print the states
for y in range(3):
    for x in range(5):
        for z in [True, False]:
            print((x, y, z), end=", ")
    print("")

# grid world (x, y)
states = [(2, 0, True), (2, 0, False),
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
transition_probabilities[(2, 0, False)][(0, 1)] = {(2, 1, True): 1}

transition_probabilities[(2, 1, False)][(0, -1)] = {(2, 0, True): 1}
transition_probabilities[(2, 0, True)][(0, 1)] = {(2, 1, True): 1}
transition_probabilities[(2, 1, True)][(0, -1)] = {(2, 0, True): 1}

transition_probabilities[(3, 1, True)][(1, 0)] = {(4, 1, True): 1}
# transition_probabilities[(3, 1, False)][(1, 0)] = {(4, 1, True): 0}

# same structure for rewards
rewards = copy.deepcopy(transition_probabilities)

# for state in rewards:
#     amount_of_actions = len(rewards[state])
#     for action in rewards[state]:
#         for next_state in rewards[state][action]:
#             if next_state in [(1, 2), (2, 2), (3, 2), (4, 2)]:
#                 rewards[state][action][next_state] = -1
#             else:
#                 rewards[state][action][next_state] = 0

# terminal states
terminal_states = [(0, 1, True), (0, 1, False), (4, 1, True)]

# start
start_state = (0, 2)

mdp = MDP(states, actions, transition_probabilities, rewards, start_state, terminal_states, random_termination=0.3,
          cost_of_living=-1.5)
