import copy

from assignment1_2 import MDP

# # print the states
# for y in range(3):
#     for x in range(5):
#         print((x, y), end=", ")
#     print("")

# grid world (x, y)
states = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
          (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
          (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
# right, left, up, down
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

transition_probabilities = {}

# generate transition_probabilities
for state in states:
    transition_probabilities[state] = {}
    for action in actions:
        if 0 <= state[0] + action[0] <= 4 and 0 <= state[1] + action[1] <= 2:
            transition_probabilities[state][action] = {(state[0] + action[0], state[1] + action[1]): 1}

# terminal states
for state in [(1, 2), (2, 2), (3, 2), (4, 2)]:
    transition_probabilities[state] = {}

# same structure for rewards
rewards = copy.deepcopy(transition_probabilities)

for state in rewards:
    amount_of_actions = len(rewards[state])
    for action in rewards[state]:
        for next_state in rewards[state][action]:
            if next_state in [(1, 2), (2, 2), (3, 2)]:
                rewards[state][action][next_state] = -1
            if next_state == (4, 2):
                rewards[state][action][next_state] = 1
            else:
                rewards[state][action][next_state] = 0

# start
start_state = (0, 2)

mdp = MDP(states, actions, transition_probabilities, rewards, start_state, random_termination=0.3,
          cost_of_living=-1.5)
