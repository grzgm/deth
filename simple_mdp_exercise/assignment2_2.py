from assignment1_2 import MDP

states = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1), (4, 2)]
actions = ['l', 'r', 'u', 'd']

transition_probabilities = {}

for x in range(5):
    for y in range(3):
        print((x, y), end=", ")
    print("")

for state in states:
    for action in actions:
        transition_probabilities[state] = {
            action: {

            }
        }

transition_probabilities = {
    '0': {'r': {'1': 1}},
    '1': {'l': {'0': 1},
          'r': {'2': 1}},
    '2': {'l': {'1': 1},
          'r': {'3': 1}},
    '3': {'l': {'2': 1},
          'r': {'4': 1}},
    '4': {'l': {'3': 1}},
}

rewards = {
    '1': {'l': {'0': -1}},
    '3': {'r': {'4': 1}},
}

terminal_states = ['0', '4']

mdp = MDP(states, actions, transition_probabilities, rewards, terminal_states, '0')
