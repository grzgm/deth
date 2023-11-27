from assignment1_2 import MDP

states = ['0', '1', '2', '3', '4']
actions = ['l', 'r']

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
