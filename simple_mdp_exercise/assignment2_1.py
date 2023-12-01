from assignment1_2 import MDP

states = ['0', '1', '2', '3', '4']
actions = ['l', 'r']

# Terminal States have no actions
transition_probabilities = {
    '0': {},
    '1': {'l': {'0': 1},
          'r': {'2': 1}},
    '2': {'l': {'1': 1},
          'r': {'3': 1}},
    '3': {'l': {'2': 1},
          'r': {'4': 1}},
    '4': {},
}

rewards = {
    '1': {'l': {'0': -1}},
    '3': {'r': {'4': 1}},
}

terminal_states = ['0', '4']

mdp = MDP(states, actions, transition_probabilities, rewards, '2', terminal_states, random_termination=0.3, cost_of_living=-1.5)

mdp.estimate_value()

print(mdp.value)
