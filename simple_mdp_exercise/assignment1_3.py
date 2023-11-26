import random
from assignment1_2 import MDP

if __name__ == "__main__":
    states = ['s0', 's1', 's2']
    actions = ['a0', 'a1']

    transition_probabilities = {
        's0': {'a0': {'s0': 0.5, 's2': 0.5},
               'a1': {'s2': 1}},
        's1': {'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
               'a1': {'s1': 0.95, 's2': 0.05}},
        's2': {'a0': {'s0': 0.4, 's2': 0.6},
               'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}},
    }

    rewards = {
        's0': {},
        's1': {'a0': {'s0': 5}},
        's2': {
            'a1': {'s0': -1}},
    }

    mdp = MDP(states, actions, transition_probabilities=transition_probabilities, rewards=rewards)  # create an MDP

    mdp.calculate_value()

    print(mdp.value)
