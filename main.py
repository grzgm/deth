import random
from math import isclose

import numpy as np


class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, start_state, gamma=0.9,
                 eps=1e6, random_termination=0.0, cost_of_living=0.0):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.inspect_probabilities()
        self.rewards = rewards
        self.current_state = start_state

        self.gamma = gamma
        self.eps = eps
        self.random_termination = random_termination
        assert 0 <= self.random_termination <= 1
        self.cost_of_living = cost_of_living

        self.value = {}
        for state in states:
            self.value[state] = 0.0

    # def reset(self):
    #     for state in self.states:
    #         self.value[state] = 0.0

    def lookup_transition_probability(self, state: str, action: str, next_state: str):
        return self.transition_probabilities[state].get(action, {}).get(next_state, 0.0)

    def lookup_reward(self, state: str, action: str, next_state: str):
        return self.transition_probabilities[state].get(action, {}).get(next_state, 0)

    def inspect_probabilities(self):
        for state in self.transition_probabilities.values():
            for action in state.values():
                assert isclose(sum(action.values()), 1, abs_tol=1e-4)

    def step(self, action):
        if(len(transition_probabilities[self.current_state]) == 0):
            return None
        next_states = list(transition_probabilities[self.current_state][action].keys())
        probabilities = list(transition_probabilities[self.current_state][action].values())
        self.current_state = np.random.choice(next_states, p=probabilities)
        return self.current_state


    # def value(self, state: str):
    #     pass

    # def action_value(self, state: str, action: str):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)
    #
    # def policy(self, state):
    #     best_action = None
    #     best_value = None
    #     for action, next_states in self.transition_probabilities[state].items():
    #         for next_state in next_states:
    #             new_value = self.transition_probabilities[state][action][next_state] * self.value[state]
    #             if best_action == None:
    #                 best_action = action
    #                 best_value = new_value
    #             elif best_value < new_value:
    #                 best_action = action
    #                 best_value = new_value
    #
    #     return best_action

    # return random.choice(self.actions)

    def estimate_value(self):
        for _ in range(int(self.eps)):
            for state in self.states:
                self.value[state] = self.action_value(state, self.policy(state))


if __name__ == "__main__":
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

    mdp = MDP(states, actions, transition_probabilities, rewards, '1', random_termination=0.3, cost_of_living=-1.5)

    for step in range(100):
        new_state = mdp.step('r')
        print(mdp.current_state)
        if (new_state==None):
            break
