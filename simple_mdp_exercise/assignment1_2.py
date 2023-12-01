import random
from math import isclose


class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, start_state, gamma=0.9,
                 eps=1e6, random_termination=0.0, cost_of_living=0.0):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.inspect_probabilities()
        self.rewards = rewards
        self.start_state = start_state
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

    # def value(self, state: str):
    #     pass

    def action_value(self, state: str, action: str):
        next_states = self.transition_probabilities[state].get(action, {})
        return sum(self.lookup_transition_probability(state, action, next_state) * (
                self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
                   in next_states)

    def policy(self, state):
        best_action = None
        best_value = None
        for action, next_states in self.transition_probabilities[state].items():
            for next_state in next_states:
                new_value = self.transition_probabilities[state][action][next_state] * self.value[state]
                if best_action == None:
                    best_action = action
                    best_value = new_value
                elif best_value < new_value:
                    best_action = action
                    best_value = new_value

        return best_action

        # return random.choice(self.actions)

    def estimate_value(self):
        for _ in range(int(self.eps)):
            for state in self.states:
                self.value[state] = self.action_value(state, self.policy(state))


if __name__ == "__main__":
    states = ['s1', 's2', 's3']
    actions = ['a', 'b', 'c']

    transition_probabilities = {
        's1': {'a': {'s1': 0.5, 's2': 0.5}},
        's2': {'a': {'s1': 0.1, 's2': 0.4, 's3': 0.5},
               'b': {'s1': 0.6, 's2': 0.3, 's3': 0.1}},
        's3': {'b': {'s2': 1.0}}
    }

    rewards = {
        's1': {'a': {'s1': 5, 's2': 10}},
        's2': {'a': {'s1': -2, 's2': 1, 's3': 4},
               'b': {'s1': 3, 's2': 2, 's3': 8}},
        's3': {'b': {'s2': 100}}
    }

    mdp = MDP(states, actions, transition_probabilities, rewards, 's1')  # create an MDP

    mdp.estimate_value()

    print(mdp.value)