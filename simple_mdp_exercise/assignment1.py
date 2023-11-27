# 1.2
class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9, eps=1e+6):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.eps = eps
        self.value = {}
        for state in states:
            self.value[state] = 0.0

    def reset(self):
        for state in states:
            self.value[state] = 0.0

    def lookup_transition_probability(self, state: str, action: str, next_state: str):
        return self.transition_probabilities[state][action].get(next_state, 0.0)

    def lookup_reward(self, state: str, action: str, next_state: str):
        return self.transition_probabilities[state][action].get(next_state, 0)

    # def value(self, state: str):
    #     pass

    def action_value(self, state: str, action: str):
        next_states = self.transition_probabilities[state][action]
        return sum(self.lookup_transition_probability(state, action, next_state) * (
                    self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
                   in next_states)

    def calculate_value(self):
        for _ in range(int(self.eps)):
            for state in self.states:
                max_value = float('-inf')
                for action in self.actions:
                    expected_return = max(max_value, self.action_value(state, action))
                    max_value = max(max_value, expected_return)
                self.value[state] = max_value


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

    mdp = MDP(states, actions, transition_probabilities=transition_probabilities, rewards=rewards)  # create an MDP

    mdp.calculate_value()

    print(mdp.value)
