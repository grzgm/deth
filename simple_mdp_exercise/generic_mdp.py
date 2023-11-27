class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, terminal_states = {}, gamma=0.9, eps=1e6, is_slippery=0.0,
                 cost_of_living=0):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.eps = eps
        self.is_slippery = is_slippery
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

    # def value(self, state: str):
    #     pass

    # def action_value(self, state: str, action: str):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)
    #
    # def policy_random(self):
    #     return random.choice(self.actions)
    #
    # def calculate_value(self):
    #     for _ in range(int(self.eps)):
    #         for state in self.states:
    #             self.value[state] = self.action_value(state, self.policy_random())