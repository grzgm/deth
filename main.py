import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9):
        """
        Initialize the Markov Decision Process.

        Parameters:
        - states (list): List of possible states.
        - actions (list): List of possible actions.
        - transition_probabilities (dict): Dictionary representing transition probabilities.
            Format: {state: {action: {next_state: probability}}}
        - rewards (dict): Dictionary representing immediate rewards.
            Format: {state: {action: {next_state: reward}}}
        - gamma (float): Discount factor for future rewards (default is 0.9).
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma

    def get_transition_probability(self, state, action, next_state):
        """
        Get the transition probability for a specific state, action, and next state.

        Parameters:
        - state: Current state.
        - action: Action taken.
        - next_state: Next state.

        Returns:
        - float: Transition probability.
        """
        return self.transition_probabilities[state][action].get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """
        Get the immediate reward for a specific state, action, and next state.

        Parameters:
        - state: Current state.
        - action: Action taken.
        - next_state: Next state.

        Returns:
        - float: Immediate reward.
        """
        return self.rewards[state][action].get(next_state, 0.0)

    def value_iteration(self, num_iterations=100):
        """
        Perform value iteration to find the optimal value function.

        Parameters:
        - num_iterations (int): Number of iterations for the value iteration algorithm.

        Returns:
        - dict: Optimal value function.
        """
        values = {state: 0.0 for state in self.states}

        for _ in range(num_iterations):
            new_values = {}
            for state in self.states:
                max_value = float('-inf')
                for action in self.actions:
                    action_value = sum(
                        self.get_transition_probability(state, action, next_state) *
                        (self.get_reward(state, action, next_state) +
                         self.gamma * values[next_state])
                        for next_state in self.states
                    )
                    max_value = max(max_value, action_value)
                new_values[state] = max_value

            values = new_values

        return values

# Example usage:
# Define states, actions, transition probabilities, and rewards
states = ['S', 'A', 'B', 'T']
actions = ['left', 'right']
transition_probabilities = {
    'S': {'left': {'S': 0.9, 'A': 0.1}, 'right': {'S': 0.1, 'A': 0.9}},
    'A': {'left': {'S': 0.8, 'A': 0.2}, 'right': {'B': 1.0}},
    'B': {'left': {'A': 0.8, 'B': 0.2}, 'right': {'T': 1.0}},
    'T': {'left': {'B': 1.0}, 'right': {'B': 1.0}}
}
rewards = {
    'S': {'left': {'S': 0, 'A': 0}, 'right': {'S': 0, 'A': 0}},
    'A': {'left': {'S': 0, 'A': 0}, 'right': {'B': 1.0}},
    'B': {'left': {'A': 0, 'B': 0}, 'right': {'T': 10.0}},
    'T': {'left': {'B': 0}, 'right': {'B': 0}}
}

# Create an MDP instance
mdp = MDP(states, actions, transition_probabilities, rewards)

# Perform value iteration to find the optimal value function
optimal_values = mdp.value_iteration()

# Print the optimal values for each state
print("Optimal Values:")
for state, value in optimal_values.items():
    print(f"{state}: {value}")
