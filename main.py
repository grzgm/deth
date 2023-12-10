import random
from math import isclose
import numpy as np


class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9,
                 eps=1e6, random_termination=0.0, cost_of_living=0.0):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.inspect_probabilities()
        self.rewards = rewards

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

    def lookup_transition_probability(self, state, action, next_state):
        return self.transition_probabilities.get(state, {}).get(action, {}).get(next_state, 0.0)

    def possible_actions(self, state):
        return [index[0] for index, probability in np.ndenumerate(self.transition_probabilities[state]) if
                probability != 0]

    def lookup_reward(self, state, action, next_state):
        return self.rewards[state, action, next_state]

    def inspect_probabilities(self):
        for state_index in range(len(states)):
            for action_index in range(len(actions)):
                probabilities_sum = sum(self.transition_probabilities[state_index, action_index, :])
                if probabilities_sum == 0:
                    continue
                else:
                    assert isclose(sum(self.transition_probabilities[state_index, action_index, :]), 1, abs_tol=1e-4)

    def step(self, current_state, action):
        probabilities = self.transition_probabilities[current_state, action]
        new_state = np.random.choice(len(probabilities), p=probabilities)
        reward = self.lookup_reward(current_state, action, new_state)
        is_terminal = len(self.possible_actions(new_state)) == 0
        return (new_state, reward, is_terminal)

    # def value(self, state):
    #     pass

    # def action_value(self, state, action):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)

    # def state_value(self, state, action):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)

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


# if __name__ == "__main__":
states = ['0', '1', '2', '3', '4', '5', '6']
actions = ['d', 'u']

transition_probabilities = np.zeros((len(states), len(actions), len(states)))

for state_index in range(len(states)):
    for action_index in range(len(actions)):
        for next_state_index in range(len(states)):
            if state_index in [0, 6]:
                continue
            elif action_index == 1 and state_index == next_state_index - 1:
                transition_probabilities[state_index, action_index, next_state_index] = 1
            elif action_index == 0 and state_index == next_state_index + 1:
                transition_probabilities[state_index, action_index, next_state_index] = 1

rewards = np.zeros((len(states), len(actions), len(states)))
rewards[1, actions.index('d'), 0] = 1
rewards[5, actions.index('u'), 6] = 0

state_value_array = []
action_value_array = np.zeros((len(states), len(actions)))

episodes = 1000
max_steps_in_episode = 1000
start_state_index = 2

# Learning Rate
alpha = 0.04
# Exploration Rate
epsilon = 1
# Discount Factor
gamma = 0.9

mdp = MDP(states, actions, transition_probabilities, rewards, random_termination=0.3, cost_of_living=-1.5)

for episode in range(episodes):
    previous_state = start_state_index
    total_reward = 0

    for step in range(max_steps_in_episode):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < epsilon:
            # random action
            action = np.random.choice(mdp.possible_actions(previous_state))
        else:
            # # best action
            # possible_actions_indexes = [index for index, element in enumerate(actions) if element in mdp.possible_actions(previous_state)]
            # # Extract the elements at the allowed indexes
            # allowed_elements = action_value_array[states.index(previous_state), possible_actions_indexes]
            #
            # # Find the index of the element with the highest value
            # max_index_in_allowed = np.unravel_index(allowed_elements.argmax(), allowed_elements.shape)
            #
            # # Convert the index from allowed_elements back to the original my_array
            # max_index_in_original = (states.index(previous_state), possible_actions_indexes[max_index_in_allowed[0]])
            # action = actions[max_index_in_original[1]]
            action = np.argmax(action_value_array[previous_state, :])

        new_state, reward, is_terminal = mdp.step(previous_state, action)

        action_value_array[previous_state, action] = (1 - alpha) * action_value_array[
            previous_state, action] + alpha * (reward + gamma * max(
            action_value_array[new_state, :]))

        previous_state = new_state
        print(mdp.possible_actions(new_state, ))
        # print(new_state, reward, is_terminal)
        print(new_state, reward)
        if (is_terminal):
            break

with np.printoptions(precision=3, suppress=True):
    print(action_value_array)