# Some help to write an MDP yourself
# YOUR CODE SHOULD BE TESTED, HOW DO YOU TEST THIS CLASS?
# For your project: It is also possible and accepted for this course to use cutomizaiton of Gymnasium
# https://gymnasium.farama.org/v0.27.1/tutorials/gymnasium_basics/environment_creation/
# Teaching how to do this is not focus of the course.

#%%
class MDP:
    def __init__(self, states, actions, transitions=None, rewards=None, gamma=0.9, eps=1e-6):
        pass 


# Question: terminal states are not an input, how does the code "detect" a terminal state? 
# Example usage (same as before):
"""
to read nested dictionaries like below, you can use code like:

for state, action_dict in transitions.items():
    for action, next_states in action_dict.items():

"""

transitions = {
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


"""
make sure the sum of input probabilities are close to 1 (do not use == 1, use isclose)

You need isclose to make sure probabilites are close to 1.
    from math import isclose
    a = 1.0
    b = 1.00000001
    assert isclose(a, b, abs_tol=1e-4)

If needed you can also normalize the probabilites (in addition and after checking they are already close to one)    
"""



mdp = MDP(states, actions, transitions=transitions, rewards=rewards) # create an MDP

state = mdp.reset() # reset/re-initialize
actions_available = mdp.get_actions() # available action in the current state of MDP
new_state, reward, terminated, truncated, info = mdp.step(actions_available[0]) # execute an action in the current state of MDP
mdp.state_space.n # total size of states
mdp.action_space.n # total size of actions

