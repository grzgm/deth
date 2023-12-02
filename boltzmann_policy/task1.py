import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

MOVING = True  # extra cleaner plot with moving-average of reward
if MOVING:
    from scipy.signal import savgol_filter

class Step:
    def __init__(self, state=None, action=None, next_state=None):
        self.state = state
        self.action = action
        self.next_state = next_state


def frozen_lake(frequent_rewards=True, random_termination=True, tau=1):
    # hyper parameters
    learning_rate = 0.1  # impacts how fast we update our estimates
    discount_factor = 0.9  # gamma, impacts the return calculations
    episodes = 4 * 1000  # number of episodes to learn, keep this a multiple of four for nice plotting
    t = 100  # Maximum steps in an episode
    cost_of_living = - 0.01  # used when frequent_rewards = True, incentive the agent for efficiency by incurring a cost to each move

    # Choose environment
    env = gym.make("FrozenLake8x8-v1", is_slippery=random_termination)

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []
    q_values_at_intervals = []  # Store Q-values at intervals

    # Task 2
    episode_sequences = np.empty([episodes, t], dtype=object)
    hit_goal_first_time = (False, -1)

    for episode in range(episodes):
        state, prob = env.reset()
        total_reward = 0

        for step in range(t):
            # Choose action based on softmax policy
            modified_vector = Q[state, :] / tau
            exp_vector = np.exp(modified_vector - np.max(modified_vector))
            action_probabilities = exp_vector / np.sum(exp_vector)

            action = np.random.choice(env.action_space.n, p=action_probabilities)

            next_state, reward, terminated, truncated, info = env.step(action)

            # Storing episode history
            step_obj = Step(state, action, next_state)
            episode_sequences[episode, step] = step_obj

            # check if hit the goal
            if reward > 0 and not hit_goal_first_time[0]:
                hit_goal_first_time = (True, episode)

            if frequent_rewards:
                if terminated & (reward == 0):
                    reward = reward - 1

                reward = reward + cost_of_living

            # Update Q-value using Q-learning equation
            Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            total_reward += reward
            state = next_state

            if terminated:
                # if reward == 1 + cost_of_living:
                #     print(f"Episode {episode} finished after {step + 1} steps. Success!")
                break

        rewards_per_episode.append(total_reward)

        # Store Q-values at intervals (e.g., every 100 episodes)
        if (episode + 1) % (episodes // 4) == 0:
            q_values_at_intervals.append(np.copy(Q))  # Store a copy of Q-values

    # Plotting rewards per episode
    plt.figure(figsize=(7, 5))
    plt.plot(rewards_per_episode, label='Total Reward')
    plt.title('Rewards per Episode')
    plt.suptitle(f'is_slippery: {random_termination}, frequent_rewards: {frequent_rewards}, tau: {tau}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # the plot is too noisy, you can use the scipy package to calculate a moving average
    if MOVING:
        moving_average_window = 20
        moving_averages = savgol_filter(rewards_per_episode, moving_average_window, 3)
        plt.plot(moving_averages, label=f'Moving Average (Window {moving_average_window})', color='orange')

    plt.legend()
    plt.show()

    return (hit_goal_first_time, episode_sequences)

    # # Plotting the heatmap of Q-values at intervals
    # fig, ax = plt.subplots(1, len(q_values_at_intervals), figsize=(15, 5))
    #
    # for i, q_values in enumerate(q_values_at_intervals):
    #     ax[i].imshow(q_values, cmap='hot', interpolation='nearest')
    #     ax[i].set_title(f'Q-Values at Episode {(episodes // 4) * (i + 1)}')
    #     ax[i].title.set_text(f'is_slippery: {random_termination}, frequent_rewards: {frequent_rewards}, tau: {tau}')
    #     ax[i].axis('off')  # Turn off axis
    #     plt.colorbar(ax[i].imshow(q_values, cmap='hot', interpolation='nearest'), ax=ax[i])
    #     plt.pause(0.1)  # Pause briefly to update the plot
    #     break

if __name__ == "__main__":
    frozen_lake(frequent_rewards=True, random_termination=True, tau=1)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=10)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=0.1)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=0.01)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=1e-3)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=1e-4)
    frozen_lake(frequent_rewards=True, random_termination=True, tau=1e-5)