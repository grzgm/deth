import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def frozen_lake(is_slippery=False, frequent_rewards=True, epsilon=0.05, alpha=0.08, gamma=0.95):
    # hyper parameters
    # epsilon = 0.05  # Exploration rate between (0-1)
    # alpha = 0.08  # Learning rate
    # gamma = 0.95  # Discount factor between (0-1)
    episodes = 4 * 2000  # Number of episodes to learn, keep this a multiple of four for nice plotting
    T = 100  # Maximum steps in an episode
    cost_of_living = -0.01  # Used when frequent_rewards = True, incentivize the agent for efficiency
    # is_slippery = True

    # Choose environment
    env = gym.make("FrozenLake8x8-v1", is_slippery=is_slippery)
    # frequent_rewards = True  # When False the original environment rewards are used

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []
    q_values_at_intervals = []  # Store Q-values at intervals

    for episode in range(episodes):
        state, prob = env.reset()
        total_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []

        for step in range(T):
            # Choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            if frequent_rewards:
                if terminated and reward == 0:  # agent fall in the hole!
                    reward = reward - 1
                reward = reward + cost_of_living

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            total_reward += reward
            state = new_state

            if terminated:
                if reward == 1 + cost_of_living:
                    print(
                        f"Episode {episode} finished after {step + 1} steps. Epsilon is {epsilon}, LR {alpha} Success!")
                break

        rewards_per_episode.append(total_reward)

        # Calculate returns and update Q-values
        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]

            G = gamma * G + reward
            Q[state][action] += alpha * (G - Q[state][action])

        # Store Q-values at intervals (e.g., every 100 episodes)
        if (episode + 1) % (episodes // 4) == 0:
            q_values_at_intervals.append(np.copy(Q))  # Store a copy of Q-values

    # Plotting rewards per episode
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode, label='Total Reward')
    plt.title('Rewards per Episode')
    plt.suptitle(f'is_slippery={is_slippery}, frequent_rewards={frequent_rewards}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Calculate moving average for a cleaner plot
    moving_average_window = 20
    moving_averages = savgol_filter(rewards_per_episode, moving_average_window, 3)
    plt.plot(moving_averages, label=f'Moving Average (Window {moving_average_window})', color='orange')

    plt.legend()

    # Plotting the heatmap of Q-values at intervals
    fig, ax = plt.subplots(1, len(q_values_at_intervals), figsize=(10, 5))

    for i, q_values in enumerate(q_values_at_intervals):
        ax[i].imshow(q_values, cmap='hot', interpolation='nearest')
        ax[i].set_title(f'Q-Values at Episode {(episodes // 4) * (i + 1)}')
        ax[i].title.set_text(f'is_slippery={is_slippery}, frequent_rewards={frequent_rewards}')
        ax[i].axis('off')  # Turn off axis
        plt.colorbar(ax[i].imshow(q_values, cmap='hot', interpolation='nearest'), ax=ax[i])
        plt.pause(0.1)  # Pause briefly to update the plot
        break

    # Display plots and pause to show the graphs
    plt.show()


if __name__ == '__main__':
    frozen_lake(is_slippery=False, frequent_rewards=True, epsilon=0.05, alpha=0.08, gamma=0.95)
