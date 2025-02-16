# Alex Hunt (113536050)
# (CS-5033-001) Machine Learning Fundamentals (Spring 2024) with Professor Diochnos
# RL Project - Temporal Difference - Testing the effictiveness of heuristics

# Import necessary libraries
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import numpy as np
import matplotlib.pyplot as plt

# Define the environment map
custom_map = [
    "SFFFFFHFF",  # S: Start, F: Frozen surface, H: Hole, G: Goal
    "FFFFFFFHF",
    "FFFFFFFFF",
    "FHFHFFFFF",
    "FFFFFFFFF",
    "FFFFFFFFF",
    "FFFFFFFFF",
    "FFFFFHFFF",
    "FFFFFFFFG"
]

# Function to get the opposite of a given action
def get_opposite_action(action):
    opposite_actions = {0: 2, 2: 0, 1: 3, 3: 1}  # Mapping of opposite actions
    return opposite_actions[action]

# Function to run the Q-learning algorithm with or without heuristic adjustments
def run_q_learning(with_heuristic):
    # Initialize the environment with the custom map and set it to non-slippery
    environment = FrozenLakeEnv(desc=custom_map, map_name=None, is_slippery=False)

    # Initialize a matrix to store action values, filled with zeros
    action_value_matrix = np.zeros([environment.observation_space.n, environment.action_space.n])

    # Define hyperparameters for the learning process
    learning_rate = 0.1
    discount_factor = 0.9
    episodes = 2000  # Total number of episodes to run

    # Initialize lists to track rewards
    episode_rewards = []
    cumulative_rewards = [0]  # Start cumulative rewards at 0

    # Loop through each episode
    for episode in range(episodes):
        current_state = environment.reset()  # Reset environment to start state
        episode_done = False
        total_reward = 0
        current_state = current_state[0]

        # Main loop for Q-learning algorithm
        for step in range(99):
            if with_heuristic:
                penalty = -10  # Define a penalty for heuristic adjustments

                # Apply penalties based on heuristic rules
                action_value_matrix[current_state, 3] += penalty
                action_value_matrix[current_state, 0] += penalty

                # Edge case adjustments
                if current_state % 9 == 0:
                    action_value_matrix[current_state, 0] += penalty
                if current_state < 9:
                    action_value_matrix[current_state, 3] += penalty
                if current_state > 71:
                    action_value_matrix[current_state, 1] += penalty
                if step != 0:
                    action_value_matrix[current_state, get_opposite_action(last_action)] += penalty
            
            # Exploration factor decreases over time
            exploration_factor = np.random.randn(1, environment.action_space.n) * (1. / (episode + 1))

            # Select action with the highest value, factoring in exploration
            action = np.argmax(action_value_matrix[current_state, :] + exploration_factor)
            last_action = action

            # Execute selected action and observe the outcome
            next_state, reward, episode_done, _, info = environment.step(action)

            # Update the action-value matrix based on the outcome
            action_value_matrix[current_state, action] = action_value_matrix[current_state, action] + \
                                                        learning_rate * (reward + discount_factor * np.max(action_value_matrix[next_state, :]) - action_value_matrix[current_state, action])
            
            total_reward += reward  # Update total reward for the episode
            current_state = next_state  # Move to the next state
            if episode_done:  # End the episode if done
                break

        # Track rewards over episodes
        episode_rewards.append(total_reward)
        cumulative_rewards.append(cumulative_rewards[-1] + total_reward)

    # Calculate the running average of rewards
    running_average = []
    running_sum = 0.0
    for reward in episode_rewards:
        running_sum = reward / 100 + running_sum * 99 / 100
        running_average.append(running_sum)

    # Output summary statistics
    print("OpenAI Gym FrozenLake:")
    print("Heuristics Active = " + str(with_heuristic))
    print(f"Average score over all episodes: {sum(episode_rewards)/episodes}")
    print("Final Action-Value Matrix")
    print(action_value_matrix)
    print("")

    return running_average, cumulative_rewards[1:], action_value_matrix

# Run Q-learning with and without heuristic adjustments
running_average_with, cumulative_rewards_with, avm_w = run_q_learning(True)
running_average_without, cumulative_rewards_without, avm_wo = run_q_learning(False)

# Plot comparison of running average rewards
plt.plot(running_average_with, label="Running Average Reward with Heuristic")
plt.plot(running_average_without, label="Running Average Reward without Heuristic")
plt.xlabel('Episodes')  # Label for the x-axis
plt.ylabel('Average Reward (Win per Try)')  # Label for the y-axis
plt.title('Heuristic effectiveness with Q-Learning')  # Title of the plot
plt.legend(loc='best')  # Show legend to identify the lines
plt.show()  # Display the plot

# Derive the optimal policy from the final action-value matrix
optimal_policy_w = np.argmax(avm_w, axis=1)
optimal_policy_wo = np.argmax(avm_wo, axis=1)

# Function to simulate and print the optimal path based on a given policy
def simulate_optimal_path(optimal_policy):
    print("Optimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
    print(optimal_policy.reshape((int(np.sqrt(environment.observation_space.n)), -1)))  # Reshape and print the policy

    location = environment.reset()  # Reset environment to start state
    location = location[0]
    goal = False
    path = []  # Initialize path taken
    while not goal:  # Loop until the goal state is reached
        print(location)  # Print current location
        path.append(optimal_policy[location])  # Append current action to path
        location, reward, goal, _, info = environment.step(optimal_policy[location])  # Take the action
    print(path)  # Print the path taken to reach the goal

# Initialize environment with render mode for visualization
environment = FrozenLakeEnv(desc=custom_map, map_name=None, is_slippery=False, render_mode='human')
print("Optimal Path with Heuristics")
simulate_optimal_path(optimal_policy_w)  # Simulate with heuristic
print("Optimal Path without Heuristics")
simulate_optimal_path(optimal_policy_wo)  # Simulate without heuristic

# Plotting the cumulative rewards
plt.figure(figsize=(14, 6))  # Set figure size
plt.subplot(1, 2, 1)  # Prepare subplot 1
plt.plot(cumulative_rewards_with, label="Cumulative Reward with Heuristic")  # Plot cumulative rewards with heuristic
plt.plot(cumulative_rewards_without, label="Cumulative Reward without Heuristic")  # Plot cumulative rewards without heuristic
plt.xlabel('Episodes')  # Label for the x-axis
plt.ylabel('Cumulative Reward')  # Label for the y-axis
plt.title('Cumulative Reward over All Episodes')  # Title of the plot
plt.legend()  # Show legend to identify the lines

plt.show()  # Display the plot