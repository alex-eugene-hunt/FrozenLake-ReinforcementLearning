# Alex Hunt (113536050)
# (CS-5033-001) Machine Learning Fundamentals (Spring 2024) with Professor Diochnos
# RL Project - Eligibility Traces - Testing the effictiveness of various variable values

# Import necessary libraries for creating and manipulating the environment and for visualization.
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import matplotlib.pyplot as plt

# Initialize the FrozenLake environment without slipperiness for deterministic movements.
env = gym.make("FrozenLake-v1", is_slippery=False)
state_dim = env.observation_space.n  # Total number of states in the environment.
action_dim = env.action_space.n  # Total number of possible actions.
goal = 15  # Define the goal state for the FrozenLake environment.

# Define the hyperparameter space for the SARSA(λ) learning algorithm.
gamma_values = [0.8, 0.9, 0.95]  # Discount factor for future rewards.
lamda_values = [0.4, 0.6, 0.8]  # Lambda values for eligibility traces.
alpha_values = [0.1, 0.05, 0.01]  # Learning rate.
eps_values = [0.1, 0.2, 0.05]  # Epsilon values for the epsilon-greedy policy.
max_episode = 2000  # Maximum number of episodes for training.

# Function to select an action based on the current state and Q-values using epsilon-greedy policy.
def choose_action(s, q, eps):
    if np.random.random() < eps:  # With probability eps, choose a random action.
        return np.random.randint(low=0, high=action_dim)
    else:  # Otherwise, choose the action with the highest Q-value for the current state.
        return np.argmax(q[s, :])

# Implementation of the SARSA(λ) algorithm.
def run_sarsa_lamda(gamma, lamda, alpha, eps):
    q = np.zeros([state_dim, action_dim])  # Initialize Q-values for all state-action pairs to zero.
    episode_rewards = []  # To keep track of the total reward per episode.
    cumulative_rewards = []  # To track cumulative rewards over all episodes.

    cumulative_reward = 0  # Initialize cumulative reward.
    for episode in range(max_episode):  # Loop over episodes.
        eligibility = np.zeros([state_dim, action_dim])  # Initialize eligibility traces.
        observation = env.reset()[0]  # Start a new episode.
        action = choose_action(observation, q, eps)  # Choose initial action.
        done = False
        total_reward = 0
        
        while not done:  # Loop within an episode.
            next_observation, reward, done, _, _ = env.step(action)  # Take the action.
            if done and next_observation != goal: reward = -1  # Penalize if not reached goal.
            if observation == next_observation: reward = -1  # Penalize if no progress made.
            next_action = choose_action(next_observation, q, eps)  # Choose next action.
            
            # Calculate the temporal difference error.
            delta = reward + gamma * q[next_observation, next_action] - q[observation, action]
            eligibility[observation, action] += 1  # Increase eligibility for visited state-action pair.
            
            # Update Q-values and eligibility traces for all state-action pairs.
            for i in range(state_dim):
                for j in range(action_dim):
                    q[i, j] += alpha * delta * eligibility[i, j]
                    eligibility[i, j] *= gamma * lamda
            
            observation, action = next_observation, next_action  # Update state and action.
            total_reward += reward  # Accumulate reward.
        
        episode_rewards.append(total_reward)  # Record total reward for this episode.
        cumulative_reward += total_reward  # Update cumulative reward.
        cumulative_rewards.append(cumulative_reward)  # Append to list of cumulative rewards.

    average_reward = sum(episode_rewards) / max_episode  # Calculate average reward.
    return average_reward, cumulative_rewards, episode_rewards, q

# Function to search for the best set of hyperparameters.
def hyperparameter_search():
    best_avg_reward = -np.inf  # Initialize best average reward.
    worst_avg_reward = np.inf  # Initialize worst average reward.
    best_params = {}  # To store the best hyperparameters.
    worst_params = {}  # To store the worst hyperparameters.
    best_performance = None  # To store the best Q-table.
    worst_performance = None  # To store the worst Q-table.

    # Grid search over the hyperparameter space.
    for gamma in gamma_values:
        for lamda in lamda_values:
            for alpha in alpha_values:
                for eps in eps_values:
                    avg_reward, cr, er, q = run_sarsa_lamda(gamma, lamda, alpha, eps)  # Execute SARSA(λ) with the current set of hyperparameters.
                    print(f"gamma: {gamma}, lamda: {lamda}, alpha: {alpha}, eps: {eps}, Avg Reward: {avg_reward}")  # Log performance.

                    # Update best and worst performance based on average reward.
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_params = {'gamma': gamma, 'lamda': lamda, 'alpha': alpha, 'eps': eps}
                        best_performance = q  # Save the best Q-table.

                    if avg_reward < worst_avg_reward:
                        worst_avg_reward = avg_reward
                        worst_params = {'gamma': gamma, 'lamda': lamda, 'alpha': alpha, 'eps': eps}
                        worst_performance = q  # Save the worst Q-table.

    # Print out the best and worst sets of hyperparameters.
    print("Best Hyperparameters:", best_params, "with Avg Reward:", best_avg_reward)
    print("Worst Hyperparameters:", worst_params, "with Avg Reward:", worst_avg_reward)

    return best_params, best_performance, worst_params, worst_performance

# Run the hyperparameter search to find the best and worst configurations.
best_params, _, worst_params, _ = hyperparameter_search()

# Analyze and compare the performance of the best vs. worst hyperparameter settings.
# Run SARSA(λ) for both the best and worst hyperparameters to get detailed rewards data.
a_best, best_cumulative_reward, best_episode_reward, q_best = run_sarsa_lamda(**best_params)
a_worst, worst_cumulative_reward, worst_episode_reward, q_worst = run_sarsa_lamda(**worst_params)

# Compute the running average of rewards for visualization.
best_running_avg = np.cumsum(best_episode_reward) / (np.arange(len(best_episode_reward)) + 1)
worst_running_avg = np.cumsum(worst_episode_reward) / (np.arange(len(worst_episode_reward)) + 1)

# Plotting the running average rewards to compare best and worst hyperparameters.
plt.figure(figsize=(12, 6))
plt.plot(best_running_avg, label="Best Hyperparameters")
plt.plot(worst_running_avg, label="Worst Hyperparameters")
plt.xlabel("Episode")  # X-axis label.
plt.ylabel("Average Reward")  # Y-axis label.
plt.title("Comparison of Best vs. Worst Hyperparameter Performance")  # Chart title.
plt.legend()  # Display legend.
plt.show()  # Display the plot.

# Function to simulate and display the optimal path based on the derived optimal policy.
def simulate_optimal_path(optimal_policy):
    print("Optimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")  # Display the mapping of actions.
    print(optimal_policy.reshape((int(np.sqrt(env.observation_space.n)), -1)))  # Display the optimal policy in grid format.

    location = env.reset()  # Reset the environment for simulation.
    location = location[0]
    goal = False
    path = []  # Initialize the path taken.
    while not goal:  # Loop until the goal is reached.
        path.append(optimal_policy[location])  # Append the chosen action to the path.
        location, reward, goal, _, info = env.step(optimal_policy[location])  # Take the action.
        print(location)  # Optional: Print each step's location.
    print("Path:", path)  # Display the complete path taken.

# Derive and display the optimal policies for both the best and worst hyperparameter settings.
optimal_policy_b = np.argmax(q_best, axis=1)
optimal_policy_w = np.argmax(q_worst, axis=1)
print("Optimal Path with the best hyperparameters")
simulate_optimal_path(optimal_policy_b)  # Simulate with the best hyperparameters.
print("Optimal Path with the worst hyperparameters")
simulate_optimal_path(optimal_policy_w)  # Simulate with the worst hyperparameters.
env.close()

custom_map = [
    "SFFFFFFFF",
    "FFFFFFFFF",
    "FFFFFFFFF",
    "FHFFFFFFF",
    "FFHFHFFFH",
    "FFFFFFHHF",
    "HHFHFHFFF",
    "FFFFFFFHF",
    "FFFFFFHFG",
]

# Initialize the environment
env = FrozenLakeEnv(desc=custom_map, map_name=None, is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n
goal = 80
max_episode = 250

a_best, best_cumulative_reward, best_episode_reward, q_best = run_sarsa_lamda(**best_params)

# Plot the cumulative rewards for both best and worst parameters
plt.figure(figsize=(12, 6))
plt.plot(best_cumulative_reward, label="Best Hyperparameters")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison of Best Hyperparameters")
plt.legend()
plt.show()