# Implementation of On-Policy Monte Carlo Control using Gymnasium

---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a starting point, and a goal. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function \(Q(s,a)\) using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

- Python 3.x
- Gymnasium
- NumPy
- Matplotlib

### Installation

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

**FrozenLake-v1** is a grid-based environment provided by Gymnasium.

It contains a 4 × 4 grid with 16 states.

| Symbol | Description |
|---|---|
| S | Starting position |
| F | Frozen surface |
| H | Hole |
| G | Goal |

### Environment Grid

```text
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

The agent can perform four actions:

| Action | Direction |
|---|---|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

The agent receives a reward of 1 when it reaches the goal and 0 for other transitions.

In this experiment, `is_slippery=False` is used, which means the agent moves in the selected direction without random slipping.

---

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

\[
S_0,A_0,R_1,S_1,A_1,R_2,\ldots,S_T
\]

The return from time step \(t\) is:

\[
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
\]

Monte Carlo Control estimates the action-value function:

\[
Q(s,a)
\]

The incremental update rule is:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha[G_t-Q(s,a)]
\]

### Parameters

| Symbol | Meaning |
|---|---|
| \(s\) | Current state |
| \(a\) | Action taken in state \(s\) |
| \(G_t\) | Return from time step \(t\) |
| \(Q(s,a)\) | Action-value estimate |
| \(\alpha\) | Learning rate |
| \(\gamma\) | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses an epsilon-greedy policy for selecting actions.

- **Exploration:** With probability \(\epsilon\), the agent selects a random action.
- **Exploitation:** With probability \(1-\epsilon\), the agent selects the action with the highest Q-value.

The greedy action is selected as:

\[
a=\arg\max_a Q(s,a)
\]

The final learned policy is:

\[
\pi(s)=\arg\max_a Q(s,a)
\]

### Epsilon Decay

The epsilon value gradually decreases during training.

\[
\epsilon=\max(\epsilon_{\min},\epsilon\times\epsilon_{decay})
\]

This allows the agent to explore more during the initial episodes and exploit learned knowledge during later episodes.

---

## Algorithm

1. Start the program.
2. Import Gymnasium, NumPy, and Matplotlib libraries.
3. Create the FrozenLake environment.
4. Initialize the Q-table with zeros.
5. Set the hyperparameters such as learning rate, discount factor, and epsilon.
6. Generate an episode using the epsilon-greedy policy.
7. Store the states, actions, and rewards of the episode.
8. Calculate returns by traversing the episode backward.
9. Update the Q-table using the First-Visit Monte Carlo method.
10. Reduce epsilon gradually after each episode.
11. Repeat the process for the specified number of episodes.
12. Extract the greedy policy from the final Q-table.
13. Calculate and display the state-value function.
14. Display the final Q-table and learned policy.
15. Plot the learning curve.
16. Stop the program.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

# Hyperparameters
num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# Initialize Q-table
Q = np.zeros((n_states, n_actions))
episode_rewards = []

# Epsilon-Greedy Action Selection
def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

# Generate One Complete Episode
def generate_episode(epsilon):

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode

# Monte Carlo Control
epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0
    returns = []

    # Calculate Returns
    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        returns.append((state, action, G))

    returns.reverse()

    # Update Q-table using First-Visit MC
    visited = set()

    for state, action, G in returns:

        if (state, action) not in visited:

            visited.add((state, action))

            Q[state, action] += alpha * (
                G - Q[state, action]
            )

    total_reward = sum(
        reward for _, _, reward in episode
    )

    episode_rewards.append(total_reward)

    # Decay Epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if (episode_num + 1) % 2000 == 0:

        print(
            f"Episode: {episode_num + 1}, "
            f"Reward: {total_reward}, "
            f"Epsilon: {epsilon:.3f}"
        )

# Extract Greedy Policy
optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# Display Results
print("Name: Rakisha R")
print("Register Number: 212225230223")

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)

# Plot Learning Curve
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)
plt.show()

env.close()
```

---

## Hyperparameters

| Parameter | Value | Description |
|---|---:|---|
| Number of episodes | 20000 | Total training episodes |
| Gamma (\(\gamma\)) | 0.99 | Discount factor |
| Alpha (\(\alpha\)) | 0.1 | Learning rate |
| Initial epsilon | 1.0 | Initial exploration rate |
| Minimum epsilon | 0.05 | Minimum exploration rate |
| Epsilon decay | 0.9995 | Exploration decay factor |
| Maximum steps | 100 | Maximum steps per episode |
| Moving average window | 500 | Episodes used for smoothing the learning curve |

---

## Output

<img width="522" height="55" alt="image" src="https://github.com/user-attachments/assets/04b82869-7361-4e2d-9c9d-5fb194d6530b" />

<img width="427" height="730" alt="image" src="https://github.com/user-attachments/assets/4feea246-acf9-479f-afee-50eb814ead18" />

<img width="957" height="595" alt="image" src="https://github.com/user-attachments/assets/6786a688-2df2-4c16-96df-20454b804caa" />


## Result

Thus, **On-Policy Monte Carlo Control** was successfully implemented using the Gymnasium `FrozenLake-v1` environment.

The agent generated complete episodes, calculated Monte Carlo returns, updated the Q-table using the First-Visit method, and learned an improved policy using epsilon-greedy action selection.

The final Q-table, estimated state-value function, learned policy, and learning curve were obtained.

---

## Inference

1. Monte Carlo Control learns from complete episodes.
2. The Q-table estimates the expected return for each state-action pair.
3. Epsilon-greedy selection helps the agent balance exploration and exploitation.
4. Epsilon decay gradually reduces random action selection during training.
5. The learned policy helps the agent select better actions to reach the goal.
6. The learning curve helps us understand the improvement in the agent's performance.

**Conclusion:** Monte Carlo Control enables an agent to learn an improved policy through experience without requiring prior knowledge of the environment's transition probabilities.

---

## References

1. [Gymnasium Documentation](https://gymnasium.farama.org/)
2. [FrozenLake Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
3. Sutton and Barto, *Reinforcement Learning: An Introduction* (Second Edition).
4. [NumPy Documentation](https://numpy.org/doc/)
5. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---
