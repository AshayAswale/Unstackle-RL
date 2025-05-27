import numpy as np
import matplotlib.pyplot as plt

data = np.load("./logs/evaluations.npz")
print(data)
timesteps = data["timesteps"]
results = data["results"]  # shape: (num_evals, n_eval_episodes)

# Plot mean and std across eval episodes
mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

plt.plot(timesteps, mean_rewards, label="Mean Reward")
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Evaluation Reward Over Time")
plt.legend()
plt.grid(True)

lengths = data["ep_lengths"]  # (num_evals, n_eval_episodes)

mean_lengths = lengths.mean(axis=1)
plt.plot(timesteps, mean_lengths, label="Mean Episode Length", color="orange")
plt.xlabel("Timesteps")
plt.ylabel("Episode Length")
plt.title("Episode Length Over Time")
plt.grid(True)
plt.legend()

plt.show()
