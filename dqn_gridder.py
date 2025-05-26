import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import Stackle
from gymnasium.wrappers import RecordVideo
training_period = 100  # record the agent's episode every 250
# Create the Lunar Lander environment
env = gym.make("Stackle/GridWorld-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="gridder", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
# Define the DQN model
model = DQN("MultiInputPolicy", env, verbose=1, device="cuda")
# Train the model
model.learn(total_timesteps=100000)  # Adjust the timesteps as needed
# Evaluate the trained mode
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Optionally, save the model
model.save("dqn_3x3_gridder")
