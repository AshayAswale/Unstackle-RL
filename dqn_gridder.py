import gymnasium as gym
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import Stackle
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

training_period = 100  # record the agent's episode every 250
# Create the Lunar Lander environment
env = gym.make("Stackle/GridWorld-v0", render_mode="rgb_array")
eval_env = gym.make("Stackle/GridWorld-v0", render_mode="rgb_array")  # Separate eval env
# env = RecordVideo(env, video_folder="gridder", name_prefix="training",
#                   episode_trigger=lambda x: x % training_period == 0)
# Define the DQN model
# model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")

model = DQN.load("best_model\\complete_but_long")
model.set_env(env)
# model = DQN(
#     "MultiInputPolicy",
#     env,
#     learning_rate=1e-4,
#     buffer_size=100_000,
#     batch_size=128,
#     exploration_fraction=0.5,
#     exploration_final_eps=0.01,
#     train_freq=4,
#     gamma=0.99,
#     verbose=1,
#     device="cuda"
# )

# Callback: evaluates every 10k steps, saves best to ./best_model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=10_000,
    deterministic=False,
    n_eval_episodes=5,
    render=False,
    verbose=1
)

# Train the model
model.learn(total_timesteps=150_000, callback=eval_callback)

# Evaluate the trained mode
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

