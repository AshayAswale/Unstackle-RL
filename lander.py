import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3")
# Define the DQN model
model = DQN("MlpPolicy", env, verbose=1)
# Train the model
model.learn(total_timesteps=100000)  # Adjust the timesteps as needed
# Evaluate the trained mode
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Optionally, save the model
model.save("dqn_lunar_lander")
import time
env = gym.make("LunarLander-v3", render_mode="human")
# Load the model if needed
# model = DQN.load("dqn_lunar_lander")
# Visualize the model's performance
episodes = 5

for episode in range(1, episodes + 1):    
  obs, _states = env.reset()    
  done = False    

  score = 0    

  while not done:        

    env.render()        

    action, _states = model.predict(obs)        
    print(action, _states)

    observation, reward, terminated, truncated, info = env.step(action)        

    done = terminated or truncated
    score += reward    

  print(f"Episode: {episode}, Score: {score}")    

  time.sleep(1)

env.close()