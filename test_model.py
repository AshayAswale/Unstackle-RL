import gymnasium as gym
import Stackle
import time
from stable_baselines3 import DQN

env = gym.make("Stackle/GridWorld-v0", render_mode="human")
# Load the model if needed
model = DQN.load("dqn_3x3_gridder")
# Visualize the model's performance
episodes = 5

for episode in range(1, episodes + 1):    
  obs, _ = env.reset()    
  done = False    

  score = 0    

  while not done:        

    env.render()        

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action.tolist())    

    done = terminated or truncated
    score += reward    

  print(f"Episode: {episode}, Score: {score}")    

  time.sleep(1)

env.close()