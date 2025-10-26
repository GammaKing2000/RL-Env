import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
from torch import nn as nn, Tensor
import matplotlib.pyplot as plt
import numpy as np

from plantos_env import PlantOSEnv


log_dir = "train_4Lakh/gym/"
models_dir = "train_4Lakh/models/"

# Ensuring the directories exist or creating them
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Create the environment in LIDAR mode
env = PlantOSEnv(
    grid_size=25, 
    num_plants=10, 
    num_obstacles=12, 
    lidar_range=4, 
    lidar_channels=12
)

env = Monitor(env, log_dir)

class CustomANN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        n_input_features = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.net(observations)


# It's a good practice to check if your custom environment is compatible with stable-baselines3
check_env(env)

# Instantiate the DQN model. We use "MlpPolicy" because the observation space is a 1D vector.
policy_kwargs = dict(
    features_extractor_class=CustomANN, # initialising the CustomANN
    features_extractor_kwargs=dict(features_dim=256),
)

model = DQN(
    "MlpPolicy", # Using the MlpPolicy
    env, # Passing the environment
    verbose=1,
    batch_size=256, # batch size for training
    buffer_size=100000, # size of the replay buffer
    exploration_final_eps=0.05, # final value of epsilon for epsilon-greedy exploration
    exploration_fraction=0.25, # fraction of total timesteps for epsilon-greedy exploration
    gamma=0.99, # discount factor
    gradient_steps=1, # number of gradient steps to perform after each rollout
    learning_rate=0.001, # learning rate
    learning_starts=10000, # number of steps before learning starts
    target_update_interval=1000, # frequency of target network updates
    train_freq=4, # frequency of training
    policy_kwargs=policy_kwargs # passing the custom policy kwargs
)

class SaveOnIntervalCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval  # Setting the interval of saving
        self.save_path = save_path  # Setting the path to save models

    def _on_step(self) -> bool:
        # Saving the model if the current timestep is a multiple of the save interval
        if self.num_timesteps % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f'model_{self.num_timesteps}')  # Defining the file name for the model
            self.model.save(save_file)  # Saving the model
            if self.verbose > 0:
                print(f'Saving model to {save_file}.zip')  # Printing a message on successful save
        return True

total_timesteps = 400000
save_interval = total_timesteps // 3
callback = SaveOnIntervalCallback(save_interval, save_path=models_dir)

# Train the model
print("Starting training with LIDAR observations...")
model.learn(total_timesteps=total_timesteps, log_interval=4)

# Save the trained model
model.save(os.path.join(models_dir, "dqn_plantos_lidar_sb3"))

print("Training finished and model saved.")

# --- Evaluation ---
print("\nStarting evaluation...")
del model # remove to demonstrate saving and loading

model = DQN.load(os.path.join(models_dir, "dqn_plantos_lidar_sb3"))

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

# Function for plotting the learning curve of the agent
def plot_results(log_folder, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")  # Loading the results
    y = np.convolve(y, np.ones((50,))/50, mode='valid')  # Smoothing the curve using a moving average of 50 episodes
    x = x[len(x) - len(y):]  # Adjusting the x-axis values
    plt.figure(figsize=(10,5))
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig('learning_curve.png')
    plt.show()

plot_results(log_dir)


env.close()
