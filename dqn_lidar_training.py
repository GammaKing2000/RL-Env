import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn, Tensor

from plantos_env import PlantOSEnv

# Create the environment in LIDAR mode
env = PlantOSEnv(
    grid_size=21, 
    num_plants=10, 
    num_obstacles=12, 
    lidar_range=4, 
    lidar_channels=12
)

class CustomANN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        n_input_features = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(n_input_features, 128),
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
    batch_size=128, # batch size for training
    buffer_size=50000, # size of the replay buffer
    exploration_final_eps=0.1, # final value of epsilon for epsilon-greedy exploration
    exploration_fraction=0.12, # fraction of total timesteps for epsilon-greedy exploration
    gamma=0.99, # discount factor
    gradient_steps=1, # number of gradient steps to perform after each rollout
    learning_rate=0.0006, # learning rate
    learning_starts=1000, # number of steps before learning starts
    target_update_interval=250, # frequency of target network updates
    train_freq=4, # frequency of training
    policy_kwargs=policy_kwargs # passing the custom policy kwargs
)

# Train the model
print("Starting training with LIDAR observations...")
model.learn(total_timesteps=2000000, log_interval=4)

# Save the trained model
model.save("dqn_plantos_lidar_sb3")

print("Training finished and model saved.")

# --- Evaluation ---
print("\nStarting evaluation...")
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_plantos_lidar_sb3")

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

env.close()
