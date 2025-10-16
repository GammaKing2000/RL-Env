import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from plantos_env import PlantOSEnv

# Custom CNN for the PlantOS environment
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": dict(features_dim=64),
}

# Create the environment
env = PlantOSEnv(grid_size=21, num_plants=8, num_obstacles=12, lidar_range=6, lidar_channels=32)

# Check the environment
check_env(env)

# Instantiate the DQN model with the custom policy
model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=10000, learning_starts=1000)

# Train the model
print("Starting training...")
model.learn(total_timesteps=20000, log_interval=4)

# Save the trained model
model.save("dqn_plantos_sb3_custom")

print("Training finished and model saved.")

# --- Evaluation ---
print("\nStarting evaluation...")
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_plantos_sb3_custom")

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

env.close()