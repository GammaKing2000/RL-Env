import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from plantos_env import PlantOSEnv

# Create the environment in LIDAR mode
env = PlantOSEnv(
    grid_size=21, 
    num_plants=8, 
    num_obstacles=12, 
    lidar_range=6, 
    lidar_channels=32,
    observation_mode='lidar'  # <-- Key change here
)

# It's a good practice to check if your custom environment is compatible with stable-baselines3
check_env(env)

# Instantiate the DQN model. We use "MlpPolicy" because the observation space is a 1D vector.
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000)

# Train the model
print("Starting training with LIDAR observations...")
model.learn(total_timesteps=20000, log_interval=4)

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
