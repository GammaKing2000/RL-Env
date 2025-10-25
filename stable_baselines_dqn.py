import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from plantos_env import PlantOSEnv

# Create the environment using the new defaults (LIDAR-only, 10 channels)
env = PlantOSEnv()

# Check the environment
check_env(env)

# Define a custom network architecture
policy_kwargs = dict(net_arch=[128, 128])

# Instantiate the DQN model with MlpPolicy and the custom network
model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=10000, learning_starts=1000)

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