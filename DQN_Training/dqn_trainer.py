import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from plantos_env import PlantOSEnv # Assuming PlantOSEnv is compatible with Gym API

# Define paths for saving models and logs
MODEL_DIR = "DQN_Training/models"
LOG_DIR = "DQN_Training/logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Create the environment
    # PlantOSEnv needs to be registered or wrapped to be fully compatible with SB3's make_vec_env
    # For a single environment, we can directly instantiate and wrap with Monitor
    
    # It's good practice to wrap the environment with Monitor for logging
    # and to ensure it's compatible with SB3's expectations.
    # We'll use the 'grid' observation mode as per the previous discussion.
    env = PlantOSEnv(grid_size=21, num_plants=8, num_obstacles=12, lidar_range=6, lidar_channels=32, observation_mode='grid')
    env = Monitor(env, LOG_DIR)

    # 2. Define the model
    # SB3's DQN supports both MlpPolicy (for flat observations) and CnnPolicy (for image-like observations)
    # Since our 'grid' observation is 4x21x21, CnnPolicy is appropriate.
    # The input shape for CnnPolicy expects (channels, height, width) which matches our env.observation_space.shape
    
    # Hyperparameters for DQN (can be tuned)
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    # 3. Define Callbacks for saving and evaluation
    # Save the best model based on evaluation reward
    eval_callback = EvalCallback(
        env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=1000, # Evaluate every 1000 steps
        deterministic=True,
        render=False
    )

    # Stop training if a certain reward threshold is reached
    # This is optional, but can prevent overtraining or save time if a good solution is found
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=env.R_GOAL * env.num_plants * 0.8, verbose=1)
    # combined_callbacks = [eval_callback, stop_callback]
    combined_callbacks = [eval_callback] # For now, just eval callback

    # Save models periodically
    training_run = "100k"
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, #20k Save every 5000 timesteps
        save_path=MODEL_DIR,
        name_prefix=f"dqn_model-{training_run}"
    )
    combined_callbacks = [eval_callback, checkpoint_callback]

    # 4. Train the model
    print("Starting DQN training with Stable Baselines3...")
    model.learn(
        total_timesteps=100000, # 100k Total number of timesteps to train
        callback=combined_callbacks
    )
    print("DQN Training Finished.")

    # 5. Save the final model
    model.save(os.path.join(MODEL_DIR, "dqn_plantos_final_model"))

    # 6. Close the environment
    env.close()