import os
import sys
import gymnasium as gym

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import logger
from training_utils import SaveOnIntervalCallback, visualise_training_logs
from plantos_env import PlantOSEnv

if __name__ == "__main__":
    training_run = "1M"

    # Define paths for saving models and logs
    MODEL_DIR = os.path.join("DQN_Training/models", training_run)
    LOG_DIR = os.path.join("DQN_Training/logs", training_run)
    TENSORBOARD_LOG_DIR = "DQN_Training/logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create the vectorized environment
    n_envs = 4
    env_kwargs = {
        'grid_size': 21,
        'num_plants': 20,
        'num_obstacles': 12,
        'lidar_range': 6,
        'lidar_channels': 32,
        'observation_mode': 'grid',
        'thirsty_plant_prob': 0.5
    }
    env = make_vec_env('PlantOS-v0', n_envs=n_envs, env_kwargs=env_kwargs)

    # 2. Define the model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-5,  # Slightly smaller learning rate for more stable updates
        buffer_size=100000,  # Increased buffer size from 100000
        learning_starts=10000, # Increase learning starts to fill the buffer a bit more
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,  # Explore for a bit longer
        exploration_final_eps=0.01, # Exploit more at the end
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256])
        #tensorboard_log=TENSORBOARD_LOG_DIR
    )
    
    # Log the training
    new_logger = logger.configure(LOG_DIR, ["stdout", "csv"])
    model.set_logger(new_logger)

    # 3. Define Callbacks for saving and evaluation
    # Get env constants from the vectorized environment
    num_plants = env.get_attr('num_plants')[0]
    r_goal = env.get_attr('R_GOAL')[0]
    reward_threshold = r_goal * num_plants * 0.8
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    
    eval_callback = EvalCallback(
        env,
        callback_after_eval=stop_callback,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=100000,
        deterministic=True,
        render=False
    )

    save_interval = 100000
    save_callback = SaveOnIntervalCallback(save_interval, MODEL_DIR)
    combined_callbacks = [eval_callback, save_callback]

    # 4. Train the model
    print("Starting DQN training with Stable Baselines3...")
    model.learn(
        total_timesteps=200000, # 200,000
        callback=combined_callbacks
    )
    print("DQN Training Finished.")
    print(f"Total timesteps trained: {model.num_timesteps}")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"This is the avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 5. Save the final model
    model.save(os.path.join(MODEL_DIR, f"dqn_plantos_final_model-{training_run}"))

    # 6. Visualize the training curve and save it
    visualise_training_logs("rollout/ep_rew_mean", "Rewards", LOG_DIR)
    visualise_training_logs("rollout/ep_len_mean", "Episode Length", LOG_DIR)

    # 7. Close the environment
    env.close()