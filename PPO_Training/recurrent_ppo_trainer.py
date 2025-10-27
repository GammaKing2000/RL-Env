import os
import sys
import gymnasium as gym

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common import logger
from collections import OrderedDict
from training_utils import SaveOnIntervalCallback, visualise_training_logs
from plantos_env import PlantOSEnv

if __name__ == "__main__":
    training_run = "1M_recurrent"

    # Define paths for saving models and logs
    MODEL_DIR = os.path.join("PPO_Training/models", training_run)
    LOG_DIR = os.path.join("PPO_Training/logs", training_run)
    TENSORBOARD_LOG_DIR = "PPO_Training/logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create the vectorized environment
    n_envs = 50
    env_kwargs = {
        'grid_size': 21,
        'num_plants': 20,
        'num_obstacles': 12,
        'lidar_range': 6,
        'lidar_channels': 32,
        'thirsty_plant_prob': 0.5
    }

    # Create vectorized environment using the registered environment ID
    env = make_vec_env('PlantOS-v0', n_envs=n_envs, env_kwargs=env_kwargs)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 2. Define the model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    # Log the training
    new_logger = logger.configure(LOG_DIR, ["stdout", "csv"])
    model.set_logger(new_logger)

    # 3. Define Callbacks for saving and evaluation
    eval_callback = EvalCallback(
        env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=50000,
        deterministic=True,
        render=False
    )

    save_interval = 100000 #100k
    save_callback = SaveOnIntervalCallback(save_interval, MODEL_DIR)
    combined_callbacks = [eval_callback, save_callback]

    # 4. Train the model
    print("Starting RecurrentPPO training with Stable Baselines3...")
    model.learn(
        total_timesteps=1000000, # 1M
        callback=combined_callbacks,
        progress_bar=True
    )
    print("RecurrentPPO Training Finished.")
    print(f"Total timesteps trained: {model.num_timesteps}")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"This is the avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 5. Save the final model
    model.save(os.path.join(MODEL_DIR, f"recurrent_ppo_plantos_final_model-{training_run}"))

    # 6. Visualize the training curve and save it
    visualise_training_logs("rollout/ep_rew_mean", "Rewards", LOG_DIR)
    visualise_training_logs("rollout/ep_len_mean", "Episode Length", LOG_DIR)

    # 7. Close the environment
    env.close()
