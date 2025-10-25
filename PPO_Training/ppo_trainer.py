import os
import sys
import gymnasium as gym

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common import logger
from collections import OrderedDict
from training_utils import SaveOnIntervalCallback, visualise_training_logs
from plantos_env import PlantOSEnv

if __name__ == "__main__":
    training_run = "1M"

    # Define paths for saving models and logs
    MODEL_DIR = os.path.join("PPO_Training/models", training_run)
    LOG_DIR = os.path.join("PPO_Training/logs", training_run)
    TENSORBOARD_LOG_DIR = "PPO_Training/logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # PPO Hyperparameters
    config = OrderedDict([('batch_size', 64),
             ('clip_range', 0.18),
             ('ent_coef', 0.0),
             ('gae_lambda', 0.95),
             ('gamma', 0.999),
             ('learning_rate', 0.0003),
             ('n_epochs', 10),
             ('n_steps', 2048),
             ('n_timesteps', 200000.0),
             ('normalize', True),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', dict(net_arch=[256, 256])),
             ('normalize_kwargs', {'norm_obs': True, 'norm_reward': False})])

    # 1. Create the vectorized environment
    n_envs = 4
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
    if config['normalize']:
        env = VecNormalize(env, norm_obs=config['normalize_kwargs']['norm_obs'], norm_reward=config['normalize_kwargs']['norm_reward'], clip_obs=10.0)

    # 2. Define the model
    model = PPO(
        config['policy'],
        env,
        device='cpu',
        verbose=1,
        learning_rate=float(config['learning_rate']),
        batch_size=int(config['batch_size']),
        gamma=float(config['gamma']),
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        gae_lambda=config['gae_lambda'],
        n_epochs=config['n_epochs'],
        n_steps=config['n_steps'],
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
    print("Starting PPO training with Stable Baselines3...")
    model.learn(
        total_timesteps=config['n_timesteps'], # 1 Million
        callback=combined_callbacks
    )
    print("PPO Training Finished.")
    print(f"Total timesteps trained: {model.num_timesteps}")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"This is the avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 5. Save the final model
    model.save(os.path.join(MODEL_DIR, f"ppo_plantos_final_model-{training_run}"))

    # 6. Visualize the training curve and save it
    visualise_training_logs("rollout/ep_rew_mean", "Rewards", LOG_DIR)
    visualise_training_logs("rollout/ep_len_mean", "Episode Length", LOG_DIR)

    # 7. Close the environment
    env.close()