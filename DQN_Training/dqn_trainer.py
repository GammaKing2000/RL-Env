import os
import sys
import gymnasium as gym

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from plantos_env import PlantOSEnv # Assuming PlantOSEnv is compatible with Gym API
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    training_run = "1M"

    # Define paths for saving models and logs
    MODEL_DIR = os.path.join("DQN_Training/models", training_run)
    LOG_DIR = os.path.join("DQN_Training/logs", training_run)
    TENSORBOARD_LOG_DIR = "DQN_Training/logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create the vectorized environment
    n_envs = 4  # You can adjust this number
    env_kwargs = {
        'grid_size': 21,
        'num_plants': 20,
        'num_obstacles': 12,
        'lidar_range': 6,
        'lidar_channels': 32,
        'observation_mode': 'grid',
        'thirsty_plant_prob': 0.5
    }
    env = make_vec_env(PlantOSEnv, n_envs=n_envs, env_kwargs=env_kwargs)

    # 2. Define the model
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
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
        tensorboard_log=TENSORBOARD_LOG_DIR
    )

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
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    combined_callbacks = [eval_callback]

    # 4. Train the model
    print("Starting DQN training with Stable Baselines3...")
    model.learn(
        total_timesteps=1000000, # 1 Million
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
    log_file = os.path.join(LOG_DIR, "monitor.csv")
    if os.path.exists(log_file):
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.read_csv(log_file, skiprows=1)
            plt.figure(figsize=(10, 5))
            plt.plot(df['l'], df['r'])
            plt.xlabel("episodes")
            plt.ylabel("rewards")
            plt.title("Training Curve")
            plt.savefig(os.path.join(MODEL_DIR, f"training_curve-{training_run}.png"))
            print(f"Training curve saved to {os.path.join(MODEL_DIR, f'training_curve-{training_run}.png')}")
        except ImportError:
            print("Please install pandas and matplotlib to visualize the training curve.")
        except Exception as e:
            print(f"An error occurred during visualization: {e}")

    # 7. Close the environment
    env.close()