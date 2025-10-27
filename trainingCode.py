import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt
import numpy as np
from plantos_env import PlantOSEnv

# Create directories for logs and models
log_dir = "train_improved/gym/"
models_dir = "train_improved/models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ============================================================================
# NEW: Wrapper to keep maze until 100% exploration
# ============================================================================

class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that keeps the same maze until the agent achieves 100% exploration.
    Only then does it generate a new maze (curriculum learning).
    """
    def __init__(self, env):
        super().__init__(env)
        self.maze_completed = False
        self.episode_count = 0
        self.successful_explorations = 0
        self.current_maze_seed = None  # NEW: Track current maze seed
        
    def reset(self, **kwargs):
        """Only generate new maze if previous one was fully explored."""
        self.episode_count += 1
        
        # Remove 'seed' from kwargs to avoid duplicate argument
        kwargs.pop('seed', None)
        
        # If maze was completed in last episode, generate a new one
        if self.maze_completed:
            print(f"\n{'='*60}")
            print(f"🎉 MAZE COMPLETED! Generating new maze...")
            print(f"Successful explorations so far: {self.successful_explorations}")
            print(f"{'='*60}\n")
            self.maze_completed = False
            
            # NEW: Generate new random seed for new maze
            self.current_maze_seed = np.random.randint(0, 10000)
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
        else:
            # Keep the same maze - reuse the current seed
            if self.current_maze_seed is None:
                # First episode - create initial maze
                self.current_maze_seed = np.random.randint(0, 10000)
            
            # ⚠️ CRITICAL: Reset with SAME seed = same maze layout
            # This will regenerate plants as thirsty again!
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
            print(f"📍 Episode {self.episode_count}: Reusing maze (seed={self.current_maze_seed})")
            
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if exploration reached 100%
        if info['exploration_percentage'] >= 100.0:
            self.maze_completed = True
            self.successful_explorations += 1
            
        return obs, reward, terminated, truncated, info


# ============================================================================
# OPTION 1: RecurrentPPO with LSTM (RECOMMENDED for exploration tasks)
# ============================================================================
# RecurrentPPO is better suited for this task because it has memory,
# allowing the agent to remember which areas have been explored

def train_with_recurrent_ppo():
    """Train using RecurrentPPO with LSTM policy for memory-based exploration."""
    
    # Create the environment with improved parameters
    env = PlantOSEnv(
        grid_size=25,
        num_plants=10,
        num_obstacles=12,
        lidar_range=4,
        lidar_channels=12
    )
    
    # NEW: Wrap with curriculum wrapper
    env = CurriculumWrapper(env)
    
    # Wrap with Monitor to track episode statistics
    env = Monitor(env, log_dir)
    
    # Verify the environment is properly configured
    check_env(env, warn=True)
    
    print("=" * 50)
    print("Training with RecurrentPPO (LSTM Policy)")
    print("WITH CURRICULUM LEARNING (same maze until 100% explored)")
    print("=" * 50)
    
    # Create RecurrentPPO model with LSTM
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,           # Standard learning rate for PPO
        n_steps=2048,                 # Steps per environment before update
        batch_size=64,                # Minibatch size
        n_epochs=10,                  # Number of epochs when optimizing
        gamma=0.99,                   # Discount factor
        gae_lambda=0.95,              # GAE lambda for advantage estimation
        clip_range=0.2,               # Clipping parameter
        ent_coef=0.01,                # Entropy coefficient for exploration
        vf_coef=0.5,                  # Value function coefficient
        max_grad_norm=0.5,            # Max gradient norm
        verbose=1,
        tensorboard_log=f"{log_dir}tensorboard/",
        policy_kwargs=dict(
            lstm_hidden_size=256,     # LSTM hidden state size
            n_lstm_layers=1,          # Number of LSTM layers
            enable_critic_lstm=True,  # Use LSTM for critic too
        )
    )
    
    # Checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="recurrent_ppo_model"
    )
    
    # Custom callback for logging
    eval_callback = EvaluationCallback(log_dir)
    
    # Train the model
    print("\nStarting training...")
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{models_dir}recurrent_ppo_final"
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Evaluate the trained model
    print("\n" + "=" * 50)
    print("Evaluating trained model...")
    print("=" * 50)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot learning curve
    plot_learning_curve(log_dir, "RecurrentPPO Learning Curve")
    
    return model


# ============================================================================
# OPTION 2: Improved DQN (if you prefer to stick with DQN)
# ============================================================================

def train_with_improved_dqn():
    """Train using DQN with improved hyperparameters."""
    
    # Create the environment
    env = PlantOSEnv(
        grid_size=25,
        num_plants=10,
        num_obstacles=12,
        lidar_range=4,
        lidar_channels=12
    )
    
    # NEW: Wrap with curriculum wrapper
    env = CurriculumWrapper(env)
    
    env = Monitor(env, log_dir)
    check_env(env, warn=True)
    
    print("=" * 50)
    print("Training with Improved DQN")
    print("WITH CURRICULUM LEARNING (same maze until 100% explored)")
    print("=" * 50)
    
    # Create DQN model with FIXED hyperparameters for exploration
    model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,              # INCREASED - faster initial learning
    buffer_size=100000,              # Sufficient buffer
    learning_starts=5000,            # START EARLIER - begin learning sooner
    batch_size=128,                  # Larger batches for stability
    tau=0.005,                       # Faster target updates
    gamma=0.99,                      # Standard discount (was too high)
    train_freq=4,                    # Train more frequently
    gradient_steps=1,
    target_update_interval=1000,    # More frequent target updates
    exploration_fraction=0.8,        # CRITICAL: Explore for 80% of training!
    exploration_initial_eps=1.0,
    exploration_final_eps=0.15,      # KEEP EXPLORING - never go below 15%
    max_grad_norm=10.0,
    verbose=1,
    policy_kwargs=dict(
        net_arch=[256, 256]          # SIMPLER network - easier to train
        )
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=models_dir,
        name_prefix="dqn_improved_model"
    )
    
    eval_callback = EvaluationCallback(log_dir)
    
    # Train the model
    print("\nStarting training...")
    total_timesteps = 10000000  # Train longer
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=100
    )
    
    # Save final model
    final_model_path = f"{models_dir}dqn_improved_final"
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluating trained model...")
    print("=" * 50)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot learning curve
    plot_learning_curve(log_dir, "Improved DQN Learning Curve")
    
    return model


# ============================================================================
# Callback for periodic evaluation during training
# ============================================================================

class EvaluationCallback(BaseCallback):
    """Custom callback for logging exploration progress."""
    
    def __init__(self, log_dir, eval_freq=10000):
        super().__init__()
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.best_mean_exploration = 0
        self.exploration_history = []
        self.maze_completion_count = 0  # NEW: Track maze completions
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Get info from the last episode
            if len(self.model.ep_info_buffer) > 0:
                recent_episodes = list(self.model.ep_info_buffer)[-10:]
                
                # Extract exploration percentages if available
                explorations = []
                for ep_info in recent_episodes:
                    if 'exploration_percentage' in ep_info:
                        explorations.append(ep_info['exploration_percentage'])
                        # NEW: Count completed mazes
                        if ep_info['exploration_percentage'] >= 100.0:
                            self.maze_completion_count += 1
                
                if explorations:
                    mean_exploration = np.mean(explorations)
                    self.exploration_history.append(mean_exploration)
                    
                    print(f"\n[Step {self.n_calls}] Mean Exploration: {mean_exploration:.2f}%")
                    print(f"Mazes completed so far: {self.maze_completion_count}")
                    
                    if mean_exploration > self.best_mean_exploration:
                        self.best_mean_exploration = mean_exploration
                        print(f"New best exploration rate: {self.best_mean_exploration:.2f}%")
        
        return True


# ============================================================================
# Helper function to plot learning curves
# ============================================================================

def plot_learning_curve(log_dir, title="Learning Curve"):
    """Plot the learning curve from training logs."""
    try:
        results = load_results(log_dir)
        
        if len(results) == 0:
            print("No results to plot yet.")
            return
        
        x, y = ts2xy(results, 'timesteps')
        
        # Plot raw rewards
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(x, y, alpha=0.3, color='blue', label='Raw Reward')
        
        # Plot smoothed rewards (moving average)
        if len(y) > 100:
            window = min(100, len(y) // 10)
            y_smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smoothed = x[:len(y_smoothed)]
            ax1.plot(x_smoothed, y_smoothed, color='red', linewidth=2, label='Smoothed Reward')
        
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        x_len, y_len = ts2xy(results, 'timesteps')
        ax2.plot(x_len, results['l'].values, alpha=0.3, color='green')
        
        if len(results['l'].values) > 100:
            window = min(100, len(results['l'].values) // 10)
            y_len_smoothed = np.convolve(results['l'].values, np.ones(window)/window, mode='valid')
            x_len_smoothed = x_len[:len(y_len_smoothed)]
            ax2.plot(x_len_smoothed, y_len_smoothed, color='orange', linewidth=2)
        
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}learning_curve.png", dpi=150)
        print(f"\nLearning curve saved to: {log_dir}learning_curve.png")
        plt.show()
        
    except Exception as e:
        print(f"Error plotting learning curve: {e}")


# ============================================================================
# Testing function for trained model
# ============================================================================

def test_trained_model(model_path, num_episodes=5):
    """Test a trained model and visualize its performance."""
    
    # Create environment
    env = PlantOSEnv(
        grid_size=25,
        num_plants=10,
        num_obstacles=12,
        lidar_range=4,
        lidar_channels=12,
        render_mode='2d'
    )
    
    # Load model
    # Use RecurrentPPO.load() or DQN.load() depending on which you trained
    try:
        model = RecurrentPPO.load(model_path)
        print("Loaded RecurrentPPO model")
        use_lstm = True
    except:
        model = DQN.load(model_path)
        print("Loaded DQN model")
        use_lstm = False
    
    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # For LSTM models
        if use_lstm:
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}")
        print(f"{'='*50}")
        
        while not done:
            # Get action from model
            if use_lstm:
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Render
            env.render('2d')
        
        print(f"Episode finished in {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Exploration: {info['exploration_percentage']:.2f}%")
        print(f"Thirsty plants remaining: {info['thirsty_plants']}")
    
    env.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PlantOS Environment Training")
    print("="*60)
    print("\nChoose training algorithm:")
    print("1. RecurrentPPO (LSTM) - RECOMMENDED for exploration")
    print("2. Improved DQN")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        model = train_with_recurrent_ppo()
        model_path = f"{models_dir}recurrent_ppo_final"
    elif choice == "2":
        model = train_with_improved_dqn()
        model_path = f"{models_dir}dqn_improved_final"
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    # Ask if user wants to test the trained model
    test = input("\nTest the trained model? (y/n): ").strip().lower()
    if test == 'y':
        test_trained_model(model_path, num_episodes=3)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
