import gymnasium as gym
from sb3_contrib import RecurrentPPO
import torch
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
import time
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
    PROGRESSIVE curriculum: Start with low threshold, gradually increase.
    This prevents the agent from getting stuck on impossible-for-now mazes.
    """
    def __init__(self, env, initial_threshold=30.0, max_threshold=100.0, threshold_increment=5.0):
        super().__init__(env)
        self.maze_completed = False
        self.episode_count = 0
        self.successful_explorations = 0
        self.current_maze_seed = None
        self.persistent_visit_counts = None
        
        # PROGRESSIVE CURRICULUM PARAMETERS
        self.exploration_threshold = initial_threshold  # Start low!
        self.max_threshold = max_threshold
        self.threshold_increment = threshold_increment
        self.episodes_on_current_maze = 0
        self.max_episodes_per_maze = 50  # Give up after 50 attempts
        
    def reset(self, **kwargs):
        """Only generate new maze if threshold reached OR max attempts exceeded."""
        self.episode_count += 1
        self.episodes_on_current_maze += 1
        
        # Remove 'seed' from kwargs to avoid duplicate argument
        kwargs.pop('seed', None)
        
        # Check if we should move to a new maze
        timeout = self.episodes_on_current_maze >= self.max_episodes_per_maze
        
        if self.maze_completed or timeout:
            if self.maze_completed:
                # Silently increase difficulty - avoid Jupyter display bugs
                self.exploration_threshold = min(
                    self.exploration_threshold + self.threshold_increment,
                    self.max_threshold
                )
                self.successful_explorations += 1
            # else:
            #     # Timeout - try again at same difficulty
            
            self.maze_completed = False
            self.episodes_on_current_maze = 0
            
            # Generate new maze
            self.current_maze_seed = np.random.randint(0, 10000)
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
            self.persistent_visit_counts = None
        else:
            # Keep the same maze
            if self.current_maze_seed is None:
                self.current_maze_seed = np.random.randint(0, 10000)
            
            obs, info = self.env.reset(seed=self.current_maze_seed, **kwargs)
            
            # Restore visit counts
            if self.persistent_visit_counts is not None:
                self.env.visit_counts = self.persistent_visit_counts.copy()
            else:
                self.persistent_visit_counts = self.env.visit_counts.copy()
            
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if exploration reached CURRENT threshold (not always 100%)
        if info['exploration_percentage'] >= self.exploration_threshold:
            self.maze_completed = True
        
        # Save visit counts at every step
        if self.persistent_visit_counts is not None:
            self.persistent_visit_counts = self.env.visit_counts.copy()
            
        return obs, reward, terminated, truncated, info


# ============================================================================
# Helper function to create environment factory for vectorized envs
# ============================================================================

def make_env_wrapper(env_kwargs, rank=0):
    """Helper function to create environment factory for vectorized training."""
    def _init():
        env = PlantOSEnv(**env_kwargs)
        env = CurriculumWrapper(env, initial_threshold=30.0, max_threshold=100.0)
        # Monitor each env with separate log file to avoid CSV corruption
        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


# ============================================================================
# OPTION 1: RecurrentPPO with LSTM (RECOMMENDED for exploration tasks)
# ============================================================================
# RecurrentPPO is better suited for this task because it has memory,
# allowing the agent to remember which areas have been explored

def train_with_recurrent_ppo(n_envs=4):
    """Train using RecurrentPPO with LSTM policy for memory-based exploration."""
    
    # Environment parameters
    env_kwargs = {
        'grid_size': 25,
        'num_plants': 10,
        'num_obstacles': 12,
        'lidar_range': 6,      # DOUBLED - agent needs better vision!
        'lidar_channels': 16   # MORE channels for better sensing
    }
    
    # Create VECTORIZED environment with N parallel instances
    print(f"Creating {n_envs} parallel environments...")
    env_fns = [make_env_wrapper(env_kwargs, rank=i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    
    print("=" * 50)
    print(f"Training with RecurrentPPO (LSTM Policy) - {n_envs} parallel envs")
    print("WITH CURRICULUM LEARNING (same maze until 100% explored)")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create RecurrentPPO model with LSTM
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,           # Standard learning rate for PPO
        n_steps=1024,                 # Steps per environment before update
        batch_size=128,                # Minibatch size
        n_epochs=10,                  # Number of epochs when optimizing
        gamma=0.99,                   # Discount factor
        gae_lambda=0.95,              # GAE lambda for advantage estimation
        clip_range=0.2,               # Clipping parameter
        ent_coef=0.02,                # Entropy coefficient for exploration
        vf_coef=0.5,                  # Value function coefficient
        max_grad_norm=0.5,            # Max gradient norm
        verbose=1,
        tensorboard_log=f"{log_dir}tensorboard/",
        device=device,
        policy_kwargs=dict(
            lstm_hidden_size=2562,     # INCREASED: 512 for better memory (was 256)
            n_lstm_layers=1,          # INCREASED: 2 layers for deeper memory (was 1)
            enable_critic_lstm=True,  # Use LSTM for critic too
            net_arch=[128, 128],      # Additional feedforward layers before LSTM
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
    total_timesteps = 100000  # Train MUCH longer for proper learning
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

def train_with_improved_dqn(n_envs=4):
    """Train using DQN with improved hyperparameters."""
    
    # Environment parameters
    env_kwargs = {
        'grid_size': 25,
        'num_plants': 10,
        'num_obstacles': 12,
        'lidar_range': 6,
        'lidar_channels': 16
    }
    
    # Create VECTORIZED environment with N parallel instances
    print(f"Creating {n_envs} parallel environments...")
    env_fns = [make_env_wrapper(env_kwargs, rank=i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    
    print("=" * 50)
    print(f"Training with Improved DQN - {n_envs} parallel envs")
    print("WITH CURRICULUM LEARNING (same maze until 100% explored)")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create DQN model with FIXED hyperparameters for exploration
    model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-4,              # SMALLER LR - more stable
    buffer_size=2000000,              # Larger buffer for more diverse experiences
    learning_starts=5000,            # START EARLIER - begin learning sooner
    batch_size=64,                   # Smaller batches - less overfitting
    tau=0.005,                       # Faster target updates
    gamma=0.99,                      # Standard discount (was too high)
    train_freq=4,                    # Train more frequently
    gradient_steps=1,
    target_update_interval=5000,     # LESS frequent - more stable
    exploration_fraction=0.7,         # EXPLORE for 70% of training (7M steps)
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,      # Minimum 5% exploration
    max_grad_norm=10.0,
    verbose=1,
    device=device,
    policy_kwargs=dict(
        net_arch=[512, 512, 256]     # LARGER network - better capacity
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
        self.maze_completion_count = 0
    
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
                        if ep_info['exploration_percentage'] >= 100.0:
                            self.maze_completion_count += 1
                
                if explorations:
                    mean_exploration = np.mean(explorations)
                    self.exploration_history.append(mean_exploration)
                    
                    # Log to file instead of printing (avoid Jupyter recursion bug)
                    with open(f"{self.log_dir}training_log.txt", "a") as f:
                        f.write(f"[Step {self.n_calls}] Mean Exploration: {mean_exploration:.2f}%\n")
                        f.write(f"Mazes completed: {self.maze_completion_count}\n")
                    
                    if mean_exploration > self.best_mean_exploration:
                        self.best_mean_exploration = mean_exploration
        
        return True


# ============================================================================
# Helper function to plot learning curves
# ============================================================================

def plot_learning_curve(log_dir, title="Learning Curve"):
    """Plot the learning curve from training logs."""
    try:
        # Load results from all environment subdirectories
        results = load_results(log_dir)
        
        if len(results) == 0:
            print("No results to plot yet.")
            return
        
        # Sort by timesteps to ensure chronological order
        results = results.sort_values('t')
        
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
    
    # Create environment MATCHING THE TRAINING PARAMETERS!
    env = PlantOSEnv(
        grid_size=25,
        num_plants=10,
        num_obstacles=12,
        lidar_range=6,     # MUST MATCH TRAINING!
        lidar_channels=16,  # MUST MATCH TRAINING!
        render_mode='2d'
    )
    
    # CRITICAL FIX: Wrap with curriculum wrapper to match training environment
    # This ensures visit counts persist and agent behavior matches training
    env = CurriculumWrapper(env, initial_threshold=100.0, max_threshold=100.0)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    # Use RecurrentPPO.load() or DQN.load() depending on which you trained
    try:
        model = RecurrentPPO.load(model_path, device=device)
        print("Loaded RecurrentPPO model")
        use_lstm = True
    except:
        model = DQN.load(model_path, device=device)
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
        
        last_progress_print = 0
        
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
            
            # Print progress every 100 steps to show it's not frozen
            if steps - last_progress_print >= 100:
                print(f"  Step {steps}: Exploration {info['exploration_percentage']:.1f}%, Reward: {total_reward:.1f}")
                last_progress_print = steps
            
            # Render (access unwrapped env to use custom render method)
            env.unwrapped.render('2d')
            
            # Small delay to prevent pygame window from becoming unresponsive
            # and to make visualization easier to follow
            time.sleep(0.01)  # 10ms delay = ~100 FPS max (faster than pygame's 30 FPS limit)
        
        print(f"\n✅ Episode finished in {steps} steps")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Exploration: {info['exploration_percentage']:.2f}%")
        print(f"   Thirsty plants remaining: {info['thirsty_plants']}")
        print(f"   Plants watered: {info['total_plants'] - info['thirsty_plants']}/{info['total_plants']}")
    
    env.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PlantOS Environment Training & Testing")
    print("="*60)
    print("\nChoose an option:")
    print("1. Train with RecurrentPPO (LSTM) - RECOMMENDED for exploration")
    print("2. Train with Improved DQN")
    print("3. Test existing model")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "3":
        # Option 3: Test existing model
        print("\n" + "="*60)
        print("Test Existing Model")
        print("="*60)
        
        # Prompt for model path
        print("\nExample paths:")
        print("  - train_improved/models/recurrent_ppo_final")
        print("  - train_improved/models/dqn_improved_final")
        print("  - train_improved/models/recurrent_ppo_model_500000_steps")
        
        model_path = input("\nEnter model path (without .zip extension): ").strip()
        
        # Verify file exists
        if not os.path.exists(f"{model_path}.zip"):
            print(f"\n❌ Error: Model file '{model_path}.zip' not found!")
            print("Please check the path and try again.")
            exit()
        
        # Ask for number of test episodes
        num_episodes_input = input("\nNumber of test episodes (default: 3): ").strip()
        num_episodes = int(num_episodes_input) if num_episodes_input else 3
        
        print(f"\n🚀 Loading and testing model: {model_path}")
        test_trained_model(model_path, num_episodes=num_episodes)
        
        print("\n" + "="*60)
        print("Testing complete!")
        print("="*60)
        
    elif choice in ["1", "2"]:
        # Options 1 & 2: Training
        # Ask for number of parallel environments
        n_envs_input = input("\nNumber of parallel environments (default: 4): ").strip()
        n_envs = int(n_envs_input) if n_envs_input else 4
        
        if choice == "1":
            model = train_with_recurrent_ppo(n_envs=n_envs)
            model_path = f"{models_dir}recurrent_ppo_final"
        elif choice == "2":
            model = train_with_improved_dqn(n_envs=n_envs)
            model_path = f"{models_dir}dqn_improved_final"
        
        # Ask if user wants to test the trained model
        test = input("\nTest the trained model? (y/n): ").strip().lower()
        if test == 'y':
            num_episodes_input = input("Number of test episodes (default: 3): ").strip()
            num_episodes = int(num_episodes_input) if num_episodes_input else 3
            test_trained_model(model_path, num_episodes=num_episodes)
        
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
    
    else:
        print("Invalid choice. Exiting.")
        exit()