#!/usr/bin/env python3
"""
Example usage of the PlantOS environment with 2D and 3D rendering.

This script demonstrates how to create, use, and visualize the environment
in both 2D (Pygame) and 3D (Ursina).
"""

import numpy as np
from plantos_env import PlantOSEnv
import time
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
import argparse

def main(model_path: str, model_type: str = 'auto', max_steps_per_episode=300):
    """
    Run a trained agent in the PlantOS environment with full 2D and 3D visualization.
    
    Args:
        model_path: Path to the trained model zip file
        model_type: Type of model ('dqn', 'ppo', or 'auto' to detect from filename)
        max_steps_per_episode: Maximum steps per episode
    """
    print("🌱 Starting PlantOS Environment with 2D and 3D Views")
    print("=" * 60)
    
    # Auto-detect model type from filename if not specified
    if model_type == 'auto':
        if 'dqn' in model_path.lower():
            model_type = 'dqn'
        elif 'ppo' in model_path.lower():
            model_type = 'ppo'
        else:
            print("⚠️  Could not auto-detect model type from filename.")
            print("Please specify --model-type dqn or --model-type ppo")
            return
    
    # Create environment with Mars Explorer-like parameters
    env = PlantOSEnv(grid_size=25, num_plants=10, num_obstacles=12, lidar_range=6, lidar_channels=16)
    
    # Load the appropriate model
    if model_type == 'dqn':
        print("📦 Loading DQN model...")
        model = DQN.load(model_path)
        use_lstm = False
    elif model_type == 'ppo':
        print("📦 Loading RecurrentPPO model...")
        model = RecurrentPPO.load(model_path)
        use_lstm = True
    else:
        print(f"❌ Unknown model type: {model_type}")
        print("Valid options: 'dqn', 'ppo', or 'auto'")
        return
    
    print(f"✅ Model loaded successfully ({model_type.upper()})")
    
    total_rewards = []
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n📺 Episode {episode}")
            print("-" * 30)
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            
            # For LSTM models, initialize hidden states
            if use_lstm:
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Take action from the trained model
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
                
                # Execute step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Render both 2D and 3D views
                env.render(mode='human')
                
                # Check if episode is done
                if terminated or truncated:
                    break
                
                # A small delay is good for visualization but not required
                time.sleep(0.05)
            
            # Episode summary
            print(f"\nEpisode {episode} finished after {step + 1} steps")
            print(f"Total episode reward: {episode_reward:.2f}")
            print(f"Exploration: {info['exploration_percentage']:.1f}%")
            print(f"Final thirsty plants: {info['thirsty_plants']}")
            
            total_rewards.append(episode_reward)
            
            print("Waiting 2 seconds before next episode...")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n⚠️  Environment interrupted by user")
    
    finally:
        # Clean up
        env.close()
        
        # Print summary
        if total_rewards:
            print("\n" + "=" * 60)
            print("📊 FINAL SUMMARY")
            print("=" * 60)
            print(f"Episodes completed: {len(total_rewards)}")
            print(f"Average reward: {np.mean(total_rewards):.2f}")
        
        print("Environment closed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent in the PlantOS environment.')
    parser.add_argument('model_path', type=str, help='Path to the trained model zip file')
    parser.add_argument('--model-type', type=str, default='auto', choices=['auto', 'dqn', 'ppo'],
                        help='Type of model: dqn, ppo, or auto (auto-detect from filename)')
    args = parser.parse_args()
    main(model_path=args.model_path, model_type=args.model_type)