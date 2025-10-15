#!/usr/bin/env python3
"""
Example usage of the PlantOS environment with 2D and 3D rendering.

This script demonstrates how to create, use, and visualize the environment
in both 2D (Pygame) and 3D (Ursina).
"""

import numpy as np
from plantos_env import PlantOSEnv
import time

def main(max_steps_per_episode=300):
    """
    Run a random agent in the PlantOS environment with full 2D and 3D visualization.
    
    Args:
        episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
    """
    print("🌱 Starting PlantOS Environment with 2D and 3D Views")
    print("=" * 60)
    
    # Create environment with Mars Explorer-like parameters
    env = PlantOSEnv(grid_size=21, num_plants=8, num_obstacles=12, lidar_range=6, lidar_channels=32)
    
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
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Take random action
                action = env.action_space.sample()
                
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
            print(f"\nEpisode {episode + 1} finished after {step + 1} steps")
            print(f"Total episode reward: {episode_reward:.2f}")
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
    main()