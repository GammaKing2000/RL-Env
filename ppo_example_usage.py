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
import sys
import traceback


def main(model_path: str, max_steps_per_episode=3000):
    """
    Run a trained agent in the PlantOS environment with full 2D and 3D visualization.
    
    Args:
        model_path: Path to the trained model zip file
        max_steps_per_episode: Maximum steps per episode
    """
    print("🌱 Starting PlantOS Environment with 2D and 3D Views for PPO")
    print("=" * 60)
    
    # Verify model file exists
    import os
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found: {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please check the path and try again.")
        sys.exit(1)
    
    try:
        # Create environment with Mars Explorer-like parameters
        print("Creating environment...")
        env = PlantOSEnv(
            grid_size=21, 
            num_plants=20, 
            num_obstacles=50, 
            lidar_range=6, 
            lidar_channels=32, 
            thirsty_plant_prob=0.5
        )
        print("✓ Environment created successfully")
        
        # Load model
        print(f"Loading model from: {model_path}")
        model = RecurrentPPO.load(model_path)
        print("✓ Model loaded successfully")
        
        # Verify observation space matches
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected by model: {model.observation_space.shape}")
        
        if obs.shape != model.observation_space.shape:
            print(f"❌ ERROR: Observation space mismatch!")
            print(f"   Environment: {obs.shape}")
            print(f"   Model expects: {model.observation_space.shape}")
            print(f"   This usually means the model was trained with different LIDAR settings.")
            env.close()
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("🚀 Starting episodes (Press Ctrl+C to stop)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR during initialization:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check that plantos_env.py is in the same directory")
        print("  2. Verify all dependencies are installed (pygame, ursina, etc.)")
        print("  3. Make sure the model path is correct")
        sys.exit(1)
    
    total_rewards = []
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n📺 Episode {episode}")
            print("-" * 60)
            
            # Reset environment
            obs, info = env.reset()
            # Reset the LSTM state
            lstm_states = None
            episode_reward = 0
            step = 0
            
            # Run episode
            for step in range(max_steps_per_episode):
                try:
                    # Take action from the trained model
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
                    
                    # Execute step
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Render both 2D and 3D views
                    env.render(mode='human')
                    
                    # Check if episode is done
                    if terminated or truncated:
                        reason = "terminated" if terminated else "truncated (timeout)"
                        print(f"  Episode ended: {reason}")
                        break
                    
                    # A small delay is good for visualization but not required
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"\n❌ ERROR at step {step}:")
                    print(f"   {type(e).__name__}: {str(e)}")
                    traceback.print_exc()
                    break
            
            # Episode summary
            print(f"\n✅ Episode {episode} Summary:")
            print("-" * 60)
            print(f"  Steps taken: {step + 1}")
            print(f"  Total reward: {episode_reward:.2f}")
            
            if 'cells_explored' in info:
                total_cells = 21 * 21
                print(f"  Cells explored: {info['cells_explored']}/{total_cells}")
                print(f"  Exploration: {info['exploration_pct']:.1f}%")
            
            if 'thirsty_plants' in info:
                print(f"  Thirsty plants remaining: {info['thirsty_plants']}")
            
            total_rewards.append(episode_reward)
            
            print(f"\nWaiting 2 seconds before next episode...")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Environment interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\n" + "=" * 60)
        print("🧹 Cleaning up...")
        try:
            env.close()
            print("✓ Environment closed successfully")
        except:
            pass
        
        # Print summary
        if total_rewards:
            print("\n" + "=" * 60)
            print("📊 FINAL SUMMARY")
            print("=" * 60)
            print(f"Episodes completed: {len(total_rewards)}")
            print(f"Average reward: {np.mean(total_rewards):.2f}")
            print(f"Best reward: {max(total_rewards):.2f}")
            print(f"Worst reward: {min(total_rewards):.2f}")
        
        print("\n✅ Script finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run a trained agent in the PlantOS environment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_usage.py "DQN_Training/models/160D_3kSteps_20M/best_model.zip"
  python example_usage.py "path/to/your/model.zip"
        """
    )
    parser.add_argument('model_path', type=str, help='Path to the trained model zip file')
    args = parser.parse_args()
    
    print(f"Script started with model: {args.model_path}\n")
    main(model_path=args.model_path)


# #!/usr/bin/env python3
# """
# Example usage of the PlantOS environment with 2D and 3D rendering.

# This script demonstrates how to create, use, and visualize the environment
# in both 2D (Pygame) and 3D (Ursina).
# """

# import numpy as np
# from plantos_env import PlantOSEnv
# import time
# from stable_baselines3 import DQN
# from sb3_contrib import RecurrentPPO
# import argparse

# def main(model_path: str, max_steps_per_episode=3000):
#     """
#     Run a trained agent in the PlantOS environment with full 2D and 3D visualization.
    
#     Args:
#         model_path: Path to the trained model zip file
#         max_steps_per_episode: Maximum steps per episode
#     """
#     print("🌱 Starting PlantOS Environment with 2D and 3D Views")
#     print("=" * 60)
    
#     # Create environment with Mars Explorer-like parameters
#     env = PlantOSEnv(grid_size=21, num_plants=20, num_obstacles=50, lidar_range=6, lidar_channels=32, thirsty_plant_prob=0.5)

#     model = DQN.load(model_path)
#     # model = RecurrentPPO.load(model_path)
    
#     total_rewards = []
    
#     try:
#         episode = 0
#         while True:
#             episode += 1
#             print(f"\n📺 Episode {episode}")
#             print("-" * 30)
            
#             # Reset environment
#             obs, info = env.reset()
#             episode_reward = 0
            
#             # Run episode
#             for step in range(max_steps_per_episode):
#                 # Take action from the trained model
#                 action, _ = model.predict(obs, deterministic=True)
                
#                 # Execute step
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 episode_reward += reward
                
#                 # Render both 2D and 3D views
#                 env.render(mode='human')
                
#                 # Check if episode is done
#                 if terminated or truncated:
#                     break
                
#                 # A small delay is good for visualization but not required
#                 time.sleep(0.05)
            
#             # Episode summary
#             print(f"\nEpisode {episode} finished after {step + 1} steps")
#             print(f"Total episode reward: {episode_reward:.2f}")
#             if 'cells_explored' in info:
#                 print(f"Cells explored: {info['cells_explored']}/{21*21}")
#                 print(f"Exploration: {info['exploration_pct']:.1f}%")     
#             total_rewards.append(episode_reward)
            
#             print("Waiting 2 seconds before next episode...")
#             time.sleep(2)
    
#     except KeyboardInterrupt:
#         print("\n⚠️  Environment interrupted by user")
    
#     finally:
#         # Clean up
#         env.close()
        
#         # Print summary
#         if total_rewards:
#             print("\n" + "=" * 60)
#             print("📊 FINAL SUMMARY")
#             print("=" * 60)
#             print(f"Episodes completed: {len(total_rewards)}")
#             print(f"Average reward: {np.mean(total_rewards):.2f}")
        
#         print("Environment closed successfully!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run a trained agent in the PlantOS environment.')
#     parser.add_argument('model_path', type=str, help='Path to the trained model zip file')
#     args = parser.parse_args()
#     main(model_path=args.model_path)