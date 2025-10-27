import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from mcts_simple import OpenLoopMCTS, Game
from plantos_env import PlantOSEnv
import numpy as np
import time

class PlantOSGame(Game):
    def __init__(self, plantos_env):
        self.env = plantos_env
        self.current_state, _ = self.env.reset()
        self.terminated = False
        self.truncated = False
        self.last_step_reward = 0

    def get_state(self):
        return self.current_state

    def number_of_players(self):
        return 1

    def current_player(self):
        return 0

    def possible_actions(self):
        return list(range(self.env.action_space.n))

    def take_action(self, action):
        if not self.has_outcome():
            self.current_state, reward, self.terminated, self.truncated, _ = self.env.step(action)
            self.last_step_reward = reward
            return reward
        return 0

    def has_outcome(self):
        return self.terminated or self.truncated

    def winner(self):
        if self.last_step_reward > 0:
            return [self.current_player()]
        else:
            return []
    
    def render(self):
        # Render the environment - this method is required by mcts_simple
        self.env.render(mode='human')

if __name__ == '__main__':
    # Define the path to the trained MCTS model
    training_run = "5M__exp1"
    model_path = r"MCTS_Training\plantos.mcts"

    if not os.path.exists(model_path):
        print(f"Error: MCTS model not found at {model_path}")
        sys.exit(1)

    print(f"Loading MCTS model from: {model_path}")

    # Create the environment for testing
    test_env = PlantOSEnv(grid_size=21, num_plants=10, num_obstacles=12, lidar_range=4, lidar_channels=12)
    game = PlantOSGame(test_env)

    # Create an MCTS instance for evaluation and load the trained model
    eval_mcts = OpenLoopMCTS(game, training=False)
    eval_mcts.load(model_path)

    print("Starting MCTS model test...")
    episode_total_reward = 0
    step_count = 0
    max_steps_per_episode = 3000 # Max steps for the environment
    eval_simulations_per_step = 100 # Number of MCTS simulations to run for each evaluation step

    try:
        while not game.has_outcome() and step_count < max_steps_per_episode:
            # Perform MCTS search to decide the best action for the current game state
            eval_mcts.self_play(iterations=eval_simulations_per_step)
            
            # Choose the best action based on the search results
            action = eval_mcts.choose_best_action(False)

            # Take the action in the environment and get the step reward
            step_reward = game.take_action(action)
            episode_total_reward += step_reward
            step_count += 1

            # Render the environment (optional)
            test_env.render(mode='human')
            time.sleep(0.05) # Small delay for visualization

            # Advance the MCTS tree's root to the chosen action's child node
            if action in eval_mcts.root.children:
                eval_mcts.root = eval_mcts.root.children[action]
            else:
                print(f"Warning: Chosen action {action} not found in children of current MCTS root. Ending episode early.")
                eval_mcts.root = None
                break

        print(f"\nTest finished after {step_count} steps.")
        print(f"Total Episode Reward: {episode_total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        test_env.close()
        print("Environment closed.")
