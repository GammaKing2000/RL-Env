import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from mcts_simple import MCTS, Game
from plantos_env import PlantOSEnv
import numpy as np

class PlantOSGame(Game):
    def __init__(self, plantos_env):
        self.env = plantos_env
        self.current_state, _ = self.env.reset()
        self.terminated = False
        self.truncated = False
        self.last_step_reward = 0 # Store reward from the last step

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
            self.last_step_reward = reward # Store the reward for this step
            return reward # Return the reward for this step
        return 0 # Return 0 reward if game has outcome

    def has_outcome(self):
        return self.terminated or self.truncated

    def winner(self):
        # For OpenLoopMCTS in a single-player game, the 'winner' concept is less direct.
        # The mcts-simple library uses this for backpropagation.
        # A common approach for single-player is to return a value indicating the outcome.
        # Here, we'll return a list with the current player if the last step reward was positive, 
        # otherwise an empty list. This might need further tuning depending on how mcts-simple
        # interprets this for OpenLoopMCTS value updates.
        if self.last_step_reward > 0:
            return [self.current_player()]
        else:
            return []

if __name__ == '__main__':
    training_run = "5M__exp1"

    # Define paths for saving models and logs
    MODEL_DIR = os.path.join("MCTS_Training/models", training_run)
    LOG_DIR = os.path.join("MCTS_Training/logs", training_run)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create the environment
    env = PlantOSEnv(grid_size=21, num_plants=10, num_obstacles=12, lidar_range=4, lidar_channels=12)
    
    # 2. Wrap the environment in the MCTS Game class
    game = PlantOSGame(env)
    
    # 3. Create the MCTS tree
    # We use OpenLoopMCTS for single-player games
    from mcts_simple import OpenLoopMCTS
    
    # MCTS Hyperparameters
    # c_param: The exploration constant. A higher value encourages more exploration.
    # The default value is 1.41 (sqrt(2)).
    c_param = 1.41
    
    tree = OpenLoopMCTS(game, training=True)
    
    # 4. Run the self-play training
    # iterations: The number of MCTS simulations to run for each move.
    # A higher number of iterations leads to better moves but slower training.
    training_iterations = 10000
    print("Starting MCTS self-play training...")
    tree.self_play(iterations=training_iterations) # Increased iterations for longer training
    print("MCTS Training Finished.")
    
    # 5. Save the trained model
    model_path = os.path.join(MODEL_DIR, "plantos.mcts")
    tree.save(model_path)
    print(f"MCTS model saved to {model_path}")
    
    # 6. Evaluate the trained agent
    print("Evaluating the trained agent...")
    
    total_rewards = []
    num_eval_episodes = 100
    eval_simulations_per_step = 100 # Number of MCTS simulations to run for each evaluation step

    for i in range(num_eval_episodes):
        # Create a new environment and game for each evaluation episode
        current_env = PlantOSEnv(grid_size=21, num_plants=10, num_obstacles=12, lidar_range=4, lidar_channels=12)
        game = PlantOSGame(current_env)
        
        # Create a new MCTS instance for evaluation.
        # Load the trained tree into this instance.
        eval_mcts = OpenLoopMCTS(game, training=False)
        eval_mcts.load(model_path) # Load the trained tree into the evaluation MCTS

        episode_total_reward = 0 # Accumulate total reward for the episode
        while not game.has_outcome():
            # Perform MCTS search to decide the best action for the current game state
            # This will populate the children of the current root node in eval_mcts
            eval_mcts.do_rollout(iterations=eval_simulations_per_step) # Perform search for the current state
            
            # Choose the best action based on the search results
            # The 'choose_best_action' method should now find children
            action = eval_mcts.choose_best_action(False) 

            # Take the action in the environment and get the step reward
            step_reward = game.take_action(action)
            episode_total_reward += step_reward # Accumulate step reward

            # Advance the MCTS tree's root to the chosen action's child node
            # This is how MCTS "moves" through the game tree
            if action in eval_mcts.root.children:
                eval_mcts.root = eval_mcts.root.children[action]
            else:
                # If the chosen action leads to an unexplored path in the loaded tree,
                # we need to decide how to handle it. For evaluation, it's often
                # best to stop if we go off the trained path.
                print(f"Warning: Chosen action {action} not found in children of current MCTS root. Ending episode early.")
                eval_mcts.root = None # Mark as off-path
                break # End episode early

        total_rewards.append(episode_total_reward)
        print(f"Episode {i+1}: Total Reward = {episode_total_reward}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Evaluation finished. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save evaluation results to a CSV file
    import pandas as pd
    df = pd.DataFrame({'total_reward': total_rewards})
    df.to_csv(os.path.join(LOG_DIR, 'evaluations.csv'), index=False)