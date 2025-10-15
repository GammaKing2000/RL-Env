#!/usr/bin/env python3
"""
Simple training script for the PlantOS environment.
This script demonstrates a basic Q-learning approach to train an agent.
"""

import numpy as np
from plantos_env import PlantOSEnv
import pickle
import os

class SimpleQLearningAgent:
    """
    A simple Q-learning agent for the PlantOS environment.
    
    This agent uses a basic Q-learning algorithm with epsilon-greedy exploration
    to learn how to water plants efficiently.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_size: Size of the state representation
            action_size: Number of possible actions
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        # Since our state space is continuous, we'll use a simple approach
        # by discretizing the state space based on plant positions and rover position
        self.q_table = {}
        
        # Training statistics
        self.training_rewards = []
        self.episode_lengths = []
    
    def get_state_key(self, obs):
        """
        Convert continuous observation to a discrete state key.
        
        Args:
            obs: Observation array from the environment
            
        Returns:
            String representation of the state for Q-table lookup
        """
        # Create a simplified state representation
        # Focus on relative positions of thirsty plants and rover
        thirsty_plants = np.where(obs[2] == 1)  # Channel 2: thirsty plants
        rover_pos = np.where(obs[3] == 1)       # Channel 3: rover position
        
        if len(rover_pos[0]) == 0:
            return "invalid_state"
        
        rover_x, rover_y = rover_pos[0][0], rover_pos[1][0]
        
        # Create state key based on rover position and nearby thirsty plants
        state_parts = [f"rover_{rover_x}_{rover_y}"]
        
        # Add information about thirsty plants relative to rover
        for i in range(len(thirsty_plants[0])):
            plant_x, plant_y = thirsty_plants[0][i], thirsty_plants[1][i]
            # Calculate relative distance and direction
            dx = plant_x - rover_x
            dy = plant_y - rover_y
            state_parts.append(f"plant_{dx}_{dy}")
        
        # Sort to ensure consistent state representation
        state_parts.sort()
        return "_".join(state_parts)
    
    def choose_action(self, obs, training=True):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            obs: Current observation
            training: Whether we're in training mode
            
        Returns:
            Chosen action
        """
        state_key = self.get_state_key(obs)
        
        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: choose best action
            return np.argmax(self.q_table[state_key])
    
    def update(self, obs, action, reward, next_obs, done):
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state: no future rewards
            max_future_q = 0
        else:
            # Non-terminal state: consider future rewards
            max_future_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def save(self, filename):
        """Save the trained Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load(self, filename):
        """Load a trained Q-table from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
            return True
        return False

def train_agent(episodes=100, max_steps=500):
    """
    Train the Q-learning agent.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
    """
    print("🚀 Starting PlantOS Q-Learning Training")
    print("=" * 50)
    
    # Create environment
    env = PlantOSEnv(grid_size=8, num_plants=3, num_obstacles=2)
    
    # Create agent
    agent = SimpleQLearningAgent(
        state_size=env.observation_space.shape,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3  # Start with high exploration
    )
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(obs, training=True)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update agent
            agent.update(obs, action, reward, next_obs, terminated)
            
            # Update episode tracking
            episode_reward += reward
            step_count += 1
            obs = next_obs
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Episode summary
        agent.training_rewards.append(episode_reward)
        agent.episode_lengths.append(step_count)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.training_rewards[-10:])
            avg_length = np.mean(agent.episode_lengths[-10:])
            print(f"Episode {episode:3d}: Reward={episode_reward:6.1f}, "
                  f"Steps={step_count:3d}, Avg Reward={avg_reward:6.1f}, "
                  f"Avg Length={avg_length:5.1f}")
        
        # Decay epsilon for exploration
        if episode % 50 == 0 and episode > 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.9)
            print(f"Epsilon reduced to {agent.epsilon:.3f}")
    
    # Training complete
    print("\n" + "=" * 50)
    print("🎉 Training Complete!")
    print("=" * 50)
    print(f"Episodes trained: {episodes}")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average reward (last 10 episodes): {np.mean(agent.training_rewards[-10:]):.2f}")
    print(f"Average episode length: {np.mean(agent.episode_lengths):.1f}")
    
    # Save trained agent
    agent.save("plantos_qtable.pkl")
    
    env.close()
    return agent

def test_trained_agent(agent, episodes=5, max_steps=200):
    """
    Test the trained agent.
    
    Args:
        agent: Trained Q-learning agent
        episodes: Number of test episodes
        max_steps: Maximum steps per episode
    """
    print("\n🧪 Testing Trained Agent")
    print("=" * 50)
    
    # Create environment
    env = PlantOSEnv(grid_size=8, num_plants=3, num_obstacles=2)
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nTest Episode {episode + 1}:")
        print(f"  Initial thirsty plants: {info['thirsty_plants']}")
        
        for step in range(max_steps):
            # Choose action (no exploration during testing)
            action = agent.choose_action(obs, training=False)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update episode tracking
            episode_reward += reward
            step_count += 1
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Episode summary
        test_rewards.append(episode_reward)
        test_lengths.append(step_count)
        
        print(f"  Final thirsty plants: {info['thirsty_plants']}")
        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Episode length: {step_count}")
    
    # Test summary
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    print(f"Test episodes: {episodes}")
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    print(f"Average test length: {np.mean(test_lengths):.1f}")
    print(f"Best test reward: {np.max(test_rewards):.2f}")
    
    env.close()

def main():
    """Main function to run training and testing."""
    print("🌱 PlantOS Q-Learning Training and Testing")
    print("=" * 50)
    
    try:
        # Train the agent
        agent = train_agent(episodes=50, max_steps=300)
        
        # Test the trained agent
        test_trained_agent(agent, episodes=3, max_steps=200)
        
        print("\n✅ Training and testing completed successfully!")
        print("The trained Q-table has been saved to 'plantos_qtable.pkl'")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have all dependencies installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
