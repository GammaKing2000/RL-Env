import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
import math
import copy # For deep copying environments

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plantos_env_mcts import PlantOSEnvMCTS # Use the MCTS-compatible environment

# --- 1. Neural Network (Policy and Value Head) ---
class MCTSNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_size, hidden_size=256):
        super(MCTSNetwork, self).__init__()
        
        input_size = observation_space_shape[0]

        # LSTM layer to process sequential observations and maintain memory
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Policy head: outputs probabilities for each action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size)
        )
        
        # Value head: outputs a single value for the state
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, hidden_state=None):
        # x is expected to be (batch_size, sequence_length, input_size)
        # For a single step, sequence_length=1
        if x.dim() == 2: # If input is (batch_size, input_size), add sequence_length dim
            x = x.unsqueeze(1)

        lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Use the last output of the LSTM for policy and value heads
        last_lstm_out = lstm_out[:, -1, :]
        
        policy_logits = self.policy_head(last_lstm_out)
        value = torch.tanh(self.value_head(last_lstm_out)) # Value typically between -1 and 1
        
        return policy_logits, value, hidden_state

# --- 2. Modified MCTS Algorithm ---
class MCTSNode:
    def __init__(self, env_state_dict, observation, action_space_size, parent=None, action=None, reward=0.0, done=False, hidden_state=None):
        self.env_state_dict = env_state_dict # Full environment state for restoring
        self.observation = observation # Observation from this state
        self.parent = parent
        self.action_taken = action
        self.reward = reward # Reward received to reach this state
        self.done = done # Whether this state is terminal
        self.hidden_state = hidden_state # LSTM hidden state for this node

        self.children = {}
        self.visits = 0
        self.value_sum = 0.0 # Sum of values from simulations passing through this node
        self.policy_prior = np.zeros(action_space_size) # Policy probabilities from NN
        self.untried_actions = list(range(action_space_size)) # Actions not yet expanded

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def is_terminal(self):
        return self.done

    def ucb_score(self, c_param):
        if self.visits == 0:
            # Return a very high value to prioritize unvisited nodes
            return float('inf')
        
        # PUCT formula for selection (more appropriate for MCTS with NN)
        # Q(s,a) + c_param * P(s,a) * sqrt(sum(visits)) / (1 + N(s,a))
        
        # For now, let's use a UCB-like approach that considers policy prior
        # This is still a heuristic, a full PUCT requires more careful implementation.
        
        total_visits_parent = self.parent.visits # N(s) for parent
        if total_visits_parent == 0: # Should not happen if parent is root and has been visited
            total_visits_parent = 1 # Avoid division by zero

        exploitation_term = self.value_sum / self.visits
        # The policy_prior for this node is the prior for the action that led to it.
        # We need the policy_prior for the action *from* the parent to this child.
        # This is stored in node.parent.policy_prior[node.action_taken]
        exploration_term = c_param * self.parent.policy_prior[self.action_taken] * math.sqrt(total_visits_parent) / (1 + self.visits)
        return exploitation_term + exploration_term

class MCTS:
    def __init__(self, neural_net, action_space_size, env_prototype, c_param=1.0, num_simulations=100):
        self.neural_net = neural_net
        self.action_space_size = action_space_size
        self.env_prototype = env_prototype # Uninitialized PlantOSEnvMCTS class
        self.c_param = c_param
        self.num_simulations = num_simulations
        self.root = None

    def run_simulations(self, env_initial_state_dict, initial_observation, initial_hidden_state):
        # The root node stores the initial environment state and observation
        self.root = MCTSNode(env_initial_state_dict, initial_observation, self.action_space_size, hidden_state=initial_hidden_state)
        
        # Predict policy and value for the root node using the neural network
        obs_tensor = torch.tensor(initial_observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value_pred, _ = self.neural_net(obs_tensor, initial_hidden_state)
            self.root.policy_prior = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        for _ in range(self.num_simulations):
            node = self.root
            current_hidden_state = initial_hidden_state # Hidden state for the current path
            
            # Create a temporary environment for this simulation path
            sim_env = self.env_prototype() # Instantiate a new environment
            sim_env.set_state(self.root.env_state_dict) # Restore to root state

            path = [node]
            while node.is_fully_expanded() and not node.is_terminal():
                node, action_taken = self._select_child(node)
                path.append(node)
                
                # Simulate environment step to get to the child's state and update hidden state
                next_obs, reward, terminated, truncated, _ = sim_env.step(action_taken)
                
                # Update hidden state for the path by re-running NN on node's observation
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, _, current_hidden_state = self.neural_net(obs_tensor, current_hidden_state)

                # Update node's state, observation, reward, done status
                node.env_state_dict = sim_env.get_state()
                node.observation = next_obs
                node.reward = reward
                node.done = terminated or truncated
                node.hidden_state = current_hidden_state # Store the updated hidden state

            # 2. Expansion: Expand a new node
            if not node.is_terminal():
                node, value_from_nn = self._expand_node(node, sim_env, current_hidden_state)
                path.append(node)
            else:
                # If terminal node, its value is its reward (or 0 if no reward yet)
                value_from_nn = node.reward # Assuming reward is stored in terminal node

            # 3. Backpropagation: Update values and visit counts
            self._backpropagate(path, value_from_nn)

    def _select_child(self, node):
        best_child = None
        best_score = -float('inf')
        action_taken = -1
        
        total_visits_parent = node.visits # N(s) for parent
        if total_visits_parent == 0: 
            total_visits_parent = 1 # Avoid division by zero

        for action in node.children.keys(): # Iterate over already expanded children
            child = node.children[action]
            if child.visits == 0:
                return child, action # Prioritize unvisited children

            q_value = child.value_sum / child.visits
            # Exploration term using policy prior and parent visits
            exploration_term = self.c_param * node.policy_prior[action] * math.sqrt(total_visits_parent) / (1 + child.visits)
            score = q_value + exploration_term

            if score > best_score:
                best_score = score
                best_child = child
                action_taken = action
        
        # If no children, or all children are fully expanded, and there are untried actions
        if best_child is None and node.untried_actions: # This means we need to expand
            action_to_expand = node.untried_actions[0] # Pick the first untried action
            # We don't return a child here, as it will be expanded in _expand_node
            return None, action_to_expand # Indicate that expansion is needed

        return best_child, action_taken

    def _expand_node(self, node, sim_env, current_hidden_state):
        # Take an untried action
        action = node.untried_actions.pop(0)
        
        # Restore sim_env to parent node's state
        sim_env.set_state(node.env_state_dict)

        # Simulate taking the action to get the next state, observation, reward, and terminal status
        next_obs, reward, terminated, truncated, _ = sim_env.step(action)
        next_env_state_dict = sim_env.get_state()
        next_done = terminated or truncated

        # Predict policy and value for the new state using the neural network
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0) # Add batch dim
        with torch.no_grad():
            policy_logits, value_pred, next_hidden_state = self.neural_net(next_obs_tensor, current_hidden_state)
            policy_prior = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        new_node = MCTSNode(next_env_state_dict, next_obs, self.action_space_size, parent=node, action=action, 
                            reward=reward, done=next_done, hidden_state=next_hidden_state)
        new_node.policy_prior = policy_prior
        node.children[action] = new_node
        
        # The value returned here is the value prediction for the *newly expanded node's* state.
        return new_node, value_pred.item()

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            # In AlphaZero, value is flipped for alternating players. Here, single player.
            # value = -value # If alternating players

    def get_action(self, initial_env_state_dict, initial_observation, initial_hidden_state, temperature=1.0):
        # Run MCTS simulations from the current state
        self.run_simulations(initial_env_state_dict, initial_observation, initial_hidden_state)
        
        # Get visit counts from the root's children
        visit_counts = np.zeros(self.action_space_size)
        for action, child in self.root.children.items():
            visit_counts[action] = child.visits
        
        # Apply temperature for exploration during self-play
        if temperature == 0: # Deterministic choice
            action_probs = np.zeros(self.action_space_size)
            action_probs[np.argmax(visit_counts)] = 1.0
        else:
            # Softmax over visit counts to get policy
            visit_counts_temp = visit_counts**(1/temperature)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)

        # Choose action based on the policy
        action = np.random.choice(self.action_space_size, p=action_probs)
        
        return action, action_probs

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, mcts_policy, outcome):
        self.buffer.append((state, mcts_policy, outcome))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, mcts_policies, outcomes = zip(*batch)
        return np.array(states), np.array(mcts_policies), np.array(outcomes)

    def __len__(self):
        return len(self.buffer)

# --- 3. Full Self-Play Training Loop ---
def train_mcts_with_nn(
    neural_net,
    mcts_agent_prototype, # MCTS class, not instance
    env_prototype,
    optimizer,
    policy_criterion,
    value_criterion,
    replay_buffer,
    num_episodes=1000,
    num_simulations_per_move=50,
    batch_size=32,
    epochs_per_update=1,
    temperature_schedule=lambda i: 1.0 # Default to constant temperature
):
    neural_net.train() # Set NN to training mode
    
    for episode in range(num_episodes):
        env = env_prototype() # Create a new environment for the episode
        current_obs, info = env.reset()
        done = False
        episode_history = [] # Store (obs, mcts_policy, hidden_state) for this episode
        episode_rewards = 0
        step_count = 0

        # Initial LSTM hidden state for the actual environment interaction
        hidden_size = neural_net.lstm.hidden_size
        current_hidden_state = (torch.zeros(1, 1, hidden_size), 
                                torch.zeros(1, 1, hidden_size))

        # Create a new MCTS agent for this episode (or reset existing one)
        mcts_agent = mcts_agent_prototype(neural_net, env.action_space.n, env_prototype, num_simulations=num_simulations_per_move)

        while not done:
            step_count += 1
            # Get current environment state dictionary for MCTS simulations
            current_env_state_dict = env.get_state()

            # Get action and MCTS policy from MCTS search
            temperature = temperature_schedule(episode) # Apply temperature schedule
            action, mcts_policy = mcts_agent.get_action(current_env_state_dict, current_obs, current_hidden_state, temperature)
            
            # Store data for training
            episode_history.append((current_obs, mcts_policy, current_hidden_state)) # Store hidden state for later
            
            # Take the chosen action in the actual environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            
            # Update the LSTM hidden state for the actual environment interaction
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, current_hidden_state = neural_net(next_obs_tensor, current_hidden_state)

            current_obs = next_obs

            if step_count >= env.max_steps: # Prevent infinite loops
                done = True
        
        # Calculate outcome for the episode (e.g., total reward)
        # Normalize outcome to be between -1 and 1 for tanh value head
        max_possible_reward = (env.num_plants * env.R_GOAL) + env.R_COMPLETE_EXPLORATION
        outcome = episode_rewards / max_possible_reward # Simple normalization
        outcome = max(-1.0, min(1.0, outcome)) # Clip to [-1, 1]

        # Add episode data to replay buffer
        for obs, policy, hidden_state_at_step in episode_history:
            replay_buffer.add(obs, policy, outcome) # All states in episode get same outcome

        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_rewards:.2f} | Buffer Size: {len(replay_buffer)}")

        # Train neural network if buffer is large enough
        if len(replay_buffer) >= batch_size:
            for _ in range(epochs_per_update):
                states, policies, outcomes = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_tensor = torch.tensor(states, dtype=torch.float32)
                policies_tensor = torch.tensor(policies, dtype=torch.float32)
                outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1) # Add dim for MSELoss

                # Forward pass
                predicted_policy_logits, predicted_value, _ = neural_net(states_tensor) # No hidden state for batch training

                # Loss calculation
                policy_loss = policy_criterion(predicted_policy_logits, policies_tensor) # CrossEntropy expects logits
                value_loss = value_criterion(predicted_value, outcomes_tensor)
                
                # Total loss (add L2 regularization if desired)
                loss = policy_loss + value_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"  NN Trained. Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    neural_net.eval() # Set NN back to evaluation mode


if __name__ == '__main__':
    # Environment setup
    env_kwargs = {
        'grid_size': 21,
        'num_plants': 10,
        'num_obstacles': 12,
        'lidar_range': 4,
        'lidar_channels': 12
    }
    # Create a function to instantiate the environment for MCTS simulations
    def make_env_mcts():
        return PlantOSEnvMCTS(**env_kwargs)

    # Initialize Neural Network
    dummy_env = make_env_mcts()
    obs_shape = dummy_env.observation_space.shape
    action_size = dummy_env.action_space.n
    dummy_env.close()

    neural_net = MCTSNetwork(obs_shape, action_size)
    # Load pre-trained weights if available, otherwise it will start with random weights
    # neural_net.load_state_dict(torch.load("path/to/pretrained_model.pth"))
    # neural_net.eval() # Set to evaluation mode for MCTS search

    # MCTS Hyperparameters
    mcts_c_param = 1.0
    mcts_num_simulations = 50

    # Training Hyperparameters
    learning_rate = 0.001
    replay_buffer_capacity = 50000
    batch_size = 32
    num_training_episodes = 100 # Reduced for initial testing
    epochs_per_update = 1 # How many times to train NN per buffer update
    # Temperature schedule for exploration during self-play
    # Higher temperature means more exploration (more random action choice from MCTS policy)
    # AlphaZero typically decays temperature over time.
    def temperature_schedule(episode_idx):
        if episode_idx < num_training_episodes * 0.5:
            return 1.0
        else:
            return 0.1

    # Optimizer and Loss Functions
    optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)
    policy_criterion = nn.CrossEntropyLoss() # Expects logits and target probabilities
    value_criterion = nn.MSELoss()

    # Replay Buffer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # --- Start Full MCTS with Neural Network Training ---
    print("\n--- Starting Full MCTS with Neural Network Training ---")
    train_mcts_with_nn(
        neural_net,
        MCTS, # Pass the MCTS class itself
        make_env_mcts,
        optimizer,
        policy_criterion,
        value_criterion,
        replay_buffer,
        num_episodes=num_training_episodes,
        num_simulations_per_move=mcts_num_simulations,
        batch_size=batch_size,
        epochs_per_update=epochs_per_update,
        temperature_schedule=temperature_schedule
    )
    print("MCTS with Neural Network Training Finished.")

    # Save the trained neural network
    model_save_path = os.path.join("MCTS_Training", "mcts_nn_model.pth")
    torch.save(neural_net.state_dict(), model_save_path)
    print(f"Neural network model saved to {model_save_path}")

    # --- Demonstrate MCTS-guided action selection for one episode with trained NN ---
    print("\n--- Demonstrating MCTS-guided action selection with trained NN ---")
    neural_net.eval() # Set to evaluation mode
    env = make_env_mcts()
    current_obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    
    hidden_size = neural_net.lstm.hidden_size
    current_hidden_state = (torch.zeros(1, 1, hidden_size), 
                            torch.zeros(1, 1, hidden_size))

    mcts_agent_eval = MCTS(neural_net, action_size, make_env_mcts, c_param=mcts_c_param, num_simulations=mcts_num_simulations)

    while not done:
        step_count += 1
        # Get current environment state dictionary for MCTS simulations
        current_env_state_dict = env.get_state()

        # MCTS search to get the best action (temperature=0 for deterministic eval)
        action, _ = mcts_agent_eval.get_action(current_env_state_dict, current_obs, current_hidden_state, temperature=0.0)
        
        # Take the chosen action in the actual environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Update the LSTM hidden state for the actual environment interaction
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, current_hidden_state = neural_net(next_obs_tensor, current_hidden_state)

        current_obs = next_obs

        if step_count >= env.max_steps: # Prevent infinite loops in demo
            done = True

    print(f"\nEvaluation Episode finished. Total Reward: {episode_reward}")
    env.close()
