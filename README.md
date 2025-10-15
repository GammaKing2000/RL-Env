# PlantOS Environment

A custom reinforcement learning environment built with Gymnasium that simulates a plant-watering rover in a 2D grid world. This environment is designed for training reinforcement learning agents to efficiently water thirsty plants while avoiding obstacles.

## Features

- **Grid-based 2D World**: Configurable grid size with customizable number of plants and obstacles
- **Procedural Content Generation**: New random map layout generated for each episode
- **Multi-channel Observations**: 4-channel observation space suitable for CNN-based agents
- **Dual Visualization Modes**: 2D top-down view and 3D perspective rendering
- **Real-time Visualization**: Pygame-based rendering with color-coded elements
- **Gymnasium API Compatible**: Fully compatible with the Gymnasium reinforcement learning library
- **Configurable Rewards**: Structured reward system to encourage efficient plant watering

## Environment Overview

### State Representation
The environment uses a 4-channel observation space:
- **Channel 0**: Obstacles (binary - 1 for obstacle, 0 for empty)
- **Channel 1**: Plant locations (binary - 1 for plant, 0 for empty)
- **Channel 2**: Plant thirst status (binary - 1 for thirsty plant, 0 for hydrated/empty)
- **Channel 3**: Rover position (one-hot - 1 at rover location, 0 elsewhere)

### Actions
- **0**: Move North
- **1**: Move East  
- **2**: Move South
- **3**: Move West
- **4**: Water (attempt to water plant at current location)

### Rewards
- **+100**: Successfully watering a thirsty plant
- **-100**: Watering an already hydrated plant
- **-10**: Invalid movement (hitting wall or obstacle)
- **-5**: Watering empty space
- **-0.1**: Small step penalty to encourage efficiency

### Episode Termination
An episode ends when all thirsty plants have been successfully watered, or when the maximum step limit (400) is reached.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PlantRover
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import gymnasium as gym
from plantos_env import PlantOSEnv

# Create the environment
env = PlantOSEnv(grid_size=10, num_plants=5, num_obstacles=3)

# Reset to get initial state
obs, info = env.reset()

# Take actions
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

# Clean up
env.close()
```

### Customizing the Environment

```python
# Create environment with custom parameters
env = PlantOSEnv(
    grid_size=15,        # 15x15 grid
    num_plants=8,        # 8 plants to water
    num_obstacles=5      # 5 obstacles to avoid
)
```

### Training a Reinforcement Learning Agent

The environment is designed to work with popular RL libraries like Stable Baselines3:

```python
from stable_baselines3 import PPO
from plantos_env import PlantOSEnv

# Create environment
env = PlantOSEnv()

# Create and train agent
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test the trained agent
obs, info = env.reset()
for step in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Visualization

The environment renders in real-time using Pygame:
- **Blue Square**: Rover position
- **Brown Square**: Thirsty plant (needs watering)
- **Green Square**: Hydrated plant (already watered)
- **Grey Square**: Obstacle/wall
- **Grid Lines**: Visual grid boundaries

## Visualization Modes

### 2D Top-down View
- Classic grid-based visualization
- Perfect for debugging and understanding agent behavior
- Color-coded elements for clear state representation
- Efficient rendering for fast training

### 3D Perspective View
- Immersive 3D visualization using OpenGL
- Realistic plant and rover models
- Dynamic lighting and shadows
- Camera controls for different viewing angles
- Toggle between views using the 'V' key

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 10 | Size of the grid (grid_size × grid_size) |
| `num_plants` | 5 | Number of plants to place on the grid |
| `num_obstacles` | 3 | Number of obstacles to place on the grid |

## Technical Details

### Observation Space
- **Shape**: (4, grid_size, grid_size)
- **Data Type**: np.uint8
- **Value Range**: [0, 1]

### Action Space
- **Type**: Discrete(5)
- **Actions**: 0-4 (North, East, South, West, Water)

### State Management
- Plants are stored as a dictionary mapping (x, y) coordinates to boolean status
- Obstacles are stored as a set of (x, y) coordinates
- Rover position is stored as a tuple (x, y)
- All positions are validated to prevent overlapping entities

## Example Training Loop

```python
from plantos_env import PlantOSEnv
import numpy as np

# Create environment
env = PlantOSEnv()

# Training loop
episode_rewards = []
for episode in range(100):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(1000):
        # Your agent's action selection logic here
        action = env.action_space.sample()  # Replace with agent action
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode}: Total Reward = {episode_reward:.2f}")

env.close()
print(f"Average reward over 100 episodes: {np.mean(episode_rewards):.2f}")
```

## Contributing

Feel free to contribute to this project by:
- Adding new features or modifications
- Improving the reward structure
- Adding new observation channels
- Optimizing performance
- Adding tests and documentation

## License

This project is open source and available under the MIT License.
