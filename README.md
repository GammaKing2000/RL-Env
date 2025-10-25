# PlantOS Environment

A custom reinforcement learning environment built with Gymnasium that simulates a rover exploring a 2D grid world. This environment is designed to train agents on an **exploration task** using LIDAR-based observations. The primary goal is to explore the entire map, with a minor secondary objective of watering thirsty plants.

## Features

- **LIDAR-Based Exploration**: Agents must rely on 1D LIDAR scans to navigate and explore the environment.
- **Procedural Content Generation**: A new random maze-like map is generated for each episode.
- **Exploration-Focused Rewards**: The reward system is heavily weighted to incentivize discovering new areas of the map.
- **Real-time 3D Visualization**: An immersive 3D view built with Ursina allows for real-time monitoring of the agent.
- **Gymnasium API Compatible**: Fully compatible with the Gymnasium reinforcement learning library.

## Environment Overview

### State Representation
The environment uses a 1D LIDAR-based observation space. The agent receives a flat vector of sensor readings from beams cast in a 360-degree circle. For each of the **10 LIDAR channels**, the observation contains 5 values:
- **1. Normalized Distance**: The distance to the detected object, scaled between 0.0 and 1.0.
- **2-5. One-Hot Encoded Type**: A 4-element one-hot vector representing the object's type:
    - `[1, 0, 0, 0]`: Empty Space
    - `[0, 1, 0, 0]`: Obstacle / Wall
    - `[0, 0, 1, 0]`: Hydrated Plant
    - `[0, 0, 0, 1]`: Thirsty Plant

The total observation space is a vector of shape **(50,)**.

### Actions
- **0**: Move North
- **1**: Move East  
- **2**: Move South
- **3**: Move West
- **4**: Water (attempt to water plant at current location)

### Rewards
The reward function is designed to primarily encourage exploration:
- **+200**: Large one-time bonus for successfully exploring 100% of the map.
- **+10**: For each new cell revealed by the LIDAR scan on a step.
- **+0.5**: Small bonus for correctly watering a thirsty plant.
- **-0.1**: Small penalty for every step taken (to encourage efficiency).
- **-0.2**: Small penalty for moving into an already-explored cell.
- **-5**: Penalty for attempting to water an empty space.
- **-10**: Penalty for colliding with a wall or obstacle (ends the episode).
- **-100**: Large penalty for watering an already hydrated plant.

### Episode Termination
An episode ends if any of the following occur:
- The agent collides with a wall or an obstacle.
- The agent successfully explores 100% of the map.
- The maximum step limit (1000 steps) is reached.

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

# Create the environment with default parameters
env = PlantOSEnv()

# Reset to get initial state
obs, info = env.reset()

# Take actions
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the 3D environment
    env.render(mode='3d')
    
    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

# Clean up
env.close()
```

### Customizing the Environment

```python
# Create environment with custom parameters
env = PlantOSEnv(
    grid_size=15,        # 15x15 grid
    num_plants=10,       # 10 plants to water
    num_obstacles=20,    # 20 obstacles to avoid
    lidar_range=4,
    lidar_channels=16
)
```

### Training a Reinforcement Learning Agent

The environment is designed to work with popular RL libraries like Stable Baselines3. Note that a simple `MlpPolicy` is stateless. For a complex exploration task, a recurrent policy (e.g., `MlpLstmPolicy`) is highly recommended to provide the agent with memory.

```python
from stable_baselines3 import PPO
from plantos_env import PlantOSEnv

# Create environment
env = PlantOSEnv()

# Create and train agent using a policy for 1D vector observations
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test the trained agent
obs, info = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(mode='3d')
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Visualization

The environment can be rendered in two ways:
- **2D View (`mode='2d'`)**: A simplified, top-down view using Pygame that is fast and useful for debugging. It shows color-coded squares for the rover, plants, and obstacles.
- **3D View (`mode='3d'` or `mode='human'`)**: An immersive 3D visualization using the Ursina engine, showing realistic models and a chase-cam perspective that follows the agent.

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 21 | Size of the grid (grid_size × grid_size) |
| `num_plants` | 8 | Number of plants to place on the grid |
| `num_obstacles` | 50 | Number of obstacles to place on the grid |
| `lidar_range` | 2 | Range of the LIDAR sensor in grid cells |
| `lidar_channels` | 10 | Number of beams cast by the LIDAR sensor |

## Technical Details

### Observation Space
- **Type**: 1D `Box` vector.
- **Shape**: `(50,)` by default (`lidar_channels` * 5).
- **Data Type**: `np.float32`
- **Value Range**: `[0.0, 1.0]`
- **Structure**: A concatenation of 5 values for each of the 10 LIDAR channels (see State Representation section above).

### Action Space
- **Type**: `Discrete(5)`
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
