# GROW-R: A Reinforcement Learning Environment for Autonomous Plant Care

GROW-R is a custom reinforcement learning environment built with Gymnasium that simulates a robotic agent, GROW-R (Grid-based Rover for Oasis and Water-Replenishment), tasked with exploring a 2D grid world and watering thirsty plants. This environment is designed for training and evaluating agents on a complex exploration and resource management task using LIDAR-based observations.

## Features

- **LIDAR-Based Navigation**: The agent relies on 1D LIDAR scans to perceive its surroundings, detecting obstacles, plants, and their hydration status.
- **Procedural Content Generation**: Each episode generates a new, randomized map with scattered obstacles and plants, ensuring the agent learns a generalizable exploration strategy.
- **Exploration-Focused Rewards**: The reward system is heavily weighted to incentivize the discovery of new areas, with secondary rewards for efficient plant watering.
- **Real-time 2D and 3D Visualization**: Monitor the agent's performance through a simple 2D Pygame view or an immersive 3D view built with the Ursina engine.
- **Gradio-Based UI**: An intuitive web interface powered by Gradio allows for live simulation, model selection, and environment customization.
- **Compatibility**: Fully compatible with the Gymnasium API and popular RL libraries like Stable Baselines3.

## Environment Overview

### State Representation
The agent's observation is a 1D vector composed of three main components:
1.  **LIDAR Scans**: A 360-degree LIDAR scan with 16 channels. Each channel provides:
    *   **Normalized Distance**: The distance to the detected object (0.0 to 1.0).
    *   **One-Hot Encoded Type**: A 4-element vector representing the object's type:
        *   `[1, 0, 0, 0]`: Empty Space
        *   `[0, 1, 0, 0]`: Obstacle / Wall
        *   `[0, 0, 1, 0]`: Hydrated Plant
        *   `[0, 0, 0, 1]`: Thirsty Plant
2.  **Rover's Position**: The agent's normalized `(x, y)` coordinates.
3.  **Local Visit Map**: A 5x5 grid centered on the agent, indicating the visit count for each cell to help the agent avoid redundant exploration.

The total observation space is a flat vector of shape **(107,)**.

### Actions
The agent can perform one of five discrete actions:
- **0**: Move North
- **1**: Move East
- **2**: Move South
- **3**: Move West
- **4**: Water (attempts to water a plant at the current location)

### Rewards
The reward function is designed to encourage efficient exploration and plant care:
- **+50**: Large bonus for exploring 100% of the map.
- **+20**: For watering a thirsty plant.
- **+10**: For each new cell discovered.
- **-0.1**: Small penalty for each step taken (to encourage efficiency).
- **-1**: Penalty for revisiting an already explored cell.
- **-5**: Penalty for attempting to water an empty space or colliding with an obstacle.
- **-10**: Penalty for watering an already hydrated plant.

### Episode Termination
An episode concludes under the following conditions:
- The agent explores 100% of the map.
- The maximum step limit of 1000 is reached.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RL-Env
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training an Agent
You can train an agent using one of the provided training scripts. For example, to train an A2C agent, run:
```bash
python A2C_training.py --mode train --timesteps 100000 --envs 8
```
This will train an A2C model for 100,000 timesteps using 8 parallel environments. The trained models and logs will be saved in the `a2c_training` directory.

### Running the Gradio Web Interface
To launch the interactive Gradio UI for live simulation and visualization:
```bash
python gradio-app/gradioUI.py
```
This will start a web server. Open the provided URL in your browser to access the interface, where you can:
-   Select a pre-trained model (DQN, PPO, or A2C).
-   Specify the path to the model file.
-   Customize environment parameters like grid size, number of plants, and obstacles.
-   Watch the simulation in a live 2D view and a separate 3D window.

### Basic Environment Interaction
You can also interact with the environment directly in a Python script:
```python
import gymnasium as gym
from plantos_env import PlantOSEnv

# Create the environment
env = PlantOSEnv()

# Reset the environment
obs, info = env.reset()

for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Render the 2D view
    env.render()

    if terminated or truncated:
        print("Episode finished!")
        obs, info = env.reset()

env.close()
```

## Project Structure
-   `plantos_env.py`: Defines the core Gymnasium environment for GROW-R.
-   `A2C_training.py` / `mcts_custom_trainer.py`: Scripts for training reinforcement learning agents.
-   `gradio-app/gradioUI.py`: The Gradio-based web interface for live simulation.
-   `plantos_3d_viewer.py`: The 3D visualization tool using the Ursina engine.
-   `assets/`: Contains images and 3D models used for rendering.
-   `requirements.txt`: A list of all necessary Python packages.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to:
-   Improve the environment dynamics or reward structure.
-   Add new features or agent capabilities.
-   Enhance the visualization tools.
-   Add more training algorithms or pre-trained models.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.