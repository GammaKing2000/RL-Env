import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Tuple, Dict, Any, Optional
import os
import math
from plantos_utils import print_reset_info, print_step_info, print_episode_summary
from plantos_3d_viewer import PlantOS3DViewer


class PlantOSEnv(gym.Env):
    # Observation Channels
    OBSTACLE_CHANNEL = 0
    PLANT_CHANNEL = 1
    THIRST_CHANNEL = 2
    ROVER_CHANNEL = 3

    # LIDAR Entity Types
    ENTITY_EMPTY = 0
    ENTITY_OBSTACLE = 1
    ENTITY_PLANT_HYDRATED = 2
    ENTITY_PLANT_THIRSTY = 3
    """
    PlantOS Environment: A 2D grid-based plant-watering rover simulation.
    
    Modified to be similar to Mars Explorer with LIDAR sensing and 21x21 grid.
    The environment simulates a rover that must water thirsty plants in a grid world
    while avoiding obstacles. It features procedural content generation for random
    maps each episode and multi-channel observations suitable for CNN-based agents.
    """
    
    def __init__(self, grid_size: int = 21, num_plants: int = 8, num_obstacles: int = 50, lidar_range: int = 2, lidar_channels: int = 10, thirsty_plant_prob: float = 0.5):
        """
        Initialize the PlantOS environment.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            num_plants: Number of plants to place on the grid
            num_obstacles: Number of obstacles to place on the grid
            lidar_range: Range of LIDAR sensor in grid cells
            lidar_channels: Number of LIDAR channels for sensing
            thirsty_plant_prob: Probability of a plant being thirsty at reset
        """
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_plants = num_plants
        self.num_obstacles = num_obstacles
        self.lidar_range = lidar_range
        self.lidar_channels = lidar_channels
        self.thirsty_plant_prob = thirsty_plant_prob
        
        # Action space: 0=North, 1=East, 2=South, 3=West, 4=Water
        self.action_space = spaces.Discrete(5)
        
        # Observation space (LIDAR only, with one-hot encoding)
        # 1 (distance) + 4 (one-hot encoded entity types)
        self.observation_space_per_channel = 1 + 4 
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(self.lidar_channels * self.observation_space_per_channel,),
            dtype=np.float32
        )
        
        # Reward constants
        self.R_GOAL = 0.5          # Small incentive for watering a thirsty plant
        self.R_MISTAKE = -100      # Penalty for watering an already hydrated plant
        self.R_INVALID = -10       # Penalty for invalid movement
        self.R_WATER_EMPTY = -5    # Penalty for watering empty space
        self.R_STEP = -0.1         # Small step penalty to encourage efficiency
        self.R_EXPLORATION = 10    # Reward for exploring new cells
        self.R_REVISIT = -0.2      # Penalty for revisiting an already explored cell
        self.R_COMPLETE_EXPLORATION = 200 # Bonus for completing exploration
        
        # Internal state variables
        self.rover_pos = None      # (x, y) coordinates of the rover
        self.plants = {}           # Dict mapping (x, y) -> plant status (True=thirsty, False=hydrated)
        self.obstacles = set()     # Set of (x, y) coordinates with obstacles
        
        # LIDAR and exploration tracking
        self.explored_map = None   # Map of explored cells
        self.ground_truth_map = None  # Complete map with all entities
        self.lidar_indexes = None  # Current LIDAR readings
        
        # Pygame variables for rendering
        self.window = None
        self.clock = None
        self.cell_size = 30  # Increased cell size for better visibility
        self.background_img = None
        self.obstacle_img = None
        self.rover_img = None
        self.plant_thirsty_img = None
        self.plant_hydrated_img = None

        self.viewer_3d = None
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000  # Like Mars Explorer
        self.previous_explored_count = 0
        self.collided_with_wall = False
        self.completion_bonus_given = False
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and generate a new random map.
        
        This method implements Procedural Content Generation (PCG) to create
        a unique map layout for each episode.
        
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.step_count = 0
        self.collided_with_wall = False
        self.completion_bonus_given = False
        
        # Clear all previous entity locations
        self.plants.clear()
        self.obstacles.clear()
        
        # Generate random map layout
        self._generate_map()
        
        # Initialize exploration map
        self._initialize_exploration()

        # Initialize previous_explored_count after the map is created
        self.previous_explored_count = np.sum(self.explored_map > 0)
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()

        # If 3D viewer exists, reset its scene to match the new map
        if self.viewer_3d:
            self.viewer_3d.reset_scene()
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0-4) representing movement or watering
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Initialize reward for this step
        reward = -self.R_STEP  # Base step penalty
        
        # Handle movement actions (0-3)
        if action < 4:
            reward += self._handle_movement(action)
        # Handle watering action (4)
        else:
            reward += self._handle_watering()
        
        # Update LIDAR and exploration
        self._update_lidar()
        reward += self._compute_exploration_reward()
        
        # Get current observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Check if episode should terminate
        terminated = self._is_episode_done(info)
        truncated = self.step_count >= self.max_steps

        # Check for and award completion bonus
        if info['exploration_percentage'] >= 100 and not self.completion_bonus_given:
            reward += self.R_COMPLETE_EXPLORATION
            self.completion_bonus_given = True
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='2d'):
        """
        Render the environment using Pygame (2D) and/or Ursina (3D).

        Args:
            mode (str): The rendering mode. Can be '2d', '3d', or 'human' (both).
        """
        if mode in ['2d', 'human']:
            self._render_2d()
        
        if mode in ['3d', 'human']:
            self._render_3d()

    def _render_3d(self):
        """Handles the Ursina 3D rendering."""
        if self.viewer_3d is None:
            # Initialize the 3D viewer on the first call
            self.viewer_3d = PlantOS3DViewer(grid_size=self.grid_size)
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        # Update the scene with the current state and render a frame
        self.viewer_3d.update_scene(self.plants, self.rover_pos)
        self.viewer_3d.render_step()

    def _render_2d(self):
        """Handles the Pygame 2D rendering."""
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((
                self.grid_size * self.cell_size,
                self.grid_size * self.cell_size
            ))
            pygame.display.set_caption('PlantOS 2D View')
            self.clock = pygame.time.Clock()

            # Load and scale textures
            try:
                script_dir = os.path.dirname(__file__)
                grass_path = os.path.join(script_dir, 'grass_texture.png')
                obstacle_path = os.path.join(script_dir, 'obstacles_texture.png')
                
                self.background_img = pygame.image.load(grass_path).convert()
                self.background_img = pygame.transform.scale(self.background_img, (self.cell_size, self.cell_size))
                
                self.obstacle_img = pygame.image.load(obstacle_path).convert_alpha()
                self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))

                # Load rover and plant sprites
                rover_path = os.path.join(script_dir, 'rover.png')
                plant_thirsty_path = os.path.join(script_dir, 'plant_thirsty.png')
                plant_hydrated_path = os.path.join(script_dir, 'plant_hydrated.png')

                self.rover_img = pygame.image.load(rover_path).convert_alpha()
                self.rover_img = pygame.transform.scale(self.rover_img, (self.cell_size, self.cell_size))

                self.plant_thirsty_img = pygame.image.load(plant_thirsty_path).convert_alpha()
                self.plant_thirsty_img = pygame.transform.scale(self.plant_thirsty_img, (self.cell_size, self.cell_size))

                self.plant_hydrated_img = pygame.image.load(plant_hydrated_path).convert_alpha()
                self.plant_hydrated_img = pygame.transform.scale(self.plant_hydrated_img, (self.cell_size, self.cell_size))

            except pygame.error as e:
                print(f"Warning: Could not load textures. Falling back to solid colors. Error: {e}")
                self.background_img = None
                self.obstacle_img = None
        
        # Draw background
        if self.background_img:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    self.window.blit(self.background_img, (x * self.cell_size, y * self.cell_size))
        else:
            self.window.fill((34, 177, 76))  # Fallback green background
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, (200, 200, 200), 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.grid_size * self.cell_size))
            pygame.draw.line(self.window, (200, 200, 200), 
                           (0, i * self.cell_size), 
                           (self.grid_size * self.cell_size, i * self.cell_size))
        
        # Draw obstacles
        for obs_x, obs_y in self.obstacles:
            if self.obstacle_img:
                self.window.blit(self.obstacle_img, (obs_x * self.cell_size, obs_y * self.cell_size))
            else:
                pygame.draw.rect(self.window, (222, 184, 135),  # Fallback light brown
                               (obs_x * self.cell_size, obs_y * self.cell_size, self.cell_size, self.cell_size))
        
        # Draw plants with sprites based on status
        for (plant_x, plant_y), is_thirsty in self.plants.items():
            if is_thirsty:
                if self.plant_thirsty_img:
                    self.window.blit(self.plant_thirsty_img, (plant_x * self.cell_size, plant_y * self.cell_size))
                else:
                    pygame.draw.rect(self.window, (255, 0, 0), (plant_x * self.cell_size, plant_y * self.cell_size, self.cell_size, self.cell_size))
            else:
                if self.plant_hydrated_img:
                    self.window.blit(self.plant_hydrated_img, (plant_x * self.cell_size, plant_y * self.cell_size))
                else:
                    pygame.draw.rect(self.window, (0, 0, 0), (plant_x * self.cell_size, plant_y * self.cell_size, self.cell_size, self.cell_size))
        
        # Draw rover sprite
        if self.rover_pos:
            rover_x, rover_y = self.rover_pos
            if self.rover_img:
                self.window.blit(self.rover_img, (rover_x * self.cell_size, rover_y * self.cell_size))
            else:
                pygame.draw.rect(self.window, (0, 0, 255), (rover_x * self.cell_size, rover_y * self.cell_size, self.cell_size, self.cell_size))
        
        # Draw LIDAR range visualization (semi-transparent circle)
        if self.rover_pos:
            rover_x, rover_y = self.rover_pos
            center_x = (rover_x + 0.5) * self.cell_size
            center_y = (rover_y + 0.5) * self.cell_size
            radius = self.lidar_range * self.cell_size
            
            # Create a surface for the LIDAR range
            lidar_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(lidar_surface, (0, 255, 255, 30), (radius, radius), radius)
            self.window.blit(lidar_surface, (center_x - radius, center_y - radius))
               
        # Draw LIDAR lines
        # if self.rover_pos and self.lidar_indexes is not None:
        #     rover_x, rover_y = self.rover_pos
        #     start_pos = ((rover_x + 0.5) * self.cell_size, (rover_y + 0.5) * self.cell_size)
        #     for point in self.lidar_indexes:
        #         end_pos = ((point[0] + 0.5) * self.cell_size, (point[1] + 0.5) * self.cell_size)
        #         pygame.draw.line(self.window, (255, 0, 0, 150), start_pos, end_pos, 1)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS

        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
    
    def close(self):
        """Close the environment and clean up Pygame and Ursina resources."""
        if self.viewer_3d:
            self.viewer_3d.close()
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def _generate_map(self):
        """Generate a maze with paths that are 3 cells wide using a randomized DFS."""
        # 1. Start with a grid full of obstacles.
        self.obstacles = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))

        # Define a smaller "meta" grid to generate the maze structure
        # Each cell in the meta-grid corresponds to a 3x3 area in the main grid
        meta_w = (self.grid_size - 1) // 4
        meta_h = (self.grid_size - 1) // 4
        
        # Visited cells in the meta-grid
        visited = np.zeros((meta_w, meta_h), dtype=bool)
        
        # Stack for DFS
        stack = []
        
        # Start carving from a random cell in the meta-grid
        start_x, start_y = random.randint(0, meta_w - 1), random.randint(0, meta_h - 1)
        stack.append((start_x, start_y))
        visited[start_x, start_y] = True
        
        # Carve out the initial 3x3 area
        for i in range(3):
            for j in range(3):
                self.obstacles.discard((start_x * 4 + 1 + i, start_y * 4 + 1 + j))

        # Randomized DFS on the meta-grid
        while stack:
            cx, cy = stack[-1]
            
            # Get unvisited neighbors
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < meta_w and 0 <= ny < meta_h and not visited[nx, ny]:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Choose a random neighbor
                nx, ny = random.choice(neighbors)
                
                # Carve a 3-cell wide path to the neighbor
                for i in range(4):
                    for j in range(3):
                        self.obstacles.discard((min(cx, nx) * 4 + 1 + (i if cx != nx else j), 
                                                min(cy, ny) * 4 + 1 + (i if cy != ny else j)))
                
                # Also carve out the new 3x3 room at the destination
                for i in range(3):
                    for j in range(3):
                        self.obstacles.discard((nx * 4 + 1 + i, ny * 4 + 1 + j))

                visited[nx, ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Randomly place plants
        available_positions = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size)) - self.obstacles

        # Check if there is enough space for plants and the rover
        if len(available_positions) < self.num_plants + 1: # +1 for the rover
            raise ValueError(
                f"Not enough available positions ({len(available_positions)}) to place "
                f"{self.num_plants} plants and 1 rover."
            )
        
        plant_positions = random.sample(list(available_positions), self.num_plants)
        for plant_pos in plant_positions:
            # Randomly assign initial plant status
            is_thirsty = random.random() < self.thirsty_plant_prob
            self.plants[plant_pos] = is_thirsty
        available_positions -= set(plant_positions)
        
        # Randomly place rover
        self.rover_pos = random.choice(list(available_positions))
    
    def _initialize_exploration(self):
        """Initialize the exploration map and ground truth map."""
        # Create ground truth map
        self.ground_truth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Mark obstacles
        for obs_x, obs_y in self.obstacles:
            self.ground_truth_map[obs_x, obs_y] = 1.0
        
        # Mark plants
        for (plant_x, plant_y) in self.plants.keys():
            self.ground_truth_map[plant_x, plant_y] = 0.5
        
        # Initialize explored map (all unexplored initially)
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Mark initial rover position as explored
        self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 0.6
        
        # Initialize LIDAR
        self._update_lidar()
    
    def _update_lidar(self):
        """Update LIDAR readings based on current rover position."""
        if self.rover_pos is None:
            return
        
        rover_x, rover_y = self.rover_pos
        lidar_indexes = []
        
        # Generate LIDAR readings in a circle around the rover
        for i in range(self.lidar_channels):
            angle = (2 * math.pi * i) / self.lidar_channels
            for r in range(1, self.lidar_range + 1):
                dx = int(r * math.cos(angle))
                dy = int(r * math.sin(angle))
                
                check_x = rover_x + dx
                check_y = rover_y + dy
                
                # Check bounds
                if 0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size:
                    # Mark as explored, ensuring even empty space is marked
                    if self.explored_map[check_x, check_y] == 0:
                        value = self.ground_truth_map[check_x, check_y]
                        if value == 0:
                            self.explored_map[check_x, check_y] = 0.1  # Mark explored empty space
                        else:
                            self.explored_map[check_x, check_y] = value
                    
                    # If we hit an obstacle, stop scanning in this direction
                    if (check_x, check_y) in self.obstacles:
                        break
                    
                    lidar_indexes.append([check_x, check_y])
                else:
                    break
        
        self.lidar_indexes = np.array(lidar_indexes) if lidar_indexes else np.empty((0, 2))
    
    def _compute_exploration_reward(self):
        """Compute reward based on newly explored cells."""
        current_explored_count = np.sum(self.explored_map > 0)
        newly_explored = current_explored_count - self.previous_explored_count
        self.previous_explored_count = current_explored_count
        
        return newly_explored * self.R_EXPLORATION
    
    def _handle_movement(self, action: int) -> float:
        """
        Handle movement actions and return the reward for this movement.
        
        Args:
            action: Movement action (0=North, 1=East, 2=South, 3=West)
            
        Returns:
            Reward for this movement action
        """
        # Define movement directions: (dx, dy)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # North, East, South, West
        dx, dy = directions[action]
        
        # Calculate new position
        new_x = self.rover_pos[0] + dx
        new_y = self.rover_pos[1] + dy
        
        # Check if new position is valid (within bounds and not an obstacle)
        if (0 <= new_x < self.grid_size and 
            0 <= new_y < self.grid_size and 
            (new_x, new_y) not in self.obstacles):
            # Valid movement
            is_revisit = self.explored_map[new_x, new_y] > 0
            self.rover_pos = (new_x, new_y)
            self.explored_map[new_x, new_y] = 0.6 # Mark new position as explored
            
            if is_revisit:
                return self.R_REVISIT
            else:
                return 0 # The positive exploration reward is handled separately
        else:
            # Invalid movement (hit wall or obstacle)
            self.collided_with_wall = True
            return self.R_INVALID
    
    def _handle_watering(self) -> float:
        """
        Handle the watering action and return the reward.
        
        Returns:
            Reward for the watering action
        """
        # Check if rover is on a plant
        if self.rover_pos in self.plants:
            plant_status = self.plants[self.rover_pos]
            
            if plant_status:  # Plant is thirsty
                # Water the plant successfully
                self.plants[self.rover_pos] = False
                return self.R_GOAL
            else:  # Plant is already hydrated
                return self.R_MISTAKE
        else:
            # Watering empty space
            return self.R_WATER_EMPTY
    
    def _is_episode_done(self, info: Dict[str, Any]) -> bool:
        """
        Check if the episode should terminate.
        
        Returns:
            True if all thirsty plants have been watered, False otherwise
        """
        # Episode is done when all plants are hydrated (no thirsty plants)
        fully_explored = info['exploration_percentage'] >= 100
        return self.collided_with_wall or fully_explored
    
    def _get_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array."""
        return self._get_lidar_obs()

    def _get_lidar_obs(self) -> np.ndarray:
        """
        Generate the LIDAR-based observation array with one-hot encoding for entity types.
        
        Returns:
            1D NumPy array with LIDAR data.
        """
        obs = np.zeros(self.lidar_channels * self.observation_space_per_channel, dtype=np.float32)
        rover_x, rover_y = self.rover_pos

        for i in range(self.lidar_channels):
            angle = (2 * math.pi * i) / self.lidar_channels
            
            distance = self.lidar_range
            entity_type = self.ENTITY_EMPTY

            # Ray-march out to the LIDAR range
            for r in range(1, self.lidar_range + 1):
                dx = int(r * math.cos(angle))
                dy = int(r * math.sin(angle))
                
                check_x = rover_x + dx
                check_y = rover_y + dy
                
                # Check if we hit something
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE # Wall is like an obstacle
                    break
                
                pos = (check_x, check_y)
                if pos in self.obstacles:
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE
                    break
                elif pos in self.plants:
                    distance = r
                    entity_type = self.ENTITY_PLANT_THIRSTY if self.plants[pos] else self.ENTITY_PLANT_HYDRATED
                    break
            
            # Populate the observation vector for this channel
            start_index = i * self.observation_space_per_channel
            # 1. Normalized distance
            obs[start_index] = distance / self.lidar_range
            
            # 2. One-hot encoded entity type
            one_hot_type = np.zeros(4, dtype=np.float32)
            one_hot_type[entity_type] = 1.0
            obs[start_index + 1 : start_index + 5] = one_hot_type
            
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the environment.
        
        Returns:
            Dictionary containing environment information
        """
        thirsty_count = sum(self.plants.values())
        hydrated_count = len(self.plants) - thirsty_count
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.grid_size * self.grid_size
        
        return {
            'rover_position': self.rover_pos,
            'thirsty_plants': thirsty_count,
            'hydrated_plants': hydrated_count,
            'total_plants': len(self.plants),
            'step_count': self.step_count,
            'explored_cells': explored_cells,
            'total_cells': total_cells,
            'exploration_percentage': (explored_cells / total_cells) * 100,
            'lidar_range': self.lidar_range,
            'lidar_channels': self.lidar_channels,
            'collided_with_wall': self.collided_with_wall
        }


# Register the environment with Gymnasium
gym.register(
    id='PlantOS-v0',
    entry_point='plantos_env:PlantOSEnv',
)

if __name__ == "__main__":
    """
    Example usage and testing of the PlantOS environment.
    
    This block demonstrates how to use the environment and runs a simple
    test with random actions.
    """
    # Create the environment with Mars Explorer-like parameters
    env = PlantOSEnv(grid_size=21, num_plants=50, num_obstacles=50, lidar_range=2, lidar_channels=32)
    
    try:
        # Reset the environment to get initial state
        obs, info = env.reset()
        print_reset_info(info, initial=True)
        
        # Run the environment for a fixed number of steps
        for step in range(1000):  # Run for 1000 steps
            # Take a random action
            action = env.action_space.sample()  # Take a random action
            
            # Execute the step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Print step information
            if step % 50 == 0:
                print_step_info(step, action, reward, info)
            
            # Check if episode is done
            if terminated or truncated:
                print_episode_summary(step, info)
                obs, info = env.reset()
                print_reset_info(info, initial=False)
        
    except KeyboardInterrupt:
        print("\nEnvironment interrupted by user.")
    
    finally:
        # Clean up
        env.close()
        print("Environment closed.")
