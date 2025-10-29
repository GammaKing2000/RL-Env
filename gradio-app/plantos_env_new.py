import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Tuple, Dict, Any, Optional
import os
import math
from plantos_utils import print_reset_info, print_step_info, print_episode_summary
from plantos_3d_viewer_new import PlantOS3DViewer

class PlantOSEnvNew(gym.Env):
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
    
    def __init__(self, grid_size: int = 21, num_plants: int = 8, num_obstacles: int = 50, 
                 lidar_range: int = 2, lidar_channels: int = 10, thirsty_plant_prob: float = 0.7,
                 observation_mode: str = "grid", render_mode: Optional[str] = None,
                 map_generation_algo: str = 'original'):
        """
        Initialize the PlantOS environment.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            num_plants: Number of plants to place on the grid
            num_obstacles: Number of obstacles to place on the grid
            lidar_range: Range of LIDAR sensor in grid cells
            lidar_channels: Number of LIDAR channels for sensing
            thirsty_plant_prob: Probability of a plant being thirsty at reset
            map_generation_algo: Algorithm for map generation ('original', 'maze', 'rooms', 'border')
        """
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_plants = num_plants
        self.num_obstacles = num_obstacles
        self.lidar_range = lidar_range
        self.lidar_channels = lidar_channels
        self.thirsty_plant_prob = thirsty_plant_prob
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.map_generation_algo = map_generation_algo
        
        # Action space: 0=North, 1=East, 2=South, 3=West, 4=Water
        self.action_space = spaces.Discrete(5)
        
        # Observation space (LIDAR only, with one-hot encoding)
        # 1 (distance) + 4 (one-hot encoded entity types)
        self.observation_space_per_channel = 1 + 4
        
        # Local visit map parameters
        self.visit_map_size = 5  # 5x5 grid around rover
        self.visit_map_cells = self.visit_map_size * self.visit_map_size  # 25 cells
        
        # Observation space components:
        # - LIDAR: lidar_channels * 5 values
        # - Position: 2 values (x, y)
        # - Local visit map: 25 values (5x5 grid)
        total_obs_size = (self.lidar_channels * self.observation_space_per_channel + 
                         2 +  # position
                         self.visit_map_cells)  # local visit counts
        
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Rewards for A2C - avg_exploration ~ 87%, 10mil steps, with curriculum learning, 512 n_env, 20 minutes
        self.R_GOAL = 200                   # watering plants
        self.R_MISTAKE = -20                # watering hydrated plant
        self.R_INVALID = -11                # invalid move (collision or out of bounds)
        self.R_WATER_EMPTY = -20            # watering empty space
        self.R_STEP = -0.1                  # small step penalty to encourage efficiency
        self.R_EXPLORATION = 10             # Bonus for exploring new cell
        self.R_REVISIT = -3                 # Small penalty for revisiting explored cell
        self.R_COMPLETE_EXPLORATION = 100   # Bonus for fully exploring the map

        # Rewards for DQN - avg_exploration ~ 97%, 10mil steps, with curriculum learning, 512 n_env, 9 minutes
        # self.R_GOAL = 20
        # self.R_MISTAKE = -10
        # self.R_INVALID = -5
        # self.R_WATER_EMPTY = -5
        # self.R_STEP = -0.1
        # self.R_EXPLORATION = 10
        # self.R_REVISIT = -1
        # self.R_COMPLETE_EXPLORATION = 50
        
        # Internal state variables
        self.rover_pos = None
        self.plants = {}
        self.obstacles = set()
        
        # LIDAR and exploration tracking
        self.explored_map = None
        self.ground_truth_map = None
        
        # ADD: Count-based exploration bonus
        self.visit_counts = None
        
        # Pygame variables for rendering
        self.window = None
        self.clock = None
        self.cell_size = 30
        self.background_img = None
        self.obstacle_img = None
        self.rover_img = None
        self.plant_thirsty_img = None
        self.plant_hydrated_img = None
        self.viewer_3d = None
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        self.collided_with_wall = False
        self.completion_bonus_given = False
        self.total_collisions = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and generate a new random map."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.step_count = 0
        self.collided_with_wall = False
        self.completion_bonus_given = False
        self.total_collisions = 0
        
        # Clear all previous entity locations
        self.plants.clear()
        self.obstacles.clear()
        
        # Generate random map layout
        self._generate_map()
        
        # Initialize exploration map
        self._initialize_exploration()
        
        # Initialize visit counts for count-based exploration
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visit_counts[self.rover_pos[0], self.rover_pos[1]] = 1
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        # If 3D viewer exists, reset its scene to match the new map
        if self.viewer_3d:
            self.viewer_3d.reset_scene()
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Initialize reward for this step
        reward = self.R_STEP  # Base step penalty
        
        # Handle movement actions (0-3)
        if action < 4:
            reward += self._handle_movement(action)
        # Handle watering action (4)
        else:
            reward += self._handle_watering()
        
        # Update LIDAR and exploration
        self._update_lidar()
        
        # Get current observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add watering action flag to info dictionary
        info['is_watering'] = (action == 4)
        
        # Check if episode should terminate
        terminated = self._is_episode_done(info)
        truncated = self.step_count >= self.max_steps
        
        # Check for and award completion bonus
        if info['exploration_percentage'] >= 100 and not self.completion_bonus_given:
            reward += self.R_COMPLETE_EXPLORATION
            self.completion_bonus_given = True
        
        return observation, reward, terminated, truncated, info
    
    def _handle_movement(self, action: int) -> float:
        """
        Handle movement actions with exploration tracking logic.
        
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
            
            was_never_visited = self.visit_counts[new_x, new_y] == 0
            
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 1
            self.rover_pos = (new_x, new_y)
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
            self.visit_counts[new_x, new_y] += 1
            
            if was_never_visited:
                return self.R_EXPLORATION
            else:
                return self.R_REVISIT
        else:
            self.collided_with_wall = True
            self.total_collisions += 1
            return self.R_INVALID
    
    def _handle_watering(self) -> float:
        """Handle the watering action and return the reward."""
        if self.rover_pos in self.plants:
            if self.plants[self.rover_pos]:  # Plant is thirsty
                self.plants[self.rover_pos] = False
                return self.R_GOAL
            else:  # Plant is already hydrated
                return self.R_MISTAKE
        else:
            return self.R_WATER_EMPTY
    
    def _initialize_exploration(self):
        """Initialize the exploration map and ground truth map."""
        self.ground_truth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs_x, obs_y in self.obstacles:
            self.ground_truth_map[obs_x, obs_y] = 1.0
        for (plant_x, plant_y) in self.plants.keys():
            self.ground_truth_map[plant_x, plant_y] = 0.5
        
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
        self._update_lidar()
    
    def _update_lidar(self):
        """Update LIDAR readings based on current rover position."""
        if self.rover_pos is None:
            return
    
    def _is_episode_done(self, info: Dict[str, Any]) -> bool:
        """Check if the episode should terminate."""
        return bool(info['exploration_percentage'] >= 100)
    
    def _get_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array."""
        return self._get_lidar_obs()
    
    def _get_lidar_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array with one-hot encoding for entity types."""
        lidar_size = self.lidar_channels * self.observation_space_per_channel
        position_size = 2
        visit_map_size = self.visit_map_cells
        total_size = lidar_size + position_size + visit_map_size
        
        obs = np.zeros(total_size, dtype=np.float32)
        rover_x, rover_y = self.rover_pos
        
        for i in range(self.lidar_channels):
            angle = (2 * math.pi * i) / self.lidar_channels
            distance = self.lidar_range
            entity_type = self.ENTITY_EMPTY
            
            for r in range(1, self.lidar_range + 1):
                dx = int(r * math.cos(angle))
                dy = int(r * math.sin(angle))
                check_x, check_y = rover_x + dx, rover_y + dy
                
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE
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
            
            start_index = i * self.observation_space_per_channel
            obs[start_index] = distance / self.lidar_range
            one_hot_type = np.zeros(4, dtype=np.float32)
            one_hot_type[entity_type] = 1.0
            obs[start_index + 1 : start_index + 5] = one_hot_type
        
        position_start = lidar_size
        obs[position_start] = rover_x / self.grid_size
        obs[position_start + 1] = rover_y / self.grid_size
        
        visit_map_start = lidar_size + position_size
        visit_map = np.zeros(self.visit_map_cells, dtype=np.float32)
        half_size = self.visit_map_size // 2
        for local_x in range(self.visit_map_size):
            for local_y in range(self.visit_map_size):
                global_x, global_y = rover_x + (local_x - half_size), rover_y + (local_y - half_size)
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    visit_count = min(self.visit_counts[global_x, global_y], 10) / 10.0
                    visit_map[local_x * self.visit_map_size + local_y] = visit_count
                else:
                    visit_map[local_x * self.visit_map_size + local_y] = 1.0
        obs[visit_map_start:visit_map_start + self.visit_map_cells] = visit_map
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the environment."""
        thirsty_count = sum(self.plants.values())
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
        return {
            'rover_position': self.rover_pos,
            'thirsty_plants': thirsty_count,
            'hydrated_plants': len(self.plants) - thirsty_count,
            'total_plants': len(self.plants),
            'step_count': self.step_count,
            'explored_cells': explored_cells,
            'total_cells': total_cells,
            'exploration_percentage': (explored_cells / total_cells) * 100,
            'lidar_range': self.lidar_range,
            'lidar_channels': self.lidar_channels,
            'collided_with_wall': self.collided_with_wall,
            'total_collisions': self.total_collisions
        }
    
    def _generate_map(self):
        """Dispatches the map generation to the selected algorithm."""
        if self.map_generation_algo == 'maze':
            self._generate_map_maze()
        elif self.map_generation_algo == 'rooms':
            self._generate_map_rooms()
        elif self.map_generation_algo == 'border':
            self._generate_map_border()
        else: # Default to 'original'
            self._generate_map_original()

    def _generate_map_original(self):
        """
        Generate an open environment with randomly scattered obstacles.
        This is much better for exploration learning than narrow mazes.
        """
        # Start with empty grid
        self.obstacles = set()
        
        # Generate random obstacle clusters (more natural than single cells)
        num_obstacle_clusters = self.num_obstacles // 3  # Create clusters instead of individual obstacles
        
        for _ in range(num_obstacle_clusters):
            # Pick random center for cluster
            center_x = random.randint(2, self.grid_size - 3)
            center_y = random.randint(2, self.grid_size - 3)
            
            # Create a small cluster (2x2 or 3x3)
            cluster_size = random.choice([2, 3])
            for dx in range(cluster_size):
                for dy in range(cluster_size):
                    obs_x = center_x + dx - cluster_size // 2
                    obs_y = center_y + dy - cluster_size // 2
                    
                    # Make sure obstacle is within bounds
                    if 0 <= obs_x < self.grid_size and 0 <= obs_y < self.grid_size:
                        self.obstacles.add((obs_x, obs_y))
        
        # Get all available positions (not obstacles)
        available_positions = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ) - self.obstacles
        
        if len(available_positions) < self.num_plants + 1:
            raise ValueError(
                f"Not enough available positions ({len(available_positions)}) to place "
                f"{self.num_plants} plants and 1 rover."
            )
        
        # Place plants randomly
        plant_positions = random.sample(list(available_positions), self.num_plants)
        for plant_pos in plant_positions:
            is_thirsty = random.random() < self.thirsty_plant_prob
            self.plants[plant_pos] = is_thirsty
        available_positions -= set(plant_positions)
        
        # Place rover in a random available position
        self.rover_pos = random.choice(list(available_positions))

    def _generate_map_maze(self):
        """
        Generates a maze with wide corridors using a Recursive Backtracking algorithm on a downscaled grid.
        """
        # The 'scale' determines the width of corridors and walls.
        # A larger scale means wider paths and thicker walls.
        scale = random.randint(3, 7)
        
        small_grid_size_x = self.grid_size // scale
        small_grid_size_y = self.grid_size // scale

        # 1. Generate a standard perfect maze on the smaller grid.
        # Using a set to store the locations of paths for efficiency.
        small_paths = set()
        
        start_x = random.randint(0, small_grid_size_x - 1)
        start_y = random.randint(0, small_grid_size_y - 1)

        stack = [(start_x, start_y)]
        small_paths.add((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # N, S, E, W
                # Look for a neighbor 2 cells away
                nx, ny = cx + dx * 2, cy + dy * 2
                
                # Check if neighbor is within bounds and not yet a path
                if (0 <= nx < small_grid_size_x and 0 <= ny < small_grid_size_y and
                    (nx, ny) not in small_paths):
                    possible_moves.append((nx, ny, dx, dy))
            
            if possible_moves:
                nx, ny, dx, dy = random.choice(possible_moves)
                
                # Carve path to neighbor, including the wall in between
                wall_between_x, wall_between_y = cx + dx, cy + dy
                small_paths.add((wall_between_x, wall_between_y))
                small_paths.add((nx, ny))
                
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()

        # 2. Scale up the maze to the full grid size.
        # If a cell in the small grid is NOT a path, it's a wall.
        self.obstacles = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                small_x = x // scale
                small_y = y // scale
                if (small_x, small_y) not in small_paths:
                    self.obstacles.add((x, y))

        # Place plants and rover in the carved-out paths
        available_positions = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ) - self.obstacles

        if len(available_positions) < self.num_plants + 1:
            # Fallback if maze generation is too restrictive for plants/rover
            print("Warning: Not enough space in the maze for plants and rover. Falling back to original map generation.")
            self._generate_map_original()
            return

        plant_positions = random.sample(list(available_positions), self.num_plants)
        for pos in plant_positions:
            self.plants[pos] = random.random() < self.thirsty_plant_prob
        available_positions -= set(plant_positions)
        self.rover_pos = random.choice(list(available_positions))

    def _generate_map_rooms(self):
        """
        Generates a map with multiple rooms connected by corridors.
        """
        self.obstacles = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))
        rooms = []
        
        num_rooms = random.randint(5, 10)
        
        for _ in range(num_rooms):
            w = random.randint(4, 8)
            h = random.randint(4, 8)
            x = random.randint(1, self.grid_size - w - 1)
            y = random.randint(1, self.grid_size - h - 1)
            
            new_room = pygame.Rect(x, y, w, h)
            
            # Check for intersections with existing rooms
            is_overlapping = False
            for other_room in rooms:
                if new_room.colliderect(other_room.inflate(2, 2)): # Inflate to add padding
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                rooms.append(new_room)

        # Carve out the rooms
        for room in rooms:
            for i in range(room.left, room.right):
                for j in range(room.top, room.bottom):
                    self.obstacles.discard((j, i)) # grid is (row, col) -> (y, x)

        # Connect the rooms with corridors
        for i in range(len(rooms) - 1):
            x1, y1 = rooms[i].center
            x2, y2 = rooms[i+1].center

            # Horizontal corridor
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.obstacles.discard((y1, x))
            # Vertical corridor
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.obstacles.discard((y, x2))

        # Place plants and rover in the carved-out paths
        available_positions = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ) - self.obstacles

        if len(available_positions) < self.num_plants + 1:
            print("Warning: Not enough space in rooms for plants/rover. Falling back to original.")
            self._generate_map_original()
            return

        plant_positions = random.sample(list(available_positions), self.num_plants)
        for pos in plant_positions:
            self.plants[pos] = random.random() < self.thirsty_plant_prob
        available_positions -= set(plant_positions)
        self.rover_pos = random.choice(list(available_positions))

    def _generate_map_border(self):
        """
        Generates an open map with a border of obstacles.
        """
        self.obstacles = set()
        # Add border obstacles
        for i in range(self.grid_size):
            self.obstacles.add((i, 0))
            self.obstacles.add((i, self.grid_size - 1))
            self.obstacles.add((0, i))
            self.obstacles.add((self.grid_size - 1, i))

        # Get all available positions (not obstacles)
        available_positions = set(
            (x, y) for x in range(1, self.grid_size - 1) for y in range(1, self.grid_size - 1)
        )

        if len(available_positions) < self.num_plants + 1:
            raise ValueError("Not enough space for plants and rover in bordered map.")
        
        # Place plants randomly
        plant_positions = random.sample(list(available_positions), self.num_plants)
        for plant_pos in plant_positions:
            is_thirsty = random.random() < self.thirsty_plant_prob
            self.plants[plant_pos] = is_thirsty
        available_positions -= set(plant_positions)
        
        # Place rover in a random available position
        self.rover_pos = random.choice(list(available_positions))

    def render(self):
        """Render the environment based on the render_mode."""
        if self.render_mode == 'human':
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
                pygame.display.set_caption("PlantOS Environment")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            frame = self._render_frame()
            self.window.blit(frame, (0, 0))
            pygame.display.flip()
            self.clock.tick(30)
            
            if self.viewer_3d is None:
                self._render_3d() # Initialize 3D viewer
            
            return None

        elif self.render_mode == 'rgb_array':
            frame = self._render_frame()
            return np.transpose(pygame.surfarray.array3d(frame), axes=(1, 0, 2))

    def _render_3d(self):
        """Handles the Ursina 3D rendering."""
        if self.viewer_3d is None:
            self.viewer_3d = PlantOS3DViewer(grid_size=self.grid_size)
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        
        info = self._get_info()
        stats = {
            'timesteps': info['step_count'],
            'collisions': info['total_collisions'],
            'thirsty_plants': info['thirsty_plants']
        }
        self.viewer_3d.update_scene(self.plants, self.rover_pos, stats)
        self.viewer_3d.render_step()

    def _load_assets(self):
        """Load all Pygame assets."""
        if hasattr(self, '_assets_loaded') and self._assets_loaded:
            return
            
        assets_dir = os.path.dirname(os.path.abspath(__file__))
        
        def get_asset_path(filename):
            # Helper to check for assets in the parent directory as a fallback
            local_path = os.path.join(assets_dir, filename)
            parent_path = os.path.join(assets_dir, '..', filename)
            if os.path.exists(local_path):
                return local_path
            elif os.path.exists(parent_path):
                return parent_path
            return None

        asset_paths = {
            'background': get_asset_path('grass_texture.png'),
            'obstacle': get_asset_path('obstacles_texture.png'),
            'rover': get_asset_path('mech_drone_agent.png'),
            'plant_thirsty': get_asset_path('dry_plant_bg.png'),
            'plant_hydrated': get_asset_path('good_plant_bg.png')
        }

        try: self.background_img = pygame.image.load(asset_paths['background']) if asset_paths['background'] else None
        except: self.background_img = None
        if self.background_img: self.background_img = pygame.transform.scale(self.background_img, (self.cell_size, self.cell_size))
        
        try: self.obstacle_img = pygame.image.load(asset_paths['obstacle']) if asset_paths['obstacle'] else None
        except: self.obstacle_img = None
        if self.obstacle_img: self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))

        try: self.rover_img = pygame.image.load(asset_paths['rover']) if asset_paths['rover'] else None
        except: self.rover_img = None
        if self.rover_img: self.rover_img = pygame.transform.scale(self.rover_img, (self.cell_size, self.cell_size))

        try: self.plant_thirsty_img = pygame.image.load(asset_paths['plant_thirsty']) if asset_paths['plant_thirsty'] else None
        except: self.plant_thirsty_img = None
        if self.plant_thirsty_img: self.plant_thirsty_img = pygame.transform.scale(self.plant_thirsty_img, (self.cell_size, self.cell_size))

        try: self.plant_hydrated_img = pygame.image.load(asset_paths['plant_hydrated']) if asset_paths['plant_hydrated'] else None
        except: self.plant_hydrated_img = None
        if self.plant_hydrated_img: self.plant_hydrated_img = pygame.transform.scale(self.plant_hydrated_img, (self.cell_size, self.cell_size))
        
        self._assets_loaded = True

    def _render_frame(self) -> pygame.Surface:
        """Render the current state to a Pygame surface."""
        if not pygame.get_init():
            pygame.init()
        
        self._load_assets()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.background_img: canvas.blit(self.background_img, rect)
                else: pygame.draw.rect(canvas, (34, 139, 34), rect)

        explored_surface = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.SRCALPHA)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.explored_map[x, y] > 0:
                    rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(explored_surface, (200, 200, 200, 100), rect)
        canvas.blit(explored_surface, (0, 0))
        
        for obs_x, obs_y in self.obstacles:
            rect = pygame.Rect(obs_y * self.cell_size, obs_x * self.cell_size, self.cell_size, self.cell_size)
            if self.obstacle_img: canvas.blit(self.obstacle_img, rect)
            else: pygame.draw.rect(canvas, (105, 105, 105), rect)
        
        for (plant_x, plant_y), is_thirsty in self.plants.items():
            rect = pygame.Rect(plant_y * self.cell_size, plant_x * self.cell_size, self.cell_size, self.cell_size)
            img = self.plant_thirsty_img if is_thirsty else self.plant_hydrated_img
            color = (255, 165, 0) if is_thirsty else (0, 255, 0)
            if img: canvas.blit(img, rect)
            else: pygame.draw.rect(canvas, color, rect)
        
        if self.rover_pos:
            rover_x, rover_y = self.rover_pos
            rover_center_x = rover_y * self.cell_size + self.cell_size // 2
            rover_center_y = rover_x * self.cell_size + self.cell_size // 2
            
            for i in range(self.lidar_channels):
                angle = (2 * math.pi * i) / self.lidar_channels
                hit_distance = self.lidar_range
                for r in range(1, self.lidar_range + 1):
                    dx, dy = int(r * math.cos(angle)), int(r * math.sin(angle))
                    check_x, check_y = rover_x + dx, rover_y + dy
                    if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                        hit_distance = r; break
                    if (check_x, check_y) in self.obstacles or (check_x, check_y) in self.plants:
                        hit_distance = r; break
                
                end_x = rover_center_x + int(hit_distance * self.cell_size * math.sin(angle))
                end_y = rover_center_y + int(hit_distance * self.cell_size * math.cos(angle))
                pygame.draw.line(canvas, (100, 100, 255), (rover_center_x, rover_center_y), (end_x, end_y), 1)
                pygame.draw.circle(canvas, (100, 100, 255), (end_x, end_y), 2)
        
        if self.rover_pos:
            rect = pygame.Rect(self.rover_pos[1] * self.cell_size, self.rover_pos[0] * self.cell_size, self.cell_size, self.cell_size)
            if self.rover_img: canvas.blit(self.rover_img, rect)
            else: pygame.draw.rect(canvas, (0, 0, 255), rect)
        
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, x * self.cell_size), (self.grid_size * self.cell_size, x * self.cell_size), 1)
            pygame.draw.line(canvas, (200, 200, 200), (x * self.cell_size, 0), (x * self.cell_size, self.grid_size * self.cell_size), 1)
            
        return canvas
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.viewer_3d:
            self.viewer_3d.close()
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

# Register the environment with Gymnasium
gym.register(
    id='PlantOS-v0',
    entry_point='plantos_env:PlantOSEnvNew',
)