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
    
    def __init__(self, grid_size: int = 21, num_plants: int = 8, num_obstacles: int = 50, 
                 lidar_range: int = 2, lidar_channels: int = 10, thirsty_plant_prob: float = 0.5,
                 observation_mode: str = "grid", render_mode: Optional[str] = None):
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
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        
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
        
        # IMPROVED REWARD CONSTANTS
        self.R_GOAL = 20                # INCREASED: Strong incentive for watering (was 20)
        self.R_MISTAKE = -10                 # REDUCED: Less harsh penalty (was -100)
        self.R_INVALID = -5                  # REDUCED: Much less harsh for wall hits (was -10)
        self.R_WATER_EMPTY = -5                # REDUCED: Less harsh (was -5)
        self.R_STEP = -0.1                    # REDUCED: Tiny step penalty (was -0.05)
        self.R_EXPLORATION = 10              # REDUCED: More realistic (was 50)
        self.R_REVISIT = -1                # REDUCED: Minimal penalty (was -0.1)
        self.R_COMPLETE_EXPLORATION = 50      # Bonus for completing exploration
        
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and generate a new random map."""
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
        
        # Check if episode should terminate
        terminated = self._is_episode_done(info)
        truncated = self.step_count >= self.max_steps
        
        # Check for and award completion bonus
        if info['exploration_percentage'] >= 100 and not self.completion_bonus_given:
            reward += self.R_COMPLETE_EXPLORATION
            self.completion_bonus_given = True
        
        # --- DEBUG: print step summary  ---
        #print(f"[DEBUG] Step {self.step_count:4d} | Pos {self.rover_pos} | "
            #f"Act {action} | Rew {reward:.2f} | "
            #f"Expl {info['exploration_percentage']:.1f}% | "
           # f"Term={terminated} Trunc={truncated}")
        
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

          # --- DEBUG: show the attempted move ---
        #print(f"[DEBUG] Move action {action}: old {self.rover_pos} -> cand ({new_x}, {new_y})")
        
        # Check if new position is valid (within bounds and not an obstacle)
        if (0 <= new_x < self.grid_size and
            0 <= new_y < self.grid_size and
            (new_x, new_y) not in self.obstacles):
            
            # CRITICAL FIX: Check if this cell has EVER been visited
            was_never_visited = self.visit_counts[new_x, new_y] == 0
            
            # Mark old position as explored
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 1
            
            # Move rover
            self.rover_pos = (new_x, new_y)
            
            # Mark new position as current
            self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
            
            # Update visit count
            self.visit_counts[new_x, new_y] += 1

            # --- DEBUG: confirm move happened ---
            #print(f"[DEBUG]    MOVED to {self.rover_pos} | visits={self.visit_counts[new_x, new_y]}")
            
            # FIXED: Reward ONLY for first-time visits
            if was_never_visited:
                # First time visiting this cell - give exploration bonus
                return self.R_EXPLORATION
            else:
                # Revisiting - just give small penalty to discourage backtracking
                return self.R_REVISIT
        else:
            # Invalid movement (hit wall or obstacle)
            self.collided_with_wall = True
            # --- DEBUG: blocked move ---
            #print(f"[DEBUG]    BLOCKED at {self.rover_pos} (hit wall/obstacle/bounds)")
            return self.R_INVALID
    
    def _handle_watering(self) -> float:
        """Handle the watering action and return the reward."""
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
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Mark initial rover position as explored
        self.explored_map[self.rover_pos[0], self.rover_pos[1]] = 2
        
        # Initialize LIDAR
        self._update_lidar()
    
    def _update_lidar(self):
        """Update LIDAR readings based on current rover position."""
        if self.rover_pos is None:
            return
    
    def _is_episode_done(self, info: Dict[str, Any]) -> bool:
        """Check if the episode should terminate."""
        fully_explored = info['exploration_percentage'] >= 100
        # REMOVED: Wall collision termination - too harsh for exploration!
        # The agent should be allowed to bump into walls without episode ending
        return bool(fully_explored)
    
    def _get_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array."""
        return self._get_lidar_obs()
    
    def _get_lidar_obs(self) -> np.ndarray:
        """Generate the LIDAR-based observation array with one-hot encoding for entity types."""
        # Calculate total observation size
        lidar_size = self.lidar_channels * self.observation_space_per_channel
        position_size = 2
        visit_map_size = self.visit_map_cells
        total_size = lidar_size + position_size + visit_map_size
        
        obs = np.zeros(total_size, dtype=np.float32)
        rover_x, rover_y = self.rover_pos
        
        # ===== LIDAR OBSERVATIONS =====
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
                    entity_type = self.ENTITY_OBSTACLE  # Wall is like an obstacle
                    break
                
                pos = (check_x, check_y)
                if pos in self.obstacles:
                    distance = r
                    entity_type = self.ENTITY_OBSTACLE
                    break
                elif pos in self.plants:
                    distance = r
                    # CRITICAL: Check current plant status from self.plants dict
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
        
        # ===== ROVER POSITION =====
        position_start = lidar_size
        obs[position_start] = rover_x / self.grid_size      # Normalized x position
        obs[position_start + 1] = rover_y / self.grid_size  # Normalized y position
        
        # ===== LOCAL VISIT MAP (5x5 grid around rover) =====
        visit_map_start = lidar_size + position_size
        visit_map = np.zeros(self.visit_map_cells, dtype=np.float32)
        
        # Extract 5x5 local visit counts centered on rover
        half_size = self.visit_map_size // 2  # 2 for 5x5
        for local_x in range(self.visit_map_size):
            for local_y in range(self.visit_map_size):
                # Convert local coordinates to global grid coordinates
                global_x = rover_x + (local_x - half_size)
                global_y = rover_y + (local_y - half_size)
                
                # Check if within grid bounds
                if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                    # Normalize visit count (cap at 10 visits, then normalize to [0, 1])
                    visit_count = min(self.visit_counts[global_x, global_y], 10) / 10.0
                    visit_map[local_x * self.visit_map_size + local_y] = visit_count
                else:
                    # Out of bounds = treat as "fully explored" (visited)
                    visit_map[local_x * self.visit_map_size + local_y] = 1.0
        
        obs[visit_map_start:visit_map_start + self.visit_map_cells] = visit_map
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the environment."""
        thirsty_count = sum(self.plants.values())
        hydrated_count = len(self.plants) - thirsty_count
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.grid_size * self.grid_size - len(self.obstacles)
        
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
    
    def _generate_map(self):
        """
        Generate an open environment with randomly scattered obstacles.
        This is much better for exploration learning than narrow mazes.
        """
        # Start with empty grid
        self.obstacles = set()
        
        # Add border walls (optional - can remove for even more openness)
        # for x in range(self.grid_size):
        #     self.obstacles.add((x, 0))
        #     self.obstacles.add((x, self.grid_size - 1))
        # for y in range(self.grid_size):
        #     self.obstacles.add((0, y))
        #     self.obstacles.add((self.grid_size - 1, y))
        
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
    
    # Keep all render methods and close method unchanged
    def render(self, mode='2d'):
        """Render the environment using Pygame (2D) and/or Ursina (3D)."""
        if mode in ['2d', 'human']:
            self._render_2d()
        if mode in ['3d', 'human']:
            self._render_3d()
    
    def _render_3d(self):
        """Handles the Ursina 3D rendering."""
        if self.viewer_3d is None:
            self.viewer_3d = PlantOS3DViewer(grid_size=self.grid_size)
            self.viewer_3d.setup_scene(self.obstacles, self.plants, self.rover_pos)
        self.viewer_3d.update_scene(self.plants, self.rover_pos)
        self.viewer_3d.render_step()
    
    def _render_2d(self):
        """Handles the Pygame 2D rendering."""
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            pygame.display.set_caption("PlantOS Environment")
            self.clock = pygame.time.Clock()
            
            # Load images if available, otherwise use colored rectangles
            assets_dir = os.path.dirname(os.path.abspath(__file__))
            
            try:
                self.background_img = pygame.image.load(os.path.join(assets_dir, 'grass_texture.png'))
                self.background_img = pygame.transform.scale(self.background_img, (self.cell_size, self.cell_size))
            except:
                self.background_img = None
            
            try:
                self.obstacle_img = pygame.image.load(os.path.join(assets_dir, 'obstacles_texture.png'))
                self.obstacle_img = pygame.transform.scale(self.obstacle_img, (self.cell_size, self.cell_size))
            except:
                self.obstacle_img = None
            
            try:
                self.rover_img = pygame.image.load(os.path.join(assets_dir, 'mech_drone_agent.png'))
                self.rover_img = pygame.transform.scale(self.rover_img, (self.cell_size, self.cell_size))
            except:
                self.rover_img = None
            
            try:
                self.plant_thirsty_img = pygame.image.load(os.path.join(assets_dir, 'dry_plant_bg.png'))
                self.plant_thirsty_img = pygame.transform.scale(self.plant_thirsty_img, (self.cell_size, self.cell_size))
            except:
                self.plant_thirsty_img = None
            
            try:
                self.plant_hydrated_img = pygame.image.load(os.path.join(assets_dir, 'good_plant_bg.png'))
                self.plant_hydrated_img = pygame.transform.scale(self.plant_hydrated_img, (self.cell_size, self.cell_size))
            except:
                self.plant_hydrated_img = None
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Draw background
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.background_img:
                    self.window.blit(self.background_img, rect)
                else:
                    pygame.draw.rect(self.window, (34, 139, 34), rect)  # Green
        
        # Draw explored areas (semi-transparent overlay)
        explored_surface = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        explored_surface.set_alpha(100)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.explored_map[x, y] > 0:
                    rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(explored_surface, (200, 200, 200), rect)
        self.window.blit(explored_surface, (0, 0))
        
        # Draw obstacles
        for obs_x, obs_y in self.obstacles:
            rect = pygame.Rect(obs_y * self.cell_size, obs_x * self.cell_size, self.cell_size, self.cell_size)
            if self.obstacle_img:
                self.window.blit(self.obstacle_img, rect)
            else:
                pygame.draw.rect(self.window, (105, 105, 105), rect)  # Gray
        
        # Draw plants
        for (plant_x, plant_y), is_thirsty in self.plants.items():
            rect = pygame.Rect(plant_y * self.cell_size, plant_x * self.cell_size, self.cell_size, self.cell_size)
            if is_thirsty:
                if self.plant_thirsty_img:
                    self.window.blit(self.plant_thirsty_img, rect)
                else:
                    pygame.draw.rect(self.window, (255, 165, 0), rect)  # Orange
            else:
                if self.plant_hydrated_img:
                    self.window.blit(self.plant_hydrated_img, rect)
                else:
                    pygame.draw.rect(self.window, (0, 255, 0), rect)  # Bright green
        
        # Draw LIDAR rays
        rover_x, rover_y = self.rover_pos
        rover_center_x = rover_y * self.cell_size + self.cell_size // 2
        rover_center_y = rover_x * self.cell_size + self.cell_size // 2
        
        for i in range(self.lidar_channels):
            angle = (2 * math.pi * i) / self.lidar_channels
            
            # Find where the ray hits
            hit_distance = self.lidar_range
            for r in range(1, self.lidar_range + 1):
                dx = int(r * math.cos(angle))
                dy = int(r * math.sin(angle))
                check_x = rover_x + dx
                check_y = rover_y + dy
                
                # Check if we hit something
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    hit_distance = r
                    break
                
                pos = (check_x, check_y)
                if pos in self.obstacles or pos in self.plants:
                    hit_distance = r
                    break
            
            # Draw the LIDAR ray
            end_x = rover_center_x + int(hit_distance * self.cell_size * math.sin(angle))
            end_y = rover_center_y + int(hit_distance * self.cell_size * math.cos(angle))
            
            # Color rays based on what they detect
            ray_color = (100, 100, 255)  # Default blue for empty space
            
            pygame.draw.line(self.window, ray_color, 
                           (rover_center_x, rover_center_y), 
                           (end_x, end_y), 1)
            
            # Draw a small circle at the hit point
            pygame.draw.circle(self.window, ray_color, (end_x, end_y), 2)
        
        # Draw rover (on top of LIDAR rays)
        rect = pygame.Rect(rover_y * self.cell_size, rover_x * self.cell_size, self.cell_size, self.cell_size)
        if self.rover_img:
            self.window.blit(self.rover_img, rect)
        else:
            pygame.draw.rect(self.window, (0, 0, 255), rect)  # Blue
        
        # Draw grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.window, (200, 200, 200), (0, x * self.cell_size), (self.grid_size * self.cell_size, x * self.cell_size), 1)
            pygame.draw.line(self.window, (200, 200, 200), (x * self.cell_size, 0), (x * self.cell_size, self.grid_size * self.cell_size), 1)
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
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
    entry_point='plantos_env:PlantOSEnv',
)
