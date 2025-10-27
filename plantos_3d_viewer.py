
from ursina import *
from ursina import application

class PlantOS3DViewer:
    """
    Manages the 3D visualization of the PlantOS environment using Ursina.
    """
    def __init__(self, grid_size: int, cell_size: int = 1):
        """
        Initializes the 3D viewer.
        Note: The Ursina app is a singleton; it can only be initialized once.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.app = Ursina(title='PlantOS 3D View', borderless=False, development_mode=True)
        
        # Scene entities
        self.ground = None
        self.rover_entity = None
        self.cell_highlighter = None
        self.plant_entities = {}  # Maps (x, y) -> Ursina Entity
        self.obstacle_entities = {} # Maps (x, y) -> Ursina Entity
        self.lidar_lines = []

        # Setup camera and lighting
        AmbientLight(color=color.rgba(1, 1, 1, 0.8))
        DirectionalLight(color=color.rgba(1, 1, 1, 0.9), direction=(-1, -1, 1))

    def setup_scene(self, obstacles: set, plants: dict, rover_pos: tuple):
        """
        Creates the initial 3D scene with static and dynamic objects.
        """
        # Get assets directory
        import os
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        
        # Create the ground plane
        grass_texture = os.path.join(assets_dir, 'grass_texture.png') if os.path.exists(os.path.join(assets_dir, 'grass_texture.png')) else None
        self.ground = Entity(
            model='plane',
            scale=(self.grid_size * self.cell_size, 1, self.grid_size * self.cell_size),
            color=color.rgb(34, 139, 34) if not grass_texture else color.white,
            texture=grass_texture,
            texture_scale=(self.grid_size, self.grid_size) if grass_texture else None
        )

        # Create obstacles
        obstacle_texture = os.path.join(assets_dir, 'obstacles_texture.png') if os.path.exists(os.path.join(assets_dir, 'obstacles_texture.png')) else None
        for obs_pos in obstacles:
            x, y = obs_pos
            self.obstacle_entities[obs_pos] = Entity(
                model='cube',
                color=color.rgb(105, 105, 105) if not obstacle_texture else color.white,
                texture=obstacle_texture,
                position=self._grid_to_world(x, y, 0.5),
                scale=(self.cell_size, self.cell_size, self.cell_size)
            )
        
        # Create initial plants and rover
        self.update_scene(plants, rover_pos)

    def update_scene(self, plants: dict, rover_pos: tuple):
        """
        Updates the 3D scene to reflect the current state of the environment.
        """
        # Get assets directory
        import os
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        
        # Update rover
        if self.rover_entity is None:
            rover_texture = os.path.join(assets_dir, 'mech_drone_agent.png')
            self.rover_entity = Entity(
                model='quad', 
                texture=rover_texture if os.path.exists(rover_texture) else None,
                color=color.blue if not os.path.exists(rover_texture) else color.white,
                billboard=True, 
                scale=self.cell_size * 2
            )
        self.rover_entity.position = self._grid_to_world(rover_pos[0], rover_pos[1], 0.5)

        if self.cell_highlighter is None:
            self.cell_highlighter = Entity(
                model='cube',
                color=color.green,
                scale=(self.cell_size, 0.1, self.cell_size),
                mode='wireframe'
            )
        self.cell_highlighter.position = self._grid_to_world(rover_pos[0], rover_pos[1], 0.05)

        # Make camera follow the rover with better positioning
        # Position camera above and behind the rover for better view
        camera_offset_y = self.grid_size * 0.8  # Height above ground
        camera_offset_z = -self.grid_size * 0.6  # Distance behind
        camera.position = self.rover_entity.position + Vec3(0, camera_offset_y, camera_offset_z)
        camera.look_at(self.rover_entity)

        # Update plants
        current_plant_positions = set(self.plant_entities.keys())
        target_plant_positions = set(plants.keys())

        # Remove plants that no longer exist (e.g. after a reset)
        for pos in current_plant_positions - target_plant_positions:
            destroy(self.plant_entities.pop(pos))

        # Get texture paths
        thirsty_texture = os.path.join(assets_dir, 'dry_plant_bg.png')
        hydrated_texture = os.path.join(assets_dir, 'good_plant_bg.png')

        # Add new plants or update existing ones
        for pos, is_thirsty in plants.items():
            if pos not in self.plant_entities:
                self.plant_entities[pos] = Entity(
                    model='quad',
                    scale=self.cell_size * 2,
                    billboard=True
                )
            
            entity = self.plant_entities[pos]
            entity.position = self._grid_to_world(pos[0], pos[1], 0.5)
            
            # Set texture or color based on plant state
            if is_thirsty:
                if os.path.exists(thirsty_texture):
                    entity.texture = thirsty_texture
                    entity.color = color.white
                else:
                    entity.texture = None
                    entity.color = color.orange  # Thirsty = orange
            else:
                if os.path.exists(hydrated_texture):
                    entity.texture = hydrated_texture
                    entity.color = color.white
                else:
                    entity.texture = None
                    entity.color = color.green  # Hydrated = green

    def reset_scene(self):
        """Destroys all entities to prepare for a new scene."""
        for entity in self.obstacle_entities.values():
            destroy(entity)
        self.obstacle_entities.clear()

        for entity in self.plant_entities.values():
            destroy(entity)
        self.plant_entities.clear()

        if self.rover_entity:
            destroy(self.rover_entity)
            self.rover_entity = None

        if self.cell_highlighter:
            destroy(self.cell_highlighter)
            self.cell_highlighter = None

    def render_step(self):
        """
        Renders a single frame of the 3D view.
        This should be called in the main application loop.
        """
        self.app.step()

    def _grid_to_world(self, grid_x, grid_y, height):
        """
        Converts grid coordinates to world coordinates for positioning entities.
        Centers the grid around the origin (0,0,0).
        """
        world_x = (grid_x - self.grid_size / 2 + 0.5) * self.cell_size
        world_z = (grid_y - self.grid_size / 2 + 0.5) * self.cell_size
        return (world_x, height * self.cell_size, world_z)

    def close(self):
        """
        Closes the Ursina application window.
        """
        application.quit()
