import sys
import os
import copy

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plantos_env import PlantOSEnv
import numpy as np
from typing import Dict, Any

class PlantOSEnvMCTS(PlantOSEnv):
    """
    PlantOSEnvMCTS inherits from PlantOSEnv and adds get_state() and set_state()
    methods, which are crucial for MCTS simulations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a deep-copyable dictionary representing the current internal state of the environment.
        """
        state = {
            'rover_pos': self.rover_pos,
            'plants': copy.deepcopy(self.plants),  # Deep copy the dictionary
            'obstacles': copy.deepcopy(self.obstacles), # Deep copy the set
            'explored_map': self.explored_map.copy(), # Copy the numpy array
            'ground_truth_map': self.ground_truth_map.copy(), # Copy the numpy array
            'step_count': self.step_count,
            'previous_explored_count': self.previous_explored_count,
            'collided_with_wall': self.collided_with_wall,
            'completion_bonus_given': self.completion_bonus_given,
            'milestone_10': self.milestone_10,
            'milestone_25': self.milestone_25,
            'milestone_50': self.milestone_50,
            'milestone_75': self.milestone_75,
            'last_action': self.last_action,
        }
        return state

    def set_state(self, state: Dict[str, Any]):
        """
        Restores the environment to a previously saved state.
        """
        self.rover_pos = state['rover_pos']
        self.plants = copy.deepcopy(state['plants'])
        self.obstacles = copy.deepcopy(state['obstacles'])
        self.explored_map = state['explored_map'].copy()
        self.ground_truth_map = state['ground_truth_map'].copy()
        self.step_count = state['step_count']
        self.previous_explored_count = state['previous_explored_count']
        self.collided_with_wall = state['collided_with_wall']
        self.completion_bonus_given = state['completion_bonus_given']
        self.milestone_10 = state['milestone_10']
        self.milestone_25 = state['milestone_25']
        self.milestone_50 = state['milestone_50']
        self.milestone_75 = state['milestone_75']
        self.last_action = state['last_action']

# Note: We do NOT register PlantOSEnvMCTS with gymnasium, as it's an internal class for MCTS.
# The original PlantOS-v0 registration remains in plantos_env.py.
