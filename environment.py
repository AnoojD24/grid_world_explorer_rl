import numpy as np
from typing import Tuple, Dict, Any

class GridWorldEnv:
    """
    Grid World environment with goals, obstacles, and dynamic elements.
    """
    
    # Action space definitions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, grid_size: int = 10, obstacle_density: float = 0.2, 
                 num_goals: int = 3, dynamic_goals: bool = True):
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.num_goals = num_goals
        self.dynamic_goals = dynamic_goals
        
        # Define action and observation spaces
        self.action_space = 4  # Up, Right, Down, Left
        self.observation_space = (grid_size, grid_size)
        
        # Initialize grid
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Place obstacles
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        obstacle_positions = np.random.choice(
            self.grid_size * self.grid_size,
            size=num_obstacles,
            replace=False
        )
        for pos in obstacle_positions:
            x, y = pos % self.grid_size, pos // self.grid_size
            self.grid[y, x] = -1  # -1 represents obstacle
        
        # Place goals
        self.goals = []
        available_positions = [(x, y) for y in range(self.grid_size) 
                               for x in range(self.grid_size) if self.grid[y, x] == 0]
        goal_positions = np.random.choice(len(available_positions), 
                                         size=min(self.num_goals, len(available_positions)),
                                         replace=False)
        for idx in goal_positions:
            x, y = available_positions[idx]
            self.grid[y, x] = 1  # 1 represents goal
            self.goals.append((x, y))
            
        # Place agent
        remaining_positions = [(x, y) for y in range(self.grid_size) 
                               for x in range(self.grid_size) if self.grid[y, x] == 0]
        if remaining_positions:
            agent_idx = np.random.choice(len(remaining_positions))
            self.agent_x, self.agent_y = remaining_positions[agent_idx]
        else:
            # If no space left, clear a random cell
            self.agent_x = np.random.randint(0, self.grid_size)
            self.agent_y = np.random.randint(0, self.grid_size)
            self.grid[self.agent_y, self.agent_x] = 0
            
        # Initialize step counter
        self.steps = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.steps += 1
        
        # Move agent
        new_x, new_y = self.agent_x, self.agent_y
        
        if action == self.UP and self.agent_y > 0:
            new_y -= 1
        elif action == self.RIGHT and self.agent_x < self.grid_size - 1:
            new_x += 1
        elif action == self.DOWN and self.agent_y < self.grid_size - 1:
            new_y += 1
        elif action == self.LEFT and self.agent_x > 0:
            new_x -= 1
        
        # Check if new position is valid
        if self.grid[new_y, new_x] != -1:  # Not an obstacle
            self.agent_x, self.agent_y = new_x, new_y
        
        # Calculate reward
        reward = -0.01  # Small penalty for each step to encourage efficiency
        done = False
        info = {}
        
        # Check for goal
        if (self.agent_x, self.agent_y) in self.goals:
            reward = 1.0
            self.goals.remove((self.agent_x, self.agent_y))
            self.grid[self.agent_y, self.agent_x] = 0
            
            # Add new goal if dynamic goals are enabled
            if self.dynamic_goals:
                available_positions = [(x, y) for y in range(self.grid_size) 
                                      for x in range(self.grid_size) 
                                      if self.grid[y, x] == 0 and (x != self.agent_x or y != self.agent_y)]
                if available_positions:
                    new_goal_idx = np.random.choice(len(available_positions))
                    new_goal_x, new_goal_y = available_positions[new_goal_idx]
                    self.grid[new_goal_y, new_goal_x] = 1
                    self.goals.append((new_goal_x, new_goal_y))
        
        # Check if episode should end
        if len(self.goals) == 0 or self.steps >= self.grid_size * self.grid_size * 2:
            done = True
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Return the current state representation."""
        # Create a copy of the grid with agent position marked as 2
        state = self.grid.copy()
        state[self.agent_y, self.agent_x] = 2  # 2 represents agent
        return state