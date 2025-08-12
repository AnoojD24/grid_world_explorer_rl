import numpy as np
from typing import Tuple, List

class Agent:
    """
    Reinforcement Learning agent with multiple algorithm implementations.
    """
    def __init__(self, observation_space: Tuple[int, int], 
                 action_space: int, algorithm: str = 'q_learning',
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 exploration_rate: float = 1.0, min_exploration_rate: float = 0.01,
                 exploration_decay: float = 0.995):
        self.grid_size = observation_space[0]
        self.action_space = action_space
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table for state-action values
        self.q_table = {}
        
        # Initialize policy for visualization
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # For keeping track of visited states
        self.visited_states = np.zeros((self.grid_size, self.grid_size))
        
        # For SARSA algorithm
        self.last_action = None
        
        # For Monte Carlo
        self.episode_memory = []
    
    def get_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        # Convert state to hashable representation
        state_key = self._get_state_key(state)
        
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        
        # Exploitation: choose the best known action
        if state_key not in self.q_table:
            self._init_state(state_key)
        
        # Get action with highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """Update agent's knowledge based on experience."""
        # Convert states to hashable representations
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self._init_state(state_key)
        if next_state_key not in self.q_table and not done:
            self._init_state(next_state_key)
        
        # Mark state as visited
        agent_y, agent_x = np.where(state == 2)
        if len(agent_y) > 0 and len(agent_x) > 0:
            self.visited_states[agent_y[0], agent_x[0]] += 1
        
        # Update based on the selected algorithm
        if self.algorithm == 'q_learning':
            self._q_learning_update(state_key, action, reward, next_state_key, done)
        elif self.algorithm == 'sarsa':
            self._sarsa_update(state_key, action, reward, next_state_key, done)
        elif self.algorithm == 'monte_carlo':
            self._monte_carlo_remember(state_key, action, reward, done)
        
        # Update policy for visualization
        self._update_policy(state)
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration_rate, 
                                       self.exploration_rate * self.exploration_decay)
    
    def _q_learning_update(self, state_key: str, action: int, reward: float, 
                          next_state_key: str, done: bool) -> None:
        """Q-learning update rule."""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
    
    def _sarsa_update(self, state_key: str, action: int, reward: float, 
                     next_state_key: str, done: bool) -> None:
        """SARSA update rule."""
        if done:
            target = reward
        else:
            next_action = self.get_action(next_state_key)
            target = reward + self.discount_factor * self.q_table[next_state_key][next_action]
        
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        self.last_action = action if not done else None
    
    def _monte_carlo_remember(self, state_key: str, action: int, reward: float, done: bool) -> None:
        """Store experience for Monte Carlo updates."""
        self.episode_memory.append((state_key, action, reward))
        
        if done:
            # Calculate returns and update Q-values
            G = 0
            for state_key, action, reward in reversed(self.episode_memory):
                G = self.discount_factor * G + reward
                self.q_table[state_key][action] += self.learning_rate * (G - self.q_table[state_key][action])
            
            # Clear episode memory
            self.episode_memory = []
    
    def _init_state(self, state_key: str) -> None:
        """Initialize Q-values for a new state."""
        self.q_table[state_key] = np.zeros(self.action_space)
    
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to a hashable representation."""
        if isinstance(state, bytes):
            return state
        return state.tobytes()
    
    def _update_policy(self, state: np.ndarray) -> None:
        """Update the policy for visualization."""
        agent_y, agent_x = np.where(state == 2)
        if len(agent_y) > 0 and len(agent_x) > 0:
            agent_y, agent_x = agent_y[0], agent_x[0]
            state_key = self._get_state_key(state)
            if state_key in self.q_table:
                self.policy[agent_y, agent_x] = np.argmax(self.q_table[state_key])
    
    def get_state_value(self, state: np.ndarray) -> float:
        """Get the value of a state (max Q-value)."""
        state_key = self._get_state_key(state)
        if state_key in self.q_table:
            return np.max(self.q_table[state_key])
        return 0.0