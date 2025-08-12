import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from environment import GridWorldEnv
from agent import Agent

class Visualization:
    """
    Visualization class for rendering the Grid World environment.
    """
    # Define colors
    COLORS = {
        'background': (20, 20, 40),
        'grid_line': (50, 50, 70),
        'empty': (30, 30, 50),
        'agent': (65, 105, 225),  # Royal Blue
        'goal': (50, 205, 50),    # Lime Green
        'obstacle': (139, 69, 19), # Saddle Brown
        'text': (255, 255, 255),
        'highlight': (255, 215, 0) # Gold
    }
    
    # Direction vectors for policy visualization
    DIRECTIONS = [
        (0, -1),  # UP
        (1, 0),   # RIGHT
        (0, 1),   # DOWN
        (-1, 0)   # LEFT
    ]
    
    def __init__(self, env: GridWorldEnv, render_mode: str = 'fancy', fps: int = 60):
        """Initialize the visualization."""
        self.env = env
        self.render_mode = render_mode
        self.fps = fps
        
        # Initialize pygame
        pygame.init()
        
        # Set window size
        self.cell_size = 60
        self.window_width = env.grid_size * self.cell_size + 400  # Extra space for info panel
        self.window_height = env.grid_size * self.cell_size
        
        # Create window
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Grid World Explorer')
        
        # Clock for controlling FPS
        self.clock = pygame.time.Clock()
        
        # Load fonts
        self.font_small = pygame.font.SysFont('Arial', 12)
        self.font_medium = pygame.font.SysFont('Arial', 16)
        self.font_large = pygame.font.SysFont('Arial', 24)
        
        # Animation variables
        self.animation_frames = 10
        self.current_frame = 0
        self.prev_agent_pos = (env.agent_x, env.agent_y)
        
        # Load textures if in fancy mode
        if render_mode == 'fancy':
            self._load_textures()
        
        # For value function visualization
        self.value_surface = None
    
    def _load_textures(self):
        """Load textures for fancy rendering mode."""
        # In a real implementation, you would load actual image files
        # Here we'll create placeholder surfaces
        self.textures = {
            'agent': self._create_agent_texture(),
            'goal': self._create_goal_texture(),
            'obstacle': self._create_obstacle_texture(),
            'floor': self._create_floor_texture()
        }
    
    def _create_agent_texture(self):
        """Create a texture for the agent."""
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        center = (self.cell_size // 2, self.cell_size // 2)
        
        # Create stronger glow effect
        glow_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        for i in range(15, 0, -1):
            alpha = 15 + i * 6
            pygame.draw.circle(
                glow_surface, 
                (*self.COLORS['agent'][:3], alpha), 
                center, 
                self.cell_size // 2 + i - 5
            )
        surface.blit(glow_surface, (0, 0))
        
        # Draw main body
        radius = self.cell_size // 2 - 8
        pygame.draw.circle(surface, self.COLORS['agent'], center, radius)
        
        # Add metallic highlight
        highlight = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius):
            alpha = 100 - (i * 100 // radius)
            pygame.draw.circle(
                highlight,
                (255, 255, 255, alpha),
                (radius, radius // 2),
                radius - i
            )
        surface.blit(highlight, (center[0] - radius, center[1] - radius))
        
        # Draw eyes
        eye_color = (255, 255, 255)
        eye_radius = self.cell_size // 10
        eye_offset = self.cell_size // 6
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                         (center[0] - eye_offset, center[1] - eye_offset // 2), 
                         eye_radius)
        # Right eye
        pygame.draw.circle(surface, eye_color,
                         (center[0] + eye_offset, center[1] - eye_offset // 2),
                         eye_radius)
        
        # Add pupils that follow movement
        pupil_color = (0, 0, 0)
        pupil_radius = eye_radius // 2
        # Left pupil
        pygame.draw.circle(surface, pupil_color,
                         (center[0] - eye_offset + 2, center[1] - eye_offset // 2),
                         pupil_radius)
        # Right pupil
        pygame.draw.circle(surface, pupil_color,
                         (center[0] + eye_offset + 2, center[1] - eye_offset // 2),
                         pupil_radius)
        
        # Add antenna
        antenna_start = (center[0], center[1] - radius + 5)
        antenna_end = (center[0], center[1] - radius - 10)
        antenna_color = (100, 150, 255)
        pygame.draw.line(surface, antenna_color, antenna_start, antenna_end, 3)
        pygame.draw.circle(surface, antenna_color, antenna_end, 4)
        
        return surface
    
    def _create_goal_texture(self):
        """Create a texture for the goal."""
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        
        # Draw a star-like shape
        center = (self.cell_size // 2, self.cell_size // 2)
        points = []
        for i in range(10):
            radius = self.cell_size // 3 if i % 2 == 0 else self.cell_size // 6
            angle = 2 * np.pi * i / 10
            points.append((
                center[0] + radius * np.sin(angle),
                center[1] + radius * np.cos(angle)
            ))
        pygame.draw.polygon(surface, self.COLORS['goal'], points)
        
        # Add glow
        glow_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        for i in range(15, 0, -1):
            alpha = 5 + i * 3
            pygame.draw.circle(
                glow_surface, 
                (*self.COLORS['goal'][:3], alpha), 
                center, 
                self.cell_size // 4 + i
            )
        surface.blit(glow_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
        
        return surface
    
    def _create_obstacle_texture(self):
        """Create a texture for obstacles."""
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.fill(self.COLORS['obstacle'])
        
        # Add some texture details
        for _ in range(20):
            x = np.random.randint(0, self.cell_size)
            y = np.random.randint(0, self.cell_size)
            radius = np.random.randint(2, 5)
            color = tuple(max(0, min(255, c + np.random.randint(-20, 20))) for c in self.COLORS['obstacle'])
            pygame.draw.circle(surface, color, (x, y), radius)
        
        return surface
    
    def _create_floor_texture(self):
        """Create a texture for the floor."""
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.fill(self.COLORS['empty'])
        
        # Add some subtle texture
        for _ in range(5):
            x = np.random.randint(0, self.cell_size)
            y = np.random.randint(0, self.cell_size)
            radius = np.random.randint(1, 3)
            color = tuple(max(0, min(255, c + np.random.randint(-10, 10))) for c in self.COLORS['empty'])
            pygame.draw.circle(surface, color, (x, y), radius)
        
        return surface
    
    def render(self, agent: Agent):
        """Render the current state of the environment."""
        # Fill background
        self.window.fill(self.COLORS['background'])
        
        # Get the state
        state = self.env._get_state()
        
        # Render grid
        self._render_grid(state, agent)
        
        # Render info panel
        self._render_info_panel(agent)
        
        # Update the display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(self.fps)
    
    def _render_grid(self, state: np.ndarray, agent: Agent):
        """Render the grid."""
        # Update the animation frame
        self.current_frame = (self.current_frame + 1) % self.animation_frames
        
        # Find agent position
        agent_y, agent_x = np.where(state == 2)
        if len(agent_y) > 0 and len(agent_x) > 0:
            agent_y, agent_x = agent_y[0], agent_x[0]
            # Update previous agent position if needed
            if self.current_frame == 0:
                self.prev_agent_pos = (agent_x, agent_y)
        
        # Draw cells
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Determine cell content
                cell_value = state[y, x]
                
                if self.render_mode == 'simple':
                    self._render_simple_cell(rect, cell_value, agent)
                else:
                    self._render_fancy_cell(rect, cell_value, x, y, agent)
        
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.window,
                self.COLORS['grid_line'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.env.grid_size * self.cell_size),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.window,
                self.COLORS['grid_line'],
                (0, i * self.cell_size),
                (self.env.grid_size * self.cell_size, i * self.cell_size),
                1
            )
    
    def _render_simple_cell(self, rect: pygame.Rect, cell_value: int, agent: Agent):
        """Render a cell in simple mode."""
        if cell_value == -1:  # Obstacle
            pygame.draw.rect(self.window, self.COLORS['obstacle'], rect)
        elif cell_value == 1:  # Goal
            pygame.draw.rect(self.window, self.COLORS['empty'], rect)
            pygame.draw.circle(
                self.window,
                self.COLORS['goal'],
                rect.center,
                self.cell_size // 3
            )
        elif cell_value == 2:  # Agent
            pygame.draw.rect(self.window, self.COLORS['empty'], rect)
            pygame.draw.circle(
                self.window,
                self.COLORS['agent'],
                rect.center,
                self.cell_size // 3
            )
        else:  # Empty
            pygame.draw.rect(self.window, self.COLORS['empty'], rect)
        
        # Draw policy arrow if available
        if cell_value == 2:
            agent_y, agent_x = np.where(self.env._get_state() == 2)
            if len(agent_y) > 0 and len(agent_x) > 0:
                policy_action = agent.policy[agent_y[0], agent_x[0]]
                self._draw_arrow(rect.center, policy_action)
    
    def _render_fancy_cell(self, rect: pygame.Rect, cell_value: int, x: int, y: int, agent: Agent):
        """Render a cell in fancy mode with textures and effects."""
        # Draw the floor texture
        self.window.blit(self.textures['floor'], rect)
        
        # Draw the appropriate texture based on cell value
        if cell_value == -1:  # Obstacle
            self.window.blit(self.textures['obstacle'], rect)
        elif cell_value == 1:  # Goal
            self.window.blit(self.textures['goal'], rect)
            
            # Add pulsating effect
            pulse = (np.sin(pygame.time.get_ticks() * 0.005) + 1) * 10
            pygame.draw.circle(
                self.window,
                (*self.COLORS['goal'][:3], 50),
                rect.center,
                self.cell_size // 4 + pulse,
                0
            )
        elif cell_value == 2:  # Agent
            # For the agent, we do smooth movement animation between cells
            current_x = self.prev_agent_pos[0]
            current_y = self.prev_agent_pos[1]
            target_x = x
            target_y = y
            
            # Calculate interpolated position for smooth movement
            interp_x = current_x + (target_x - current_x) * (self.current_frame / self.animation_frames)
            interp_y = current_y + (target_y - current_y) * (self.current_frame / self.animation_frames)
            
            # Position for agent
            agent_rect = pygame.Rect(
                interp_x * self.cell_size,
                interp_y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            
            # Draw agent
            self.window.blit(self.textures['agent'], agent_rect)
            
            # Draw policy arrow
            policy_action = agent.policy[y, x]
            self._draw_arrow((agent_rect.centerx, agent_rect.centery), policy_action, fancy=True)
        
        # Display state value for all cells
        state_copy = self.env._get_state().copy()
        state_copy[y, x] = 2  # Temporarily mark this as agent position
        value = agent.get_state_value(state_copy)
        
        # Render the value as text
        if value > 0:
            value_text = self.font_small.render(f"{value:.2f}", True, self.COLORS['text'])
            self.window.blit(value_text, (rect.x + 2, rect.y + 2))
    
    def _draw_arrow(self, center: tuple, direction: int, fancy: bool = False):
        """Draw an arrow indicating policy direction."""
        if fancy:
            # Draw a fancier arrow with gradient and better shape
            arrow_length = self.cell_size // 3
            arrow_width = self.cell_size // 6
            dir_vec = self.DIRECTIONS[direction]
            
            # Arrow endpoint
            end_x = center[0] + dir_vec[0] * arrow_length
            end_y = center[1] + dir_vec[1] * arrow_length
            
            # Calculate perpendicular vector for arrow head
            perp_x, perp_y = -dir_vec[1], dir_vec[0]
            
            # Arrow head points
            arrow_head = [
                (end_x, end_y),
                (end_x - dir_vec[0] * arrow_width - perp_x * arrow_width, 
                 end_y - dir_vec[1] * arrow_width - perp_y * arrow_width),
                (end_x - dir_vec[0] * arrow_width + perp_x * arrow_width, 
                 end_y - dir_vec[1] * arrow_width + perp_y * arrow_width),
            ]
            
            # Draw arrow shaft with gradient
            for i in range(10):
                progress = i / 10
                shaft_x = center[0] + dir_vec[0] * arrow_length * progress
                shaft_y = center[1] + dir_vec[1] * arrow_length * progress
                radius = arrow_width // 2 * (1 - progress * 0.7)
                alpha = 255 - 100 * progress
                color = (*self.COLORS['highlight'][:3], alpha)
                pygame.draw.circle(self.window, color, (shaft_x, shaft_y), radius)
            
            # Draw arrow head
            pygame.draw.polygon(self.window, self.COLORS['highlight'], arrow_head)
        else:
            # Simple arrow
            arrow_length = self.cell_size // 4
            dir_vec = self.DIRECTIONS[direction]
            end_x = center[0] + dir_vec[0] * arrow_length
            end_y = center[1] + dir_vec[1] * arrow_length
            pygame.draw.line(self.window, self.COLORS['highlight'], center, (end_x, end_y), 3)
    
    def _render_info_panel(self, agent: Agent):
        """Render the information panel."""
        panel_rect = pygame.Rect(
            self.env.grid_size * self.cell_size,
            0,
            self.window_width - self.env.grid_size * self.cell_size,
            self.window_height
        )
        pygame.draw.rect(self.window, (40, 40, 60), panel_rect)
        
        # Display algorithm info
        title_text = self.font_large.render(f"Algorithm: {agent.algorithm.upper()}", True, self.COLORS['text'])
        self.window.blit(title_text, (panel_rect.x + 20, 20))
        
        # Display exploration rate
        explore_text = self.font_medium.render(
            f"Exploration rate: {agent.exploration_rate:.3f}", 
            True, 
            self.COLORS['text']
        )
        self.window.blit(explore_text, (panel_rect.x + 20, 70))
        
        # Display goal count
        goals_text = self.font_medium.render(
            f"Remaining goals: {len(self.env.goals)}", 
            True, 
            self.COLORS['text']
        )
        self.window.blit(goals_text, (panel_rect.x + 20, 100))
        
        # Display steps
        steps_text = self.font_medium.render(
            f"Steps: {self.env.steps}", 
            True, 
            self.COLORS['text']
        )
        self.window.blit(steps_text, (panel_rect.x + 20, 130))
        
        # Render value function visualization
        self._render_value_function(agent, panel_rect)
    
    def _render_value_function(self, agent: Agent, panel_rect: pygame.Rect):
        """Render a visualization of the value function."""
        # Create value function heatmap
        grid_size = self.env.grid_size
        value_map = np.zeros((grid_size, grid_size))
        
        for y in range(grid_size):
            for x in range(grid_size):
                state = self.env._get_state().copy()
                # Skip obstacles
                if state[y, x] == -1:
                    continue
                
                # Set agent position for value estimation
                agent_y, agent_x = np.where(state == 2)
                if len(agent_y) > 0 and len(agent_x) > 0:
                    state[agent_y[0], agent_x[0]] = 0
                state[y, x] = 2
                value_map[y, x] = agent.get_state_value(state)
        
        # Normalize values for better visualization
        if np.max(value_map) > 0:
            value_map = value_map / np.max(value_map)
        
        # Create a colormap
        colors = ['darkblue', 'blue', 'cyan', 'yellow', 'red']
        cmap = LinearSegmentedColormap.from_list('value_cmap', colors, N=100)
        
        # Create the heatmap
        fig = Figure(figsize=(4, 4), dpi=80)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(value_map, cmap=cmap, interpolation='nearest')
        ax.set_title("Value Function")
        fig.colorbar(im)
        ax.axis('off')
        
        # Convert matplotlib figure to pygame surface
        canvas.draw()
        buf = canvas.buffer_rgba()
        X, Y = canvas.get_width_height()
        heatmap = pygame.image.frombuffer(buf, (X, Y), "RGBA")
        
        # Scale to fit panel
        heatmap_width = panel_rect.width - 40
        heatmap_height = int(heatmap_width * (Y / X))
        heatmap = pygame.transform.scale(heatmap, (heatmap_width, heatmap_height))
        
        # Blit to screen
        self.window.blit(heatmap, (panel_rect.x + 20, 180))
    
    def display_episode_summary(self, episode: int, total_reward: float, agent: Agent):
        """Display a summary at the end of an episode."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        self.window.blit(overlay, (0, 0))
        
        # Display episode info
        episode_text = self.font_large.render(
            f"Episode {episode+1} Complete", 
            True, 
            self.COLORS['text']
        )
        reward_text = self.font_large.render(
            f"Total Reward: {total_reward:.2f}", 
            True, 
            self.COLORS['text']
        )
        continue_text = self.font_medium.render(
            "Starting next episode...", 
            True, 
            self.COLORS['text']
        )
        
        # Calculate positions
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        
        self.window.blit(episode_text, 
                        (center_x - episode_text.get_width() // 2, 
                         center_y - 50))
        self.window.blit(reward_text, 
                        (center_x - reward_text.get_width() // 2, 
                         center_y))
        self.window.blit(continue_text, 
                        (center_x - continue_text.get_width() // 2, 
                         center_y + 50))
        
        # Update display
        pygame.display.flip()