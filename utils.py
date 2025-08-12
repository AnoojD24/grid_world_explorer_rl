import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from scipy.ndimage import gaussian_filter1d

def plot_learning_curve(episode_rewards: List[float], episode_lengths: List[int], 
                       exploration_rates: List[float], filename: str = 'learning_curve.png',
                       smoothing_factor: float = 0.8):
    """
    Plot learning curves for the agent with improved visualization.
    
    Args:
        episode_rewards: List of total rewards for each episode
        episode_lengths: List of episode lengths (steps)
        exploration_rates: List of exploration rates over episodes
        filename: Where to save the plot
        smoothing_factor: Factor for smoothing the curves (0 to 1)
    """
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    episodes = np.arange(len(episode_rewards))
    
    # Calculate smoothed curves
    smoothed_rewards = gaussian_filter1d(episode_rewards, sigma=len(episode_rewards) * (1 - smoothing_factor))
    smoothed_lengths = gaussian_filter1d(episode_lengths, sigma=len(episode_lengths) * (1 - smoothing_factor))
    
    # Plot rewards
    ax = axes[0]
    sns.scatterplot(x=episodes, y=episode_rewards, alpha=0.15, color='blue', ax=ax, label='Raw')
    sns.lineplot(x=episodes, y=smoothed_rewards, color='blue', linewidth=2, ax=ax, label='Smoothed')
    ax.set_title('Training Progress: Episode Rewards', fontsize=12, pad=20)
    ax.set_ylabel('Total Reward')
    ax.legend()
    
    # Plot episode lengths
    ax = axes[1]
    sns.scatterplot(x=episodes, y=episode_lengths, alpha=0.15, color='red', ax=ax, label='Raw')
    sns.lineplot(x=episodes, y=smoothed_lengths, color='red', linewidth=2, ax=ax, label='Smoothed')
    ax.set_title('Training Progress: Episode Lengths', fontsize=12, pad=20)
    ax.set_ylabel('Steps per Episode')
    ax.legend()
    
    # Plot exploration rate
    ax = axes[2]
    sns.lineplot(x=episodes, y=exploration_rates, color='green', linewidth=2, ax=ax)
    ax.set_title('Training Progress: Exploration Rate', fontsize=12, pad=20)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Exploration Rate (ε)')
    
    # Add summary statistics
    for ax, data, label in zip(axes, 
                              [episode_rewards, episode_lengths, exploration_rates],
                              ['Reward', 'Steps', 'ε']):
        stats_text = f'Mean {label}: {np.mean(data):.2f}\nMax {label}: {np.max(data):.2f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_value_function(value_grid: np.ndarray, filename: str = 'value_function.png'):
    """
    Plot the value function as an enhanced heatmap.
    
    Args:
        value_grid: 2D array of state values
        filename: Where to save the plot
    """
    sns.set_style("white")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with improved aesthetics
    # Scale values for better visualization if they're too small
    max_abs_value = np.max(np.abs(value_grid))
    if max_abs_value > 0:
        scaled_grid = value_grid / max_abs_value
    else:
        scaled_grid = value_grid
    
    # Create heatmap with improved aesthetics
    ax = sns.heatmap(scaled_grid,
                     cmap='RdYlBu_r',    # Better for value visualization
                     center=0,            # Center the colormap at 0
                     annot=value_grid,    # Show original values in cells
                     fmt='.4f',           # Show 4 decimal places
                     cbar_kws={'label': 'State Value'},
                     square=True,         # Make cells square
                     annot_kws={'size': 8})  # Adjust annotation text size
    
    # Customize the plot
    ax.set_title('Value Function Heatmap', pad=20)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add text with summary statistics
    stats_text = f'Min Value: {np.min(value_grid):.2f}\nMax Value: {np.max(value_grid):.2f}\nMean Value: {np.mean(value_grid):.2f}'
    plt.text(1.15, 0.95, stats_text, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_policy_visualization(policy_grid: np.ndarray, filename: str = 'policy.png'):
    """
    Create an enhanced visualization of the policy.
    
    Args:
        policy_grid: 2D array where each cell contains the best action index
        filename: Where to save the plot
    """
    sns.set_style("white")
    
    # Define arrow characters and colors
    directions = {
        0: ('↑', 'north'),  # UP
        1: ('→', 'east'),   # RIGHT
        2: ('↓', 'south'),  # DOWN
        3: ('←', 'west')    # LEFT
    }
    
    # Create figure
    rows, cols = policy_grid.shape
    fig, ax = plt.subplots(figsize=(max(6, cols), max(6, rows)))
    
    # Create background grid
    ax.set_facecolor('#f0f0f0')
    
    # Create arrow grid with improved visualization
    for y in range(rows):
        for x in range(cols):
            action = policy_grid[y, x]
            arrow, direction = directions[action]
            
            # Add cell background
            cell = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=True, 
                               facecolor='white', edgecolor='#cccccc')
            ax.add_patch(cell)
            
            # Add arrow
            ax.text(x, y, arrow, ha='center', va='center', fontsize=14,
                   fontweight='bold', color='#2c3e50')
    
    # Customize the plot
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_title('Policy Map', pad=20, fontsize=14)
    ax.grid(True, color='#cccccc', linestyle='-', linewidth=1)
    
    # Add legend
    legend_elements = [plt.Text(0, 0, f"{directions[i][0]}: {directions[i][1].title()}")
                      for i in range(4)]
    ax.legend(handles=legend_elements, loc='center left', 
             bbox_to_anchor=(1, 0.5), title='Actions')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()