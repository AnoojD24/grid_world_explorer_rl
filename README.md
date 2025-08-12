# Grid World Explorer

A visually appealing Grid World environment for reinforcement learning exploration.

<img width="1000" height="632" alt="grid explorer" src="https://github.com/user-attachments/assets/ac4d0f48-e7ee-4a45-95d3-abfe25d78bed" />


## Overview

Grid World Explorer is an educational tool that demonstrates core reinforcement learning concepts through a visually engaging grid-based environment. The agent must navigate a world containing obstacles and collect rewards, using various RL algorithms to learn optimal policies.

## Features

- Visually appealing grid world with modern visual effects
- Multiple reinforcement learning algorithms:
  - Q-learning
  - SARSA
  - Monte Carlo learning
- Real-time visualization of:
  - Agent's policy
  - Value function (as a heatmap)
  - Learning progress
- Dynamic environment with obstacles and goals
- Smooth animations and visual feedback

## Reinforcement Learning Concepts Demonstrated

- Markov Decision Processes (MDPs)
- Exploration vs. Exploitation
- Value Functions and Policies
- Temporal Difference Learning
- Function Approximation basics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/grid-world-explorer.git
cd grid-world-explorer

# Install dependencies
pip install pygame numpy matplotlib
```

## Usage

```bash
# Basic usage with default parameters
python main.py

# Specify algorithm and grid size
python main.py --grid_size 15 --algorithm sarsa

# Available algorithms: q_learning, sarsa, monte_carlo
python main.py --algorithm monte_carlo

# Change visualization style
python main.py --render_mode simple
```

## Command Line Arguments

- `--grid_size`: Size of the grid (default: 10)
- `--algorithm`: RL algorithm to use (default: q_learning)
- `--render_mode`: Visualization style (simple or fancy, default: fancy)
- `--episodes`: Number of training episodes (default: 1000)
- `--fps`: Frames per second for visualization (default: 60)

## Controls

- Close the window to exit the simulation

## Understanding the Visualization

- **Blue Circle**: Agent
- **Green Star**: Goal/Reward
- **Brown Blocks**: Obstacles
- **Arrows**: Current policy direction
- **Right Panel**: 
  - Current algorithm information
  - Exploration rate
  - Value function heatmap
  - Remaining goals
