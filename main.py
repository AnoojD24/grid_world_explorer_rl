import pygame
import time
import argparse
import numpy as np
from environment import GridWorldEnv
from agent import Agent
from visualization import Visualization

def parse_args():
    parser = argparse.ArgumentParser(description='Grid World Explorer')
    parser.add_argument('--grid_size', type=int, default=10, help='Size of grid')
    parser.add_argument('--algorithm', type=str, default='q_learning', 
                        choices=['q_learning', 'sarsa', 'monte_carlo'],
                        help='RL algorithm to use')
    parser.add_argument('--render_mode', type=str, default='fancy',
                        choices=['simple', 'fancy'], help='Visualization style')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for visualization')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize environment and agent
    env = GridWorldEnv(grid_size=args.grid_size)
    agent = Agent(env.observation_space, env.action_space, algorithm=args.algorithm)
    
    # Initialize visualization
    viz = Visualization(env, render_mode=args.render_mode, fps=args.fps)
    
    # Lists to store training history
    episode_rewards = []
    episode_lengths = []
    exploration_rates = []
    
    # Create run-specific directory for outputs
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('data', f'grid{args.grid_size}_algo{args.algorithm}_ep{args.episodes}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save run parameters
    with open(os.path.join(run_dir, 'parameters.txt'), 'w') as f:
        f.write(f"Grid Size: {args.grid_size}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Render Mode: {args.render_mode}\n")
        f.write(f"FPS: {args.fps}\n")
    
    # Training loop
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render environment
            viz.render(agent)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        exploration_rates.append(agent.exploration_rate)
        
        print(f"Episode {episode+1}/{args.episodes}, Total Reward: {total_reward}, Steps: {steps}, ε: {agent.exploration_rate:.3f}")
        
        # Display episode summary
        viz.display_episode_summary(episode, total_reward, agent)
        time.sleep(1)  # Short pause between episodes
        
        # Create learning curves plot every 5 episodes
        if (episode + 1) % 5 == 0:
            from utils import plot_learning_curve, plot_value_function, create_policy_visualization
            
            # Plot learning curves
            plot_learning_curve(
                episode_rewards, 
                episode_lengths, 
                exploration_rates, 
                os.path.join(run_dir, f'learning_curve_episode_{episode+1}.png')
            )
            
            # Plot value function for current state
            value_grid = np.zeros((env.grid_size, env.grid_size))
            for y in range(env.grid_size):
                for x in range(env.grid_size):
                    state = env._get_state().copy()
                    if state[y, x] != -1:  # Skip obstacles
                        agent_y, agent_x = np.where(state == 2)
                        if len(agent_y) > 0:
                            state[agent_y[0], agent_x[0]] = 0  # Clear current agent position
                        state[y, x] = 2  # Place agent at (y,x)
                        value_grid[y, x] = agent.get_state_value(state)
            plot_value_function(
                value_grid, 
                os.path.join(run_dir, f'value_function_episode_{episode+1}.png')
            )
            
            # Plot policy
            create_policy_visualization(
                agent.policy, 
                os.path.join(run_dir, f'policy_episode_{episode+1}.png')
            )
            
            # Save statistics as CSV
            import pandas as pd
            stats_df = pd.DataFrame({
                'episode': range(1, len(episode_rewards) + 1),
                'reward': episode_rewards,
                'steps': episode_lengths,
                'exploration_rate': exploration_rates
            })
            stats_df.to_csv(os.path.join(run_dir, 'training_stats.csv'), index=False)
        
    pygame.quit()

if __name__ == "__main__":
    main()