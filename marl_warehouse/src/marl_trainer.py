"""
Multi-Agent Reinforcement Learning Training System

This module coordinates the training of multiple IPPO agents in the warehouse environment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import logging

from warehouse_env import WarehouseEnvironment
from ippo_agent import IPPOAgent

class MARLTrainer:
    """
    Multi-Agent Reinforcement Learning Trainer
    
    Coordinates training of multiple IPPO agents in the warehouse environment.
    """
    
    def __init__(self,
                 env_config: Dict[str, Any] = None,
                 agent_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None,
                 save_dir: str = "experiments"):
        """
        Initialize MARL trainer.
        
        Args:
            env_config: Environment configuration
            agent_config: Agent configuration
            training_config: Training configuration
            save_dir: Directory to save results
        """
        # Default configurations
        self.env_config = env_config or {
            'width': 15,
            'height': 15,
            'num_agents': 3,
            'max_packages': 8,
            'package_spawn_rate': 0.1,
            'max_steps': 500
        }
        
        self.agent_config = agent_config or {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.training_config = training_config or {
            'total_episodes': 1000,
            'update_frequency': 10,  # Update every N episodes
            'batch_size': 64,
            'num_epochs': 4,
            'eval_frequency': 50,  # Evaluate every N episodes
            'save_frequency': 100,  # Save models every N episodes
            'log_frequency': 10   # Log statistics every N episodes
        }
        
        # Create save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize environment and agents
        self.env = WarehouseEnvironment(**self.env_config)
        self.agents = self._create_agents()
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'coordination_scores': [],
            'delivery_rates': [],
            'collision_rates': [],
            'agent_stats': {f'agent_{i}': [] for i in range(self.env_config['num_agents'])}
        }
        
        # Current episode data
        self.current_episode = 0
        self.best_performance = -float('inf')
        
        self.logger.info("MARL Trainer initialized")
        self.logger.info(f"Environment: {self.env_config}")
        self.logger.info(f"Agents: {self.agent_config}")
        self.logger.info(f"Training: {self.training_config}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_agents(self) -> List[IPPOAgent]:
        """Create IPPO agents"""
        agents = []
        for i in range(self.env_config['num_agents']):
            agent = IPPOAgent(
                agent_id=i,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                **self.agent_config
            )
            agents.append(agent)
        return agents
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            training_results: Dictionary containing training results
        """
        self.logger.info("Starting MARL training...")
        
        for episode in range(self.training_config['total_episodes']):
            self.current_episode = episode
            
            # Run episode
            episode_stats = self._run_episode()
            
            # Update training history
            self._update_training_history(episode_stats)
            
            # Update agents
            if episode % self.training_config['update_frequency'] == 0:
                self._update_agents()
            
            # Evaluation
            if episode % self.training_config['eval_frequency'] == 0:
                eval_stats = self._evaluate()
                self.logger.info(f"Episode {episode} - Evaluation: {eval_stats}")
            
            # Logging
            if episode % self.training_config['log_frequency'] == 0:
                self._log_progress(episode, episode_stats)
            
            # Save models
            if episode % self.training_config['save_frequency'] == 0:
                self._save_models(episode)
        
        # Final evaluation and save
        final_eval = self._evaluate()
        self._save_models('final')
        self._save_training_history()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final evaluation: {final_eval}")
        
        return {
            'training_history': self.training_history,
            'final_evaluation': final_eval,
            'best_performance': self.best_performance
        }
    
    def _run_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Returns:
            episode_stats: Statistics from the episode
        """
        observations, _ = self.env.reset()
        episode_rewards = {f'agent_{i}': 0 for i in range(len(self.agents))}
        episode_length = 0
        episode_deliveries = 0
        episode_collisions = 0
        
        done = False
        while not done:
            # Get actions from all agents
            actions = {}
            action_infos = {}
            
            for i, agent in enumerate(self.agents):
                action, action_info = agent.select_action(observations[f'agent_{i}'])
                actions[f'agent_{i}'] = action
                action_infos[f'agent_{i}'] = action_info
            
            # Step environment
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Store experiences for each agent
            for i, agent in enumerate(self.agents):
                agent_name = f'agent_{i}'
                agent.store_experience(
                    observation=observations[agent_name],
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    next_observation=next_observations[agent_name],
                    done=terminated[agent_name] or truncated[agent_name],
                    action_info=action_infos[agent_name]
                )
                
                episode_rewards[agent_name] += rewards[agent_name]
            
            # Update observations
            observations = next_observations
            episode_length += 1
            
            # Track episode statistics
            episode_deliveries = info.get('delivered_packages', 0)
            
            # Check if episode is done
            done = any(terminated.values()) or any(truncated.values())
        
        # Calculate episode statistics
        total_reward = sum(episode_rewards.values())
        avg_reward = total_reward / len(self.agents)
        coordination_score = info.get('coordination_score', 0)
        delivery_rate = episode_deliveries / max(1, info.get('total_packages', 1))
        
        episode_stats = {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'episode_length': episode_length,
            'deliveries': episode_deliveries,
            'coordination_score': coordination_score,
            'delivery_rate': delivery_rate,
            'agent_rewards': episode_rewards,
            'info': info
        }
        
        return episode_stats
    
    def _update_agents(self):
        """Update all agents using their collected experiences"""
        for i, agent in enumerate(self.agents):
            metrics = agent.update(
                batch_size=self.training_config['batch_size'],
                num_epochs=self.training_config['num_epochs']
            )
            
            if metrics:
                self.logger.debug(f"Agent {i} update metrics: {metrics}")
    
    def _evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate current policy performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            evaluation_stats: Average performance statistics
        """
        # Set agents to evaluation mode
        for agent in self.agents:
            agent.set_eval_mode()
        
        eval_rewards = []
        eval_lengths = []
        eval_deliveries = []
        eval_coordination = []
        
        for _ in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                actions = {}
                
                # Get deterministic actions
                for i, agent in enumerate(self.agents):
                    action, _ = agent.select_action(observations[f'agent_{i}'], deterministic=True)
                    actions[f'agent_{i}'] = action
                
                observations, rewards, terminated, truncated, info = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                done = any(terminated.values()) or any(truncated.values())
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_deliveries.append(info.get('delivered_packages', 0))
            eval_coordination.append(info.get('coordination_score', 0))
        
        # Set agents back to training mode
        for agent in self.agents:
            agent.set_train_mode()
        
        eval_stats = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_deliveries': np.mean(eval_deliveries),
            'avg_coordination': np.mean(eval_coordination),
            'delivery_rate': np.mean(eval_deliveries) / max(1, self.env_config['max_packages'])
        }
        
        # Update best performance
        if eval_stats['avg_reward'] > self.best_performance:
            self.best_performance = eval_stats['avg_reward']
            self._save_models('best')
        
        return eval_stats
    
    def _update_training_history(self, episode_stats: Dict[str, Any]):
        """Update training history with episode statistics"""
        self.training_history['episode_rewards'].append(episode_stats['total_reward'])
        self.training_history['episode_lengths'].append(episode_stats['episode_length'])
        self.training_history['coordination_scores'].append(episode_stats['coordination_score'])
        self.training_history['delivery_rates'].append(episode_stats['delivery_rate'])
        
        # Update agent-specific statistics
        for agent_name, reward in episode_stats['agent_rewards'].items():
            if agent_name in self.training_history['agent_stats']:
                self.training_history['agent_stats'][agent_name].append(reward)
    
    def _log_progress(self, episode: int, episode_stats: Dict[str, Any]):
        """Log training progress"""
        recent_rewards = self.training_history['episode_rewards'][-10:]
        recent_coordination = self.training_history['coordination_scores'][-10:]
        recent_delivery_rates = self.training_history['delivery_rates'][-10:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_coordination = np.mean(recent_coordination) if recent_coordination else 0
        avg_delivery_rate = np.mean(recent_delivery_rates) if recent_delivery_rates else 0
        
        self.logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {episode_stats['total_reward']:7.2f} (avg: {avg_reward:7.2f}) | "
            f"Length: {episode_stats['episode_length']:3d} | "
            f"Deliveries: {episode_stats['deliveries']:2d} | "
            f"Coordination: {avg_coordination:.3f} | "
            f"Delivery Rate: {avg_delivery_rate:.3f}"
        )
    
    def _save_models(self, suffix: str):
        """Save agent models"""
        models_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            model_path = os.path.join(models_dir, f'agent_{i}_{suffix}.pth')
            agent.save_model(model_path)
        
        self.logger.info(f"Models saved with suffix: {suffix}")
    
    def _save_training_history(self):
        """Save training history to file"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, dict):
                serializable_history[key] = {k: v for k, v in value.items()}
            else:
                serializable_history[key] = value
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved to: {history_file}")
    
    def plot_training_progress(self, save_plots: bool = True):
        """Plot training progress"""
        if not self.training_history['episode_rewards']:
            self.logger.warning("No training data to plot")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MARL Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'], alpha=0.7)
        axes[0, 0].plot(self._smooth_curve(self.training_history['episode_rewards']), color='red', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Coordination scores
        axes[0, 1].plot(self.training_history['coordination_scores'], alpha=0.7, color='green')
        axes[0, 1].plot(self._smooth_curve(self.training_history['coordination_scores']), color='darkgreen', linewidth=2)
        axes[0, 1].set_title('Coordination Scores')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Coordination Score')
        axes[0, 1].grid(True)
        
        # Delivery rates
        axes[1, 0].plot(self.training_history['delivery_rates'], alpha=0.7, color='orange')
        axes[1, 0].plot(self._smooth_curve(self.training_history['delivery_rates']), color='darkorange', linewidth=2)
        axes[1, 0].set_title('Delivery Rates')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Delivery Rate')
        axes[1, 0].grid(True)
        
        # Episode lengths
        axes[1, 1].plot(self.training_history['episode_lengths'], alpha=0.7, color='purple')
        axes[1, 1].plot(self._smooth_curve(self.training_history['episode_lengths']), color='darkmagenta', linewidth=2)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(self.save_dir, 'training_progress.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to: {plot_file}")
        
        plt.show()
    
    def _smooth_curve(self, data: List[float], window: int = 50) -> List[float]:
        """Apply moving average smoothing to data"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def load_models(self, episode_suffix: str):
        """Load saved models"""
        models_dir = os.path.join(self.save_dir, 'models')
        
        for i, agent in enumerate(self.agents):
            model_path = os.path.join(models_dir, f'agent_{i}_{episode_suffix}.pth')
            if os.path.exists(model_path):
                agent.load_model(model_path)
                self.logger.info(f"Loaded model for agent {i} from: {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
    
    def demonstrate(self, num_episodes: int = 3, render: bool = True) -> List[Dict[str, Any]]:
        """
        Demonstrate trained agents.
        
        Args:
            num_episodes: Number of episodes to demonstrate
            render: Whether to render the environment
            
        Returns:
            demonstration_data: List of episode data for visualization
        """
        self.logger.info(f"Starting demonstration with {num_episodes} episodes...")
        
        # Set agents to evaluation mode
        for agent in self.agents:
            agent.set_eval_mode()
        
        demonstration_data = []
        
        for episode in range(num_episodes):
            self.logger.info(f"Demonstration episode {episode + 1}")
            
            observations, _ = self.env.reset()
            episode_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'info': []
            }
            
            done = False
            step = 0
            
            while not done:
                # Store current state
                episode_data['states'].append(self.env.get_state_dict())
                
                # Get actions
                actions = {}
                for i, agent in enumerate(self.agents):
                    action, _ = agent.select_action(observations[f'agent_{i}'], deterministic=True)
                    actions[f'agent_{i}'] = action
                
                episode_data['actions'].append(actions)
                
                # Step environment
                observations, rewards, terminated, truncated, info = self.env.step(actions)
                
                episode_data['rewards'].append(rewards)
                episode_data['info'].append(info)
                
                if render:
                    self.env.render()
                    print(f"Step {step}: Rewards = {rewards}")
                    print(f"Info: {info}")
                    print("-" * 50)
                
                step += 1
                done = any(terminated.values()) or any(truncated.values())
            
            demonstration_data.append(episode_data)
            
            if render:
                total_reward = sum(sum(rewards.values()) for rewards in episode_data['rewards'])
                final_info = episode_data['info'][-1]
                print(f"\nEpisode {episode + 1} Summary:")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Steps: {step}")
                print(f"Packages Delivered: {final_info.get('delivered_packages', 0)}")
                print(f"Coordination Score: {final_info.get('coordination_score', 0):.3f}")
                print("=" * 50)
        
        # Set agents back to training mode
        for agent in self.agents:
            agent.set_train_mode()
        
        return demonstration_data

