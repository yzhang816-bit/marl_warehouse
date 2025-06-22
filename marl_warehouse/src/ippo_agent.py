"""
Independent Proximal Policy Optimization (IPPO) Agent

This module implements an IPPO agent for multi-agent reinforcement learning
in the warehouse environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for IPPO agent.
    
    The network processes the multi-modal observation space and outputs
    both policy (actor) and value function (critic) estimates.
    """
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Local grid processing (CNN)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Calculate conv output size
        conv_output_size = 32 * 7 * 7  # 32 channels, 7x7 grid
        
        # Agent status and global info processing
        agent_status_dim = 5
        global_info_dim = 6
        
        # Fusion layer
        total_input_dim = conv_output_size + agent_status_dim + global_info_dim
        self.fusion = nn.Linear(total_input_dim, hidden_dim)
        
        # Shared layers
        self.shared_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_dim, action_space.n)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, observation):
        """
        Forward pass through the network.
        
        Args:
            observation: Dictionary containing local_grid, agent_status, global_info
            
        Returns:
            action_logits: Logits for action selection
            value: State value estimate
        """
        # Process local grid with CNN
        local_grid = observation['local_grid']
        if len(local_grid.shape) == 3:
            local_grid = local_grid.unsqueeze(0)  # Add batch dimension
        
        # Transpose to (batch, channels, height, width)
        local_grid = local_grid.permute(0, 3, 1, 2)
        
        x_conv = F.relu(self.conv1(local_grid))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))
        x_conv = x_conv.reshape(x_conv.size(0), -1)  # Flatten
        
        # Process agent status and global info
        agent_status = observation['agent_status']
        global_info = observation['global_info']
        
        if len(agent_status.shape) == 1:
            agent_status = agent_status.unsqueeze(0)
        if len(global_info.shape) == 1:
            global_info = global_info.unsqueeze(0)
        
        # Concatenate all features
        x_combined = torch.cat([x_conv, agent_status, global_info], dim=1)
        
        # Fusion layer
        x = F.relu(self.fusion(x_combined))
        x = self.dropout(x)
        
        # Shared layers
        x = F.relu(self.shared_layer1(x))
        x = self.dropout(x)
        x = F.relu(self.shared_layer2(x))
        
        # Actor and critic heads
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return action_logits, value

class ExperienceBuffer:
    """Experience replay buffer for IPPO training"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class IPPOAgent:
    """
    Independent Proximal Policy Optimization Agent
    
    This agent implements the IPPO algorithm for multi-agent reinforcement learning.
    Each agent learns independently while sharing the same environment.
    """
    
    def __init__(self, 
                 agent_id: int,
                 observation_space,
                 action_space,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize IPPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            observation_space: Observation space from environment
            action_space: Action space from environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
        """
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Neural network
        self.network = ActorCriticNetwork(observation_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience storage
        self.experience_buffer = ExperienceBuffer()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'explained_variance': [],
            'clipfrac': [],
            'approx_kl': []
        }
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        self.network.train()
    
    def select_action(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Select action based on current policy.
        
        Args:
            observation: Current observation
            deterministic: Whether to select action deterministically
            
        Returns:
            action: Selected action
            action_info: Additional information about action selection
        """
        with torch.no_grad():
            # Convert observation to tensors
            obs_tensor = self._observation_to_tensor(observation)
            
            # Forward pass
            action_logits, value = self.network(obs_tensor)
            
            # Create action distribution
            action_dist = Categorical(logits=action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = action_dist.sample()
            
            # Calculate action probability
            action_log_prob = action_dist.log_prob(action)
            
            action_info = {
                'action_log_prob': action_log_prob.item(),
                'value': value.item(),
                'entropy': action_dist.entropy().item()
            }
            
            return action.item(), action_info
    
    def store_experience(self, 
                        observation: Dict[str, np.ndarray],
                        action: int,
                        reward: float,
                        next_observation: Dict[str, np.ndarray],
                        done: bool,
                        action_info: Dict[str, Any]):
        """Store experience in buffer"""
        experience = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done,
            'action_log_prob': action_info['action_log_prob'],
            'value': action_info['value']
        }
        
        self.experience_buffer.push(experience)
        
        # Update episode statistics
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update the agent's policy using PPO.
        
        Args:
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            
        Returns:
            training_metrics: Dictionary of training metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {}
        
        # Sample experiences
        experiences = self.experience_buffer.sample(len(self.experience_buffer))
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(experiences)
        
        # Convert to tensors
        observations = [exp['observation'] for exp in experiences]
        actions = torch.tensor([exp['action'] for exp in experiences], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([exp['action_log_prob'] for exp in experiences], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        clipfracs = []
        approx_kls = []
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(experiences))
            
            for start_idx in range(0, len(experiences), batch_size):
                end_idx = min(start_idx + batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                batch_obs_tensor = self._batch_observations_to_tensor(batch_obs)
                action_logits, values = self.network(batch_obs_tensor)
                
                # Calculate new action probabilities
                action_dist = Categorical(logits=action_logits)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                
                # Calculate clipping statistics
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clipfracs.append(clipfrac)
                    
                    # Approximate KL divergence
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                    approx_kls.append(approx_kl)
        
        # Clear experience buffer
        self.experience_buffer.clear()
        
        # Calculate metrics
        num_updates = num_epochs * (len(experiences) // batch_size + (1 if len(experiences) % batch_size > 0 else 0))
        
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'total_loss': total_loss / num_updates,
            'clipfrac': np.mean(clipfracs),
            'approx_kl': np.mean(approx_kls)
        }
        
        # Update training statistics
        for key, value in metrics.items():
            self.training_stats[key].append(value)
        
        return metrics
    
    def _compute_gae(self, experiences: List[Dict]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates
        """
        advantages = []
        returns = []
        
        # Calculate values for all states
        values = []
        for exp in experiences:
            values.append(exp['value'])
        
        # Add bootstrap value (0 for terminal states)
        next_value = 0
        
        # Calculate advantages using GAE
        gae = 0
        for i in reversed(range(len(experiences))):
            exp = experiences[i]
            
            if i == len(experiences) - 1:
                next_non_terminal = 1.0 - exp['done']
                next_value = 0  # Bootstrap value for last state
            else:
                next_non_terminal = 1.0 - experiences[i + 1]['done']
                next_value = values[i + 1]
            
            delta = exp['reward'] + self.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def _observation_to_tensor(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor format"""
        tensor_obs = {}
        for key, value in observation.items():
            tensor_obs[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        return tensor_obs
    
    def _batch_observations_to_tensor(self, observations: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Convert batch of observations to tensor format"""
        batch_obs = {}
        
        # Stack observations for each key
        for key in observations[0].keys():
            batch_data = np.stack([obs[key] for obs in observations])
            batch_obs[key] = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
        
        return batch_obs
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.training_stats.copy()
    
    def get_episode_stats(self) -> Dict[str, List[float]]:
        """Get episode statistics"""
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy()
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
    
    def set_eval_mode(self):
        """Set network to evaluation mode"""
        self.network.eval()
    
    def set_train_mode(self):
        """Set network to training mode"""
        self.network.train()

