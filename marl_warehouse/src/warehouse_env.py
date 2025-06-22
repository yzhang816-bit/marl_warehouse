"""
Multi-Agent Warehouse Environment for Reinforcement Learning

This module implements a warehouse environment where multiple robotic agents
must coordinate to efficiently collect and deliver packages.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """Available actions for warehouse agents"""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICK_UP = 4
    DROP_OFF = 5
    WAIT = 6

@dataclass
class Package:
    """Represents a package in the warehouse"""
    id: int
    position: Tuple[int, int]
    destination: Tuple[int, int]
    priority: int = 1
    size: int = 1
    created_time: int = 0
    picked_up: bool = False
    delivered: bool = False
    assigned_agent: Optional[int] = None

@dataclass
class Agent:
    """Represents a warehouse robot agent"""
    id: int
    position: Tuple[int, int]
    carrying_package: Optional[Package] = None
    capacity: int = 1
    battery: float = 100.0
    last_action: Optional[ActionType] = None

class WarehouseEnvironment(gym.Env):
    """
    Multi-agent warehouse environment for reinforcement learning.
    
    The environment simulates a warehouse where multiple robotic agents
    must coordinate to efficiently collect and deliver packages.
    """
    
    def __init__(self, 
                 width: int = 15,
                 height: int = 15,
                 num_agents: int = 3,
                 max_packages: int = 8,
                 package_spawn_rate: float = 0.1,
                 max_steps: int = 1000):
        """
        Initialize the warehouse environment.
        
        Args:
            width: Width of the warehouse grid
            height: Height of the warehouse grid
            num_agents: Number of robotic agents
            max_packages: Maximum number of packages in the environment
            package_spawn_rate: Probability of spawning a new package each step
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.max_packages = max_packages
        self.package_spawn_rate = package_spawn_rate
        self.max_steps = max_steps
        
        # Initialize environment state
        self.agents: List[Agent] = []
        self.packages: List[Package] = []
        self.delivery_zones = self._create_delivery_zones()
        self.obstacles = self._create_obstacles()
        self.step_count = 0
        self.package_id_counter = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(ActionType))
        
        # Observation space: local grid view + global information
        self.observation_space = spaces.Dict({
            'local_grid': spaces.Box(
                low=0, high=10, shape=(7, 7, 4), dtype=np.float32
            ),  # 7x7 local view with 4 channels (walls, agents, packages, delivery_zones)
            'agent_status': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            ),  # [x_pos_norm, y_pos_norm, carrying_package, battery_norm, capacity_used]
            'global_info': spaces.Box(
                low=0, high=1, shape=(6,), dtype=np.float32
            )   # [total_packages, delivered_packages, avg_agent_distance, coordination_score, time_norm, efficiency]
        })
        
        self.reset()
    
    def _create_delivery_zones(self) -> List[Tuple[int, int]]:
        """Create delivery zones in the warehouse"""
        zones = []
        # Create delivery zones in corners and center
        zones.extend([
            (1, 1), (1, 2), (2, 1), (2, 2),  # Top-left
            (self.width-3, 1), (self.width-2, 1), (self.width-3, 2), (self.width-2, 2),  # Top-right
            (1, self.height-3), (1, self.height-2), (2, self.height-3), (2, self.height-2),  # Bottom-left
            (self.width-3, self.height-3), (self.width-2, self.height-3), 
            (self.width-3, self.height-2), (self.width-2, self.height-2),  # Bottom-right
        ])
        # Center delivery zone
        center_x, center_y = self.width // 2, self.height // 2
        zones.extend([
            (center_x-1, center_y-1), (center_x, center_y-1), (center_x+1, center_y-1),
            (center_x-1, center_y), (center_x, center_y), (center_x+1, center_y),
            (center_x-1, center_y+1), (center_x, center_y+1), (center_x+1, center_y+1),
        ])
        return zones
    
    def _create_obstacles(self) -> List[Tuple[int, int]]:
        """Create obstacles in the warehouse"""
        obstacles = []
        # Add some random obstacles to create interesting navigation challenges
        for _ in range(self.width * self.height // 20):  # 5% of grid as obstacles
            x = random.randint(3, self.width - 4)
            y = random.randint(3, self.height - 4)
            if (x, y) not in self.delivery_zones:
                obstacles.append((x, y))
        return obstacles
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset agents
        self.agents = []
        for i in range(self.num_agents):
            # Place agents in different corners
            if i == 0:
                pos = (0, 0)
            elif i == 1:
                pos = (self.width - 1, 0)
            elif i == 2:
                pos = (0, self.height - 1)
            else:
                pos = (self.width - 1, self.height - 1)
            
            self.agents.append(Agent(id=i, position=pos))
        
        # Reset packages
        self.packages = []
        self.package_id_counter = 0
        
        # Spawn initial packages
        for _ in range(min(3, self.max_packages)):
            self._spawn_package()
        
        self.step_count = 0
        
        # Return initial observations
        observations = {}
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_observation(i)
        
        info = self._get_info()
        return observations, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary mapping agent names to action indices
            
        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            terminated: Whether each agent's episode is terminated
            truncated: Whether each agent's episode is truncated
            info: Additional information
        """
        self.step_count += 1
        
        # Convert actions to ActionType
        agent_actions = {}
        for agent_name, action_idx in actions.items():
            agent_id = int(agent_name.split('_')[1])
            agent_actions[agent_id] = ActionType(action_idx)
        
        # Execute actions for each agent
        action_results = {}
        for agent_id, action in agent_actions.items():
            action_results[agent_id] = self._execute_action(agent_id, action)
        
        # Handle collisions
        self._resolve_collisions()
        
        # Spawn new packages
        if random.random() < self.package_spawn_rate and len(self.packages) < self.max_packages:
            self._spawn_package()
        
        # Calculate rewards
        rewards = self._calculate_rewards(action_results)
        
        # Get new observations
        observations = {}
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_observation(i)
        
        # Check termination conditions
        terminated = {}
        truncated = {}
        for i in range(self.num_agents):
            terminated[f'agent_{i}'] = False  # No individual termination conditions
            truncated[f'agent_{i}'] = self.step_count >= self.max_steps
        
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _execute_action(self, agent_id: int, action: ActionType) -> Dict[str, Any]:
        """Execute an action for a specific agent"""
        agent = self.agents[agent_id]
        result = {'success': False, 'collision': False, 'pickup': False, 'delivery': False}
        
        agent.last_action = action
        
        if action in [ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]:
            new_pos = self._get_new_position(agent.position, action)
            
            # Check bounds and obstacles
            if self._is_valid_position(new_pos):
                agent.position = new_pos
                result['success'] = True
            else:
                result['collision'] = True
        
        elif action == ActionType.PICK_UP:
            if agent.carrying_package is None:
                # Find package at current position
                for package in self.packages:
                    if (package.position == agent.position and 
                        not package.picked_up and not package.delivered):
                        package.picked_up = True
                        package.assigned_agent = agent_id
                        agent.carrying_package = package
                        result['success'] = True
                        result['pickup'] = True
                        break
        
        elif action == ActionType.DROP_OFF:
            if agent.carrying_package is not None:
                package = agent.carrying_package
                # Check if at delivery zone
                if agent.position in self.delivery_zones:
                    package.delivered = True
                    package.position = agent.position
                    agent.carrying_package = None
                    result['success'] = True
                    result['delivery'] = True
                else:
                    # Drop package at current location
                    package.position = agent.position
                    package.picked_up = False
                    package.assigned_agent = None
                    agent.carrying_package = None
                    result['success'] = True
        
        elif action == ActionType.WAIT:
            result['success'] = True
        
        # Update battery (simplified)
        if action != ActionType.WAIT:
            agent.battery = max(0, agent.battery - 0.1)
        
        return result
    
    def _get_new_position(self, current_pos: Tuple[int, int], action: ActionType) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = current_pos
        
        if action == ActionType.MOVE_UP:
            return (x, max(0, y - 1))
        elif action == ActionType.MOVE_DOWN:
            return (x, min(self.height - 1, y + 1))
        elif action == ActionType.MOVE_LEFT:
            return (max(0, x - 1), y)
        elif action == ActionType.MOVE_RIGHT:
            return (min(self.width - 1, x + 1), y)
        
        return current_pos
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not out of bounds or obstacle)"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check obstacles
        if pos in self.obstacles:
            return False
        
        return True
    
    def _resolve_collisions(self):
        """Handle collisions between agents"""
        positions = {}
        for agent in self.agents:
            pos = agent.position
            if pos in positions:
                # Collision detected - move both agents back to previous valid positions
                # For simplicity, we'll just mark it but not move them back
                pass
            else:
                positions[pos] = agent.id
    
    def _spawn_package(self):
        """Spawn a new package at a random location"""
        if len(self.packages) >= self.max_packages:
            return
        
        # Find valid spawn position (not on agent, obstacle, or delivery zone)
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = (x, y)
            
            # Check if position is free
            if (pos not in self.obstacles and 
                pos not in self.delivery_zones and
                not any(agent.position == pos for agent in self.agents) and
                not any(pkg.position == pos and not pkg.delivered for pkg in self.packages)):
                
                # Choose random delivery zone
                destination = random.choice(self.delivery_zones)
                
                package = Package(
                    id=self.package_id_counter,
                    position=pos,
                    destination=destination,
                    priority=random.randint(1, 3),
                    created_time=self.step_count
                )
                
                self.packages.append(package)
                self.package_id_counter += 1
                break
    
    def _get_observation(self, agent_id: int) -> Dict[str, np.ndarray]:
        """Get observation for a specific agent"""
        agent = self.agents[agent_id]
        
        # Local grid observation (7x7 around agent)
        local_grid = np.zeros((7, 7, 4), dtype=np.float32)
        
        agent_x, agent_y = agent.position
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                world_x = agent_x + dx
                world_y = agent_y + dy
                grid_x = dx + 3
                grid_y = dy + 3
                
                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    pos = (world_x, world_y)
                    
                    # Channel 0: Walls/Obstacles
                    if pos in self.obstacles:
                        local_grid[grid_y, grid_x, 0] = 1.0
                    
                    # Channel 1: Other agents
                    for other_agent in self.agents:
                        if other_agent.id != agent_id and other_agent.position == pos:
                            local_grid[grid_y, grid_x, 1] = 1.0
                    
                    # Channel 2: Packages
                    for package in self.packages:
                        if package.position == pos and not package.delivered:
                            local_grid[grid_y, grid_x, 2] = package.priority / 3.0
                    
                    # Channel 3: Delivery zones
                    if pos in self.delivery_zones:
                        local_grid[grid_y, grid_x, 3] = 1.0
                else:
                    # Out of bounds - treat as wall
                    local_grid[grid_y, grid_x, 0] = 1.0
        
        # Agent status
        agent_status = np.array([
            agent.position[0] / self.width,  # Normalized x position
            agent.position[1] / self.height,  # Normalized y position
            1.0 if agent.carrying_package else 0.0,  # Carrying package
            agent.battery / 100.0,  # Normalized battery
            len([p for p in self.packages if p.assigned_agent == agent_id]) / agent.capacity  # Capacity used
        ], dtype=np.float32)
        
        # Global information
        total_packages = len(self.packages)
        delivered_packages = len([p for p in self.packages if p.delivered])
        avg_agent_distance = self._calculate_average_agent_distance()
        coordination_score = self._calculate_coordination_score()
        time_norm = self.step_count / self.max_steps
        efficiency = delivered_packages / max(1, total_packages)
        
        global_info = np.array([
            total_packages / self.max_packages,
            delivered_packages / max(1, total_packages),
            avg_agent_distance,
            coordination_score,
            time_norm,
            efficiency
        ], dtype=np.float32)
        
        return {
            'local_grid': local_grid,
            'agent_status': agent_status,
            'global_info': global_info
        }
    
    def _calculate_average_agent_distance(self) -> float:
        """Calculate average distance between agents (normalized)"""
        if len(self.agents) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]
                distance = abs(agent1.position[0] - agent2.position[0]) + abs(agent1.position[1] - agent2.position[1])
                total_distance += distance
                count += 1
        
        max_distance = self.width + self.height
        return (total_distance / count) / max_distance if count > 0 else 0.0
    
    def _calculate_coordination_score(self) -> float:
        """Calculate a coordination score based on agent behaviors"""
        # Simple coordination metric: how well agents distribute across the warehouse
        occupied_zones = set()
        zone_size = 3
        
        for agent in self.agents:
            zone_x = agent.position[0] // zone_size
            zone_y = agent.position[1] // zone_size
            occupied_zones.add((zone_x, zone_y))
        
        max_zones = (self.width // zone_size + 1) * (self.height // zone_size + 1)
        return len(occupied_zones) / max_zones
    
    def _calculate_rewards(self, action_results: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        
        # Global metrics for shared rewards
        total_delivered = len([p for p in self.packages if p.delivered])
        total_packages = len(self.packages)
        
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            result = action_results.get(agent_id, {})
            reward = 0.0
            
            # Individual rewards
            if result.get('delivery', False):
                reward += 10.0  # Large reward for successful delivery
            
            if result.get('pickup', False):
                reward += 2.0  # Reward for picking up packages
            
            if result.get('collision', False):
                reward -= 1.0  # Penalty for collisions
            
            # Efficiency rewards
            if agent.carrying_package:
                # Reward for moving towards delivery zone while carrying package
                package = agent.carrying_package
                min_distance_to_delivery = min(
                    abs(agent.position[0] - zone[0]) + abs(agent.position[1] - zone[1])
                    for zone in self.delivery_zones
                )
                reward += 0.1 * (1.0 / (min_distance_to_delivery + 1))
            else:
                # Reward for moving towards packages when not carrying
                if self.packages:
                    available_packages = [p for p in self.packages if not p.picked_up and not p.delivered]
                    if available_packages:
                        min_distance_to_package = min(
                            abs(agent.position[0] - pkg.position[0]) + abs(agent.position[1] - pkg.position[1])
                            for pkg in available_packages
                        )
                        reward += 0.05 * (1.0 / (min_distance_to_package + 1))
            
            # Coordination rewards (shared)
            coordination_bonus = 0.0
            if total_packages > 0:
                efficiency = total_delivered / total_packages
                coordination_bonus = efficiency * 2.0
            
            # Time penalty to encourage efficiency
            reward -= 0.01
            
            # Battery penalty
            if agent.battery < 20:
                reward -= 0.5
            
            rewards[f'agent_{agent_id}'] = reward + coordination_bonus
        
        return rewards
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state"""
        return {
            'step_count': self.step_count,
            'total_packages': len(self.packages),
            'delivered_packages': len([p for p in self.packages if p.delivered]),
            'packages_in_transit': len([p for p in self.packages if p.picked_up and not p.delivered]),
            'agent_positions': [agent.position for agent in self.agents],
            'agent_carrying': [agent.carrying_package is not None for agent in self.agents],
            'coordination_score': self._calculate_coordination_score(),
            'average_agent_distance': self._calculate_average_agent_distance()
        }
    
    def render(self, mode='human'):
        """Render the environment (basic text representation)"""
        if mode == 'human':
            grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
            
            # Add obstacles
            for x, y in self.obstacles:
                grid[y][x] = '#'
            
            # Add delivery zones
            for x, y in self.delivery_zones:
                grid[y][x] = 'D'
            
            # Add packages
            for package in self.packages:
                if not package.delivered and not package.picked_up:
                    x, y = package.position
                    grid[y][x] = 'P'
            
            # Add agents
            for agent in self.agents:
                x, y = agent.position
                if agent.carrying_package:
                    grid[y][x] = f'A{agent.id}*'
                else:
                    grid[y][x] = f'A{agent.id}'
            
            # Print grid
            print(f"\nStep {self.step_count}")
            print("=" * (self.width * 4))
            for row in grid:
                print(" ".join(f"{cell:>3}" for cell in row))
            print("=" * (self.width * 4))
            
            # Print status
            print(f"Packages: {len([p for p in self.packages if not p.delivered])}/{len(self.packages)} remaining")
            print(f"Delivered: {len([p for p in self.packages if p.delivered])}")
            for agent in self.agents:
                carrying = "carrying package" if agent.carrying_package else "empty"
                print(f"Agent {agent.id}: {agent.position} ({carrying}, battery: {agent.battery:.1f}%)")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete environment state for visualization"""
        return {
            'width': self.width,
            'height': self.height,
            'step_count': self.step_count,
            'agents': [
                {
                    'id': agent.id,
                    'position': agent.position,
                    'carrying_package': agent.carrying_package.id if agent.carrying_package else None,
                    'battery': agent.battery,
                    'last_action': agent.last_action.name if agent.last_action else None
                }
                for agent in self.agents
            ],
            'packages': [
                {
                    'id': package.id,
                    'position': package.position,
                    'destination': package.destination,
                    'priority': package.priority,
                    'picked_up': package.picked_up,
                    'delivered': package.delivered,
                    'assigned_agent': package.assigned_agent
                }
                for package in self.packages
            ],
            'delivery_zones': self.delivery_zones,
            'obstacles': self.obstacles,
            'info': self._get_info()
        }

