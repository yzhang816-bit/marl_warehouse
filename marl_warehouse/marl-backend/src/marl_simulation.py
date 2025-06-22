"""
Simplified MARL simulation for web backend
This is a lightweight version of the warehouse environment for real-time web simulation
"""

import numpy as np
import random
import time
import json
from typing import Dict, List, Tuple, Any

class SimpleWarehouseEnvironment:
    """Simplified warehouse environment for web simulation"""
    
    def __init__(self, width=15, height=15, num_agents=3, max_packages=5):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.max_packages = max_packages
        
        # Environment state
        self.agents = []
        self.packages = []
        self.delivery_zones = []
        self.obstacles = []
        self.step_count = 0
        self.episode_count = 0
        
        # Performance metrics
        self.total_reward = 0
        self.delivered_packages = 0
        self.coordination_score = 0.5
        self.efficiency = 0.0
        
        # Initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize the warehouse environment"""
        # Create agents
        self.agents = []
        agent_positions = [(0, 0), (self.width-1, 0), (0, self.height-1)]
        for i in range(self.num_agents):
            self.agents.append({
                'id': i,
                'position': list(agent_positions[i]) if i < len(agent_positions) else [random.randint(0, self.width-1), random.randint(0, self.height-1)],
                'carrying': None,
                'battery': 100.0,
                'lastAction': None,
                'reward': 0.0
            })
        
        # Create delivery zones
        self.delivery_zones = [
            [1, 1], [1, 2], [2, 1], [2, 2],  # Top-left
            [self.width-2, 1], [self.width-2, 2], [self.width-1, 1], [self.width-1, 2],  # Top-right
            [1, self.height-2], [1, self.height-1], [2, self.height-2], [2, self.height-1],  # Bottom-left
            [self.width-2, self.height-2], [self.width-2, self.height-1], [self.width-1, self.height-2], [self.width-1, self.height-1],  # Bottom-right
            # Central delivery zone
            [self.width//2-1, self.height//2-1], [self.width//2, self.height//2-1], [self.width//2+1, self.height//2-1],
            [self.width//2-1, self.height//2], [self.width//2, self.height//2], [self.width//2+1, self.height//2],
            [self.width//2-1, self.height//2+1], [self.width//2, self.height//2+1], [self.width//2+1, self.height//2+1]
        ]
        
        # Create obstacles
        self.obstacles = [
            [5, 3], [9, 6], [11, 9], [4, 12], [8, 2],
            [3, 7], [12, 4], [6, 11], [10, 13], [2, 5]
        ]
        
        # Create packages
        self._spawn_packages()
    
    def _spawn_packages(self):
        """Spawn packages in the environment"""
        self.packages = []
        for i in range(min(self.max_packages, 5)):
            # Find valid position for package
            while True:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                
                # Check if position is valid (not on agent, delivery zone, or obstacle)
                valid = True
                for agent in self.agents:
                    if agent['position'] == [x, y]:
                        valid = False
                        break
                
                if [x, y] in self.delivery_zones or [x, y] in self.obstacles:
                    valid = False
                
                if valid:
                    break
            
            self.packages.append({
                'id': i,
                'position': [x, y],
                'destination': random.choice(self.delivery_zones),
                'priority': random.randint(1, 3),
                'delivered': False
            })
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step"""
        self.step_count += 1
        
        # Simple agent behavior simulation
        for agent in self.agents:
            self._update_agent(agent)
        
        # Update metrics
        self._update_metrics()
        
        # Return current state
        return self.get_state()
    
    def _update_agent(self, agent: Dict[str, Any]):
        """Update single agent state"""
        # Decrease battery
        agent['battery'] = max(0, agent['battery'] - 0.1)
        
        if agent['battery'] <= 0:
            agent['lastAction'] = 'RECHARGE'
            return
        
        # Simple AI behavior
        if agent['carrying'] is None:
            # Look for nearest package
            nearest_package = self._find_nearest_package(agent['position'])
            if nearest_package:
                # Move towards package
                if self._move_towards_target(agent, nearest_package['position']):
                    # Try to pick up package if adjacent
                    if self._is_adjacent(agent['position'], nearest_package['position']):
                        if random.random() < 0.3:  # 30% chance to pick up
                            agent['carrying'] = nearest_package['id']
                            agent['lastAction'] = 'PICKUP'
                            agent['reward'] += 5
                            return
                
                agent['lastAction'] = 'MOVE'
            else:
                # Random movement if no packages
                self._random_move(agent)
                agent['lastAction'] = 'MOVE'
        else:
            # Carrying package - move to delivery zone
            nearest_delivery = self._find_nearest_delivery_zone(agent['position'])
            if self._move_towards_target(agent, nearest_delivery):
                # Try to deliver if at delivery zone
                if agent['position'] in self.delivery_zones:
                    if random.random() < 0.4:  # 40% chance to deliver
                        carried_package = next((p for p in self.packages if p['id'] == agent['carrying']), None)
                        if carried_package:
                            carried_package['delivered'] = True
                            carried_package['position'] = agent['position'][:]
                            self.delivered_packages += 1
                        
                        agent['carrying'] = None
                        agent['lastAction'] = 'DELIVER'
                        agent['reward'] += 10
                        return
                
                agent['lastAction'] = 'MOVE'
    
    def _find_nearest_package(self, position: List[int]) -> Dict[str, Any]:
        """Find nearest undelivered package"""
        available_packages = [p for p in self.packages if not p['delivered'] and not any(a['carrying'] == p['id'] for a in self.agents)]
        if not available_packages:
            return None
        
        return min(available_packages, key=lambda p: self._manhattan_distance(position, p['position']))
    
    def _find_nearest_delivery_zone(self, position: List[int]) -> List[int]:
        """Find nearest delivery zone"""
        return min(self.delivery_zones, key=lambda d: self._manhattan_distance(position, d))
    
    def _manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_adjacent(self, pos1: List[int], pos2: List[int]) -> bool:
        """Check if two positions are adjacent"""
        return self._manhattan_distance(pos1, pos2) <= 1
    
    def _move_towards_target(self, agent: Dict[str, Any], target: List[int]) -> bool:
        """Move agent towards target position"""
        current = agent['position']
        
        # Calculate direction
        dx = 0 if current[0] == target[0] else (1 if target[0] > current[0] else -1)
        dy = 0 if current[1] == target[1] else (1 if target[1] > current[1] else -1)
        
        # Choose random direction if multiple options
        if dx != 0 and dy != 0:
            if random.random() < 0.5:
                dx = 0
            else:
                dy = 0
        
        # Try to move
        new_x = current[0] + dx
        new_y = current[1] + dy
        
        if self._is_valid_position(new_x, new_y):
            agent['position'] = [new_x, new_y]
            return True
        
        return False
    
    def _random_move(self, agent: Dict[str, Any]):
        """Move agent randomly"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]  # Include staying still
        dx, dy = random.choice(directions)
        
        new_x = agent['position'][0] + dx
        new_y = agent['position'][1] + dy
        
        if self._is_valid_position(new_x, new_y):
            agent['position'] = [new_x, new_y]
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        if [x, y] in self.obstacles:
            return False
        
        # Check if another agent is at this position
        for agent in self.agents:
            if agent['position'] == [x, y]:
                return False
        
        return True
    
    def _update_metrics(self):
        """Update performance metrics"""
        # Update total reward
        self.total_reward = sum(agent['reward'] for agent in self.agents)
        
        # Update coordination score (based on agent distances)
        if len(self.agents) > 1:
            distances = []
            for i, agent1 in enumerate(self.agents):
                for agent2 in self.agents[i+1:]:
                    dist = self._manhattan_distance(agent1['position'], agent2['position'])
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            max_distance = np.sqrt(self.width**2 + self.height**2)
            self.coordination_score = max(0, 1 - (avg_distance / max_distance))
        
        # Update efficiency (packages delivered per step)
        if self.step_count > 0:
            self.efficiency = min(1.0, self.delivered_packages / (self.step_count / 50))
    
    def reset(self):
        """Reset the environment"""
        self.step_count = 0
        self.episode_count += 1
        self.total_reward = 0
        self.delivered_packages = 0
        self.coordination_score = 0.5
        self.efficiency = 0.0
        
        self._initialize_environment()
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            'step': self.step_count,
            'episode': self.episode_count,
            'warehouse': {
                'width': self.width,
                'height': self.height,
                'agents': self.agents,
                'packages': self.packages,
                'deliveryZones': self.delivery_zones,
                'obstacles': self.obstacles
            },
            'metrics': {
                'totalReward': self.total_reward,
                'deliveredPackages': self.delivered_packages,
                'coordinationScore': self.coordination_score,
                'efficiency': self.efficiency,
                'averageDistance': 1 - self.coordination_score
            }
        }

class MARLSimulationManager:
    """Manages MARL simulation sessions"""
    
    def __init__(self):
        self.environments = {}  # session_id -> environment
        self.training_sessions = {}  # session_id -> training_state
    
    def create_session(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new simulation session"""
        env = SimpleWarehouseEnvironment(
            width=config.get('width', 15),
            height=config.get('height', 15),
            num_agents=config.get('num_agents', 3),
            max_packages=config.get('max_packages', 5)
        )
        
        self.environments[session_id] = env
        self.training_sessions[session_id] = {
            'is_running': False,
            'is_training': False,
            'config': config
        }
        
        return env.get_state()
    
    def step_simulation(self, session_id: str) -> Dict[str, Any]:
        """Step simulation for session"""
        if session_id not in self.environments:
            raise ValueError(f"Session {session_id} not found")
        
        env = self.environments[session_id]
        return env.step()
    
    def reset_simulation(self, session_id: str) -> Dict[str, Any]:
        """Reset simulation for session"""
        if session_id not in self.environments:
            raise ValueError(f"Session {session_id} not found")
        
        env = self.environments[session_id]
        return env.reset()
    
    def get_state(self, session_id: str) -> Dict[str, Any]:
        """Get current state for session"""
        if session_id not in self.environments:
            raise ValueError(f"Session {session_id} not found")
        
        env = self.environments[session_id]
        return env.get_state()
    
    def update_config(self, session_id: str, config: Dict[str, Any]):
        """Update simulation configuration"""
        if session_id in self.training_sessions:
            self.training_sessions[session_id]['config'].update(config)
    
    def set_running(self, session_id: str, is_running: bool):
        """Set simulation running state"""
        if session_id in self.training_sessions:
            self.training_sessions[session_id]['is_running'] = is_running
    
    def set_training(self, session_id: str, is_training: bool):
        """Set training state"""
        if session_id in self.training_sessions:
            self.training_sessions[session_id]['is_training'] = is_training
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.environments:
            del self.environments[session_id]
        if session_id in self.training_sessions:
            del self.training_sessions[session_id]

# Global simulation manager
simulation_manager = MARLSimulationManager()

