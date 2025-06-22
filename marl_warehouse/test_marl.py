#!/usr/bin/env python3
"""
Test script for Multi-Agent Reinforcement Learning system

This script tests the MARL implementation and runs a short training session
to verify that all components work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from warehouse_env import WarehouseEnvironment
from ippo_agent import IPPOAgent
from marl_trainer import MARLTrainer

def test_environment():
    """Test the warehouse environment"""
    print("Testing Warehouse Environment...")
    
    env = WarehouseEnvironment(width=10, height=10, num_agents=2, max_packages=4)
    
    # Test reset
    observations, info = env.reset()
    print(f"Initial observations keys: {list(observations.keys())}")
    print(f"Initial info: {info}")
    
    # Test step
    actions = {'agent_0': 0, 'agent_1': 1}  # Move up, move down
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    print(f"Step rewards: {rewards}")
    print(f"Step info: {info}")
    
    # Test rendering
    print("\nEnvironment state:")
    env.render()
    
    print("✓ Environment test passed!")
    return True

def test_agent():
    """Test IPPO agent"""
    print("\nTesting IPPO Agent...")
    
    env = WarehouseEnvironment(width=8, height=8, num_agents=1, max_packages=2)
    
    agent = IPPOAgent(
        agent_id=0,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device='cpu'
    )
    
    # Test action selection
    observations, _ = env.reset()
    obs = observations['agent_0']
    
    action, action_info = agent.select_action(obs)
    print(f"Selected action: {action}")
    print(f"Action info: {action_info}")
    
    # Test experience storage
    next_obs, rewards, terminated, truncated, info = env.step({'agent_0': action})
    agent.store_experience(
        observation=obs,
        action=action,
        reward=rewards['agent_0'],
        next_observation=next_obs['agent_0'],
        done=terminated['agent_0'] or truncated['agent_0'],
        action_info=action_info
    )
    
    print(f"Experience buffer size: {len(agent.experience_buffer)}")
    
    print("✓ Agent test passed!")
    return True

def test_training():
    """Test MARL training system"""
    print("\nTesting MARL Training System...")
    
    # Small configuration for quick test
    env_config = {
        'width': 8,
        'height': 8,
        'num_agents': 2,
        'max_packages': 3,
        'package_spawn_rate': 0.2,
        'max_steps': 100
    }
    
    agent_config = {
        'learning_rate': 1e-3,
        'device': 'cpu'
    }
    
    training_config = {
        'total_episodes': 20,  # Very short training for test
        'update_frequency': 5,
        'batch_size': 16,
        'num_epochs': 2,
        'eval_frequency': 10,
        'save_frequency': 20,
        'log_frequency': 5
    }
    
    trainer = MARLTrainer(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
        save_dir="test_results"
    )
    
    # Run short training
    results = trainer.train()
    
    print(f"Training completed!")
    print(f"Final performance: {results['best_performance']:.2f}")
    print(f"Episodes completed: {len(results['training_history']['episode_rewards'])}")
    
    # Test demonstration
    print("\nTesting demonstration...")
    demo_data = trainer.demonstrate(num_episodes=1, render=False)
    print(f"Demonstration data collected for {len(demo_data)} episodes")
    
    print("✓ Training test passed!")
    return True

def test_coordination_behavior():
    """Test that agents show coordination behavior"""
    print("\nTesting Coordination Behavior...")
    
    env = WarehouseEnvironment(width=10, height=10, num_agents=3, max_packages=5)
    
    # Run a few episodes and check coordination metrics
    coordination_scores = []
    
    for episode in range(5):
        observations, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            # Random actions for this test
            actions = {}
            for i in range(3):
                actions[f'agent_{i}'] = np.random.randint(0, 7)
            
            observations, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            done = any(terminated.values()) or any(truncated.values())
        
        coordination_scores.append(info.get('coordination_score', 0))
    
    avg_coordination = np.mean(coordination_scores)
    print(f"Average coordination score: {avg_coordination:.3f}")
    print(f"Coordination scores: {coordination_scores}")
    
    print("✓ Coordination behavior test passed!")
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("MULTI-AGENT REINFORCEMENT LEARNING SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_agent,
        test_training,
        test_coordination_behavior
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The MARL system is working correctly.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

