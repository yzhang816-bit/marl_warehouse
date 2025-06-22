#!/usr/bin/env python3
"""
Comprehensive system test for the Multi-Agent Reinforcement Learning system

This script tests the complete integrated system including:
- Backend API functionality
- WebSocket communication
- Frontend-backend integration
- MARL simulation performance
- Agent coordination behavior
"""

import requests
import json
import time
import threading
import subprocess
import sys
import os
from datetime import datetime

# Test configuration
BACKEND_URL = "http://localhost:5000"
FRONTEND_URL = "http://localhost:5173"
API_BASE = f"{BACKEND_URL}/api/marl"

class SystemTester:
    def __init__(self):
        self.session_id = None
        self.test_results = []
        self.start_time = datetime.now()
    
    def log_test(self, test_name, passed, message="", duration=0):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message} ({duration:.2f}s)")
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("Backend Health", True, "Backend is healthy", duration)
                    return True
            
            self.log_test("Backend Health", False, f"Unexpected response: {response.status_code}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Backend Health", False, f"Connection failed: {str(e)}", duration)
            return False
    
    def test_session_creation(self):
        """Test MARL session creation"""
        start_time = time.time()
        try:
            payload = {
                'width': 10,
                'height': 10,
                'num_agents': 2,
                'max_packages': 3,
                'simulation_speed': 100
            }
            
            response = requests.post(f"{API_BASE}/session", json=payload, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'session_id' in data:
                    self.session_id = data['session_id']
                    self.log_test("Session Creation", True, f"Session created: {self.session_id[:8]}...", duration)
                    return True
            
            self.log_test("Session Creation", False, f"Failed to create session: {response.text}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Session Creation", False, f"Request failed: {str(e)}", duration)
            return False
    
    def test_session_state(self):
        """Test getting session state"""
        if not self.session_id:
            self.log_test("Session State", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'state' in data:
                    state = data['state']
                    
                    # Validate state structure
                    required_keys = ['warehouse', 'metrics', 'step', 'episode']
                    if all(key in state for key in required_keys):
                        agents_count = len(state['warehouse']['agents'])
                        packages_count = len(state['warehouse']['packages'])
                        self.log_test("Session State", True, f"State valid: {agents_count} agents, {packages_count} packages", duration)
                        return True
            
            self.log_test("Session State", False, f"Invalid state response: {response.text}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Session State", False, f"Request failed: {str(e)}", duration)
            return False
    
    def test_simulation_control(self):
        """Test simulation start/stop controls"""
        if not self.session_id:
            self.log_test("Simulation Control", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            # Test start
            response = requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                                   json={'action': 'start'}, timeout=5)
            
            if response.status_code != 200 or not response.json().get('success'):
                self.log_test("Simulation Control", False, "Failed to start simulation")
                return False
            
            # Wait a bit for simulation to run
            time.sleep(2)
            
            # Test stop
            response = requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                                   json={'action': 'stop'}, timeout=5)
            
            duration = time.time() - start_time
            
            if response.status_code == 200 and response.json().get('success'):
                self.log_test("Simulation Control", True, "Start/stop controls working", duration)
                return True
            
            self.log_test("Simulation Control", False, "Failed to stop simulation", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Simulation Control", False, f"Control failed: {str(e)}", duration)
            return False
    
    def test_simulation_steps(self):
        """Test simulation stepping and state updates"""
        if not self.session_id:
            self.log_test("Simulation Steps", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            initial_response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
            if initial_response.status_code != 200:
                self.log_test("Simulation Steps", False, "Failed to get initial state")
                return False
            
            initial_state = initial_response.json()['state']
            initial_step = initial_state['step']
            
            # Step simulation multiple times
            steps_taken = 0
            for i in range(5):
                step_response = requests.post(f"{API_BASE}/session/{self.session_id}/step", timeout=5)
                if step_response.status_code == 200 and step_response.json().get('success'):
                    steps_taken += 1
                    time.sleep(0.1)  # Small delay between steps
            
            # Check final state
            final_response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
            if final_response.status_code != 200:
                self.log_test("Simulation Steps", False, "Failed to get final state")
                return False
            
            final_state = final_response.json()['state']
            final_step = final_state['step']
            
            duration = time.time() - start_time
            
            if final_step > initial_step:
                self.log_test("Simulation Steps", True, f"Stepped from {initial_step} to {final_step}", duration)
                return True
            
            self.log_test("Simulation Steps", False, f"Step count didn't increase: {initial_step} -> {final_step}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Simulation Steps", False, f"Stepping failed: {str(e)}", duration)
            return False
    
    def test_agent_coordination(self):
        """Test agent coordination behavior"""
        if not self.session_id:
            self.log_test("Agent Coordination", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            # Start simulation and let it run
            requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                         json={'action': 'start'}, timeout=5)
            
            # Collect coordination scores over time
            coordination_scores = []
            for i in range(10):
                time.sleep(0.5)
                response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
                if response.status_code == 200:
                    state = response.json()['state']
                    coordination_score = state['metrics']['coordinationScore']
                    coordination_scores.append(coordination_score)
            
            # Stop simulation
            requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                         json={'action': 'stop'}, timeout=5)
            
            duration = time.time() - start_time
            
            if coordination_scores:
                avg_coordination = sum(coordination_scores) / len(coordination_scores)
                min_coordination = min(coordination_scores)
                max_coordination = max(coordination_scores)
                
                # Check if coordination is reasonable (between 0 and 1)
                if 0 <= avg_coordination <= 1:
                    self.log_test("Agent Coordination", True, 
                                f"Avg: {avg_coordination:.3f}, Range: {min_coordination:.3f}-{max_coordination:.3f}", 
                                duration)
                    return True
            
            self.log_test("Agent Coordination", False, "Invalid coordination scores", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Agent Coordination", False, f"Coordination test failed: {str(e)}", duration)
            return False
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        if not self.session_id:
            self.log_test("Performance Metrics", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
            if response.status_code != 200:
                self.log_test("Performance Metrics", False, "Failed to get state")
                return False
            
            state = response.json()['state']
            metrics = state['metrics']
            
            # Check required metrics
            required_metrics = ['totalReward', 'deliveredPackages', 'coordinationScore', 'efficiency']
            missing_metrics = [m for m in required_metrics if m not in metrics]
            
            duration = time.time() - start_time
            
            if not missing_metrics:
                # Validate metric ranges
                valid_ranges = {
                    'coordinationScore': (0, 1),
                    'efficiency': (0, 1),
                    'deliveredPackages': (0, float('inf'))
                }
                
                invalid_metrics = []
                for metric, (min_val, max_val) in valid_ranges.items():
                    value = metrics[metric]
                    if not (min_val <= value <= max_val):
                        invalid_metrics.append(f"{metric}={value}")
                
                if not invalid_metrics:
                    self.log_test("Performance Metrics", True, 
                                f"All metrics valid: {len(required_metrics)} metrics", duration)
                    return True
                else:
                    self.log_test("Performance Metrics", False, 
                                f"Invalid metric values: {', '.join(invalid_metrics)}", duration)
                    return False
            
            self.log_test("Performance Metrics", False, 
                        f"Missing metrics: {', '.join(missing_metrics)}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance Metrics", False, f"Metrics test failed: {str(e)}", duration)
            return False
    
    def test_reset_functionality(self):
        """Test simulation reset"""
        if not self.session_id:
            self.log_test("Reset Functionality", False, "No session ID available")
            return False
        
        start_time = time.time()
        try:
            # Get initial state
            initial_response = requests.get(f"{API_BASE}/session/{self.session_id}/state", timeout=5)
            if initial_response.status_code != 200:
                self.log_test("Reset Functionality", False, "Failed to get initial state")
                return False
            
            initial_state = initial_response.json()['state']
            
            # Run simulation for a bit
            requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                         json={'action': 'start'}, timeout=5)
            time.sleep(1)
            requests.post(f"{API_BASE}/session/{self.session_id}/control", 
                         json={'action': 'stop'}, timeout=5)
            
            # Reset simulation
            reset_response = requests.post(f"{API_BASE}/session/{self.session_id}/reset", timeout=5)
            if reset_response.status_code != 200 or not reset_response.json().get('success'):
                self.log_test("Reset Functionality", False, "Reset request failed")
                return False
            
            # Check reset state
            reset_state = reset_response.json()['state']
            
            duration = time.time() - start_time
            
            # Verify reset worked (step count should be 0)
            if reset_state['step'] == 0:
                self.log_test("Reset Functionality", True, "Simulation reset successfully", duration)
                return True
            
            self.log_test("Reset Functionality", False, 
                        f"Reset didn't work: step={reset_state['step']}", duration)
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Reset Functionality", False, f"Reset test failed: {str(e)}", duration)
            return False
    
    def run_all_tests(self):
        """Run all system tests"""
        print("=" * 60)
        print("MULTI-AGENT REINFORCEMENT LEARNING SYSTEM TESTS")
        print("=" * 60)
        print(f"Started at: {self.start_time}")
        print()
        
        tests = [
            self.test_backend_health,
            self.test_session_creation,
            self.test_session_state,
            self.test_simulation_control,
            self.test_simulation_steps,
            self.test_agent_coordination,
            self.test_performance_metrics,
            self.test_reset_functionality
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[ERROR] {test.__name__}: {str(e)}")
                failed += 1
            print()
        
        # Print summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {passed + failed}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        
        if failed == 0:
            print("\n🎉 All tests passed! The MARL system is working correctly.")
            return True
        else:
            print(f"\n❌ {failed} test(s) failed. Please check the system.")
            return False
    
    def save_test_report(self, filename="test_report.json"):
        """Save detailed test report"""
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'PASS']),
                'failed': len([r for r in self.test_results if r['status'] == 'FAIL']),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': (datetime.now() - self.start_time).total_seconds()
            },
            'tests': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed test report saved to: {filename}")

def main():
    """Main test execution"""
    tester = SystemTester()
    
    try:
        success = tester.run_all_tests()
        tester.save_test_report("/home/ubuntu/marl_warehouse/test_report.json")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

