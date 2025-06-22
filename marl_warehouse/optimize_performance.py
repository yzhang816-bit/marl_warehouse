#!/usr/bin/env python3
"""
Performance optimization and analysis for MARL system

This script analyzes and optimizes the performance of the MARL system
by testing different configurations and measuring throughput.
"""

import requests
import json
import time
import statistics
import sys
from datetime import datetime

API_BASE = "http://localhost:5000/api/marl"

class PerformanceOptimizer:
    def __init__(self):
        self.results = []
    
    def create_test_session(self, config):
        """Create a test session with given configuration"""
        try:
            response = requests.post(f"{API_BASE}/session", json=config, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['session_id']
        except Exception as e:
            print(f"Failed to create session: {e}")
        return None
    
    def measure_simulation_throughput(self, session_id, duration=10):
        """Measure simulation steps per second"""
        try:
            # Get initial state
            response = requests.get(f"{API_BASE}/session/{session_id}/state", timeout=5)
            if response.status_code != 200:
                return None
            
            initial_step = response.json()['state']['step']
            
            # Start simulation
            requests.post(f"{API_BASE}/session/{session_id}/control", 
                         json={'action': 'start'}, timeout=5)
            
            start_time = time.time()
            time.sleep(duration)
            end_time = time.time()
            
            # Stop simulation
            requests.post(f"{API_BASE}/session/{session_id}/control", 
                         json={'action': 'stop'}, timeout=5)
            
            # Get final state
            response = requests.get(f"{API_BASE}/session/{session_id}/state", timeout=5)
            if response.status_code != 200:
                return None
            
            final_step = response.json()['state']['step']
            actual_duration = end_time - start_time
            
            steps_per_second = (final_step - initial_step) / actual_duration
            return {
                'steps_per_second': steps_per_second,
                'total_steps': final_step - initial_step,
                'duration': actual_duration
            }
            
        except Exception as e:
            print(f"Throughput measurement failed: {e}")
            return None
    
    def measure_api_latency(self, session_id, num_requests=50):
        """Measure API response latency"""
        latencies = []
        
        for _ in range(num_requests):
            start_time = time.time()
            try:
                response = requests.get(f"{API_BASE}/session/{session_id}/state", timeout=5)
                if response.status_code == 200:
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
            except:
                continue
        
        if latencies:
            return {
                'avg_latency_ms': statistics.mean(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'median_latency_ms': statistics.median(latencies),
                'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))],
                'successful_requests': len(latencies),
                'total_requests': num_requests
            }
        return None
    
    def test_configuration(self, config, test_name):
        """Test a specific configuration"""
        print(f"\nTesting: {test_name}")
        print(f"Config: {config}")
        
        session_id = self.create_test_session(config)
        if not session_id:
            print("❌ Failed to create session")
            return None
        
        # Measure throughput
        print("Measuring simulation throughput...")
        throughput = self.measure_simulation_throughput(session_id, duration=5)
        
        # Measure API latency
        print("Measuring API latency...")
        latency = self.measure_api_latency(session_id, num_requests=20)
        
        result = {
            'name': test_name,
            'config': config,
            'throughput': throughput,
            'latency': latency,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Print results
        if throughput:
            print(f"✓ Throughput: {throughput['steps_per_second']:.1f} steps/sec")
        if latency:
            print(f"✓ Avg Latency: {latency['avg_latency_ms']:.1f}ms")
            print(f"✓ P95 Latency: {latency['p95_latency_ms']:.1f}ms")
        
        return result
    
    def run_optimization_tests(self):
        """Run various configuration tests"""
        print("=" * 60)
        print("MARL SYSTEM PERFORMANCE OPTIMIZATION")
        print("=" * 60)
        
        # Test different environment sizes
        size_configs = [
            {'width': 8, 'height': 8, 'num_agents': 2, 'max_packages': 3, 'simulation_speed': 100},
            {'width': 12, 'height': 12, 'num_agents': 3, 'max_packages': 4, 'simulation_speed': 100},
            {'width': 16, 'height': 16, 'num_agents': 4, 'max_packages': 5, 'simulation_speed': 100},
        ]
        
        for i, config in enumerate(size_configs):
            self.test_configuration(config, f"Environment Size {config['width']}x{config['height']}")
        
        # Test different agent counts
        agent_configs = [
            {'width': 12, 'height': 12, 'num_agents': 2, 'max_packages': 4, 'simulation_speed': 100},
            {'width': 12, 'height': 12, 'num_agents': 3, 'max_packages': 4, 'simulation_speed': 100},
            {'width': 12, 'height': 12, 'num_agents': 4, 'max_packages': 4, 'simulation_speed': 100},
        ]
        
        for config in agent_configs:
            self.test_configuration(config, f"{config['num_agents']} Agents")
        
        # Test different simulation speeds
        speed_configs = [
            {'width': 12, 'height': 12, 'num_agents': 3, 'max_packages': 4, 'simulation_speed': 50},
            {'width': 12, 'height': 12, 'num_agents': 3, 'max_packages': 4, 'simulation_speed': 100},
            {'width': 12, 'height': 12, 'num_agents': 3, 'max_packages': 4, 'simulation_speed': 200},
        ]
        
        for config in speed_configs:
            self.test_configuration(config, f"Speed {config['simulation_speed']}ms")
    
    def analyze_results(self):
        """Analyze and report optimization results"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Find best configurations
        valid_results = [r for r in self.results if r['throughput'] and r['latency']]
        
        if not valid_results:
            print("No valid results for analysis")
            return
        
        # Best throughput
        best_throughput = max(valid_results, key=lambda x: x['throughput']['steps_per_second'])
        print(f"🏆 Best Throughput: {best_throughput['name']}")
        print(f"   {best_throughput['throughput']['steps_per_second']:.1f} steps/sec")
        print(f"   Config: {best_throughput['config']}")
        
        # Best latency
        best_latency = min(valid_results, key=lambda x: x['latency']['avg_latency_ms'])
        print(f"\n🏆 Best Latency: {best_latency['name']}")
        print(f"   {best_latency['latency']['avg_latency_ms']:.1f}ms average")
        print(f"   Config: {best_latency['config']}")
        
        # Performance summary
        throughputs = [r['throughput']['steps_per_second'] for r in valid_results]
        latencies = [r['latency']['avg_latency_ms'] for r in valid_results]
        
        print(f"\n📊 Performance Summary:")
        print(f"   Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} steps/sec")
        print(f"   Latency range: {min(latencies):.1f} - {max(latencies):.1f}ms")
        print(f"   Average throughput: {statistics.mean(throughputs):.1f} steps/sec")
        print(f"   Average latency: {statistics.mean(latencies):.1f}ms")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        # Analyze environment size impact
        size_results = [r for r in valid_results if 'Environment Size' in r['name']]
        if len(size_results) >= 2:
            size_throughputs = [(r['config']['width'] * r['config']['height'], 
                               r['throughput']['steps_per_second']) for r in size_results]
            size_throughputs.sort()
            
            if size_throughputs[-1][1] > size_throughputs[0][1] * 1.2:
                print(f"   - Larger environments may provide better throughput")
            elif size_throughputs[0][1] > size_throughputs[-1][1] * 1.2:
                print(f"   - Smaller environments provide better throughput")
            else:
                print(f"   - Environment size has minimal impact on throughput")
        
        # Analyze agent count impact
        agent_results = [r for r in valid_results if 'Agents' in r['name']]
        if len(agent_results) >= 2:
            agent_throughputs = [(r['config']['num_agents'], 
                                r['throughput']['steps_per_second']) for r in agent_results]
            agent_throughputs.sort()
            
            if len(agent_throughputs) >= 3:
                # Check if throughput decreases with more agents
                if agent_throughputs[0][1] > agent_throughputs[-1][1]:
                    print(f"   - Fewer agents provide better performance")
                elif agent_throughputs[-1][1] > agent_throughputs[0][1]:
                    print(f"   - More agents can be handled efficiently")
        
        print(f"\n✅ Optimization analysis complete!")
    
    def save_results(self, filename="performance_results.json"):
        """Save detailed performance results"""
        report = {
            'summary': {
                'total_tests': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'best_throughput': None,
                'best_latency': None
            },
            'results': self.results
        }
        
        # Add best results to summary
        valid_results = [r for r in self.results if r['throughput'] and r['latency']]
        if valid_results:
            best_throughput = max(valid_results, key=lambda x: x['throughput']['steps_per_second'])
            best_latency = min(valid_results, key=lambda x: x['latency']['avg_latency_ms'])
            
            report['summary']['best_throughput'] = {
                'name': best_throughput['name'],
                'steps_per_second': best_throughput['throughput']['steps_per_second'],
                'config': best_throughput['config']
            }
            
            report['summary']['best_latency'] = {
                'name': best_latency['name'],
                'avg_latency_ms': best_latency['latency']['avg_latency_ms'],
                'config': best_latency['config']
            }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed performance report saved to: {filename}")

def main():
    """Main optimization execution"""
    optimizer = PerformanceOptimizer()
    
    try:
        optimizer.run_optimization_tests()
        optimizer.analyze_results()
        optimizer.save_results("/home/ubuntu/marl_warehouse/performance_results.json")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

