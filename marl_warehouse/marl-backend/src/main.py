import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
import threading
import time
from marl_simulation import SimpleWarehouseEnvironment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'marl-secret-key-2025'

# Configure CORS for all origins
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Configure SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global storage for simulation sessions
sessions = {}
simulation_threads = {}

class MARLSession:
    def __init__(self, config):
        self.env = SimpleWarehouseEnvironment(
            width=config.get('width', 15),
            height=config.get('height', 15),
            num_agents=config.get('num_agents', 3),
            max_packages=config.get('max_packages', 5)
        )
        self.is_running_flag = False
        self.is_training_flag = False
        self.simulation_speed = config.get('simulation_speed', 500)
    
    def get_state(self):
        return self.env.get_state()
    
    def step(self):
        return self.env.step()
    
    def reset(self):
        return self.env.reset()
    
    def set_running(self, running):
        self.is_running_flag = running
    
    def is_running(self):
        return self.is_running_flag
    
    def set_training(self, training):
        self.is_training_flag = training
    
    def is_training(self):
        return self.is_training_flag
    
    def get_speed(self):
        return self.simulation_speed
    
    def update_config(self, config):
        if 'simulation_speed' in config:
            self.simulation_speed = config['simulation_speed']

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "service": "MARL Backend",
        "status": "healthy"
    })

@app.route('/api/marl/session', methods=['POST'])
def create_session():
    try:
        data = request.get_json() or {}
        
        session_id = str(uuid.uuid4())
        
        # Create new simulation with configuration
        config = {
            'width': data.get('width', 15),
            'height': data.get('height', 15),
            'num_agents': data.get('num_agents', 3),
            'max_packages': data.get('max_packages', 5),
            'simulation_speed': data.get('simulation_speed', 500)
        }
        
        simulation = MARLSession(config)
        sessions[session_id] = simulation
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "initial_state": simulation.get_state()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/marl/session/<session_id>/state', methods=['GET'])
def get_session_state(session_id):
    try:
        if session_id not in sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
        simulation = sessions[session_id]
        return jsonify({
            "success": True,
            "state": simulation.get_state()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/marl/session/<session_id>/control', methods=['POST'])
def control_simulation(session_id):
    try:
        if session_id not in sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
        data = request.get_json()
        action = data.get('action')
        
        simulation = sessions[session_id]
        
        if action == 'start':
            start_simulation_thread(session_id)
        elif action == 'stop':
            stop_simulation_thread(session_id)
        elif action == 'start_training':
            simulation.set_training(True)
            start_simulation_thread(session_id)
        elif action == 'stop_training':
            simulation.set_training(False)
        
        return jsonify({
            "success": True,
            "state": simulation.get_state()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/marl/session/<session_id>/reset', methods=['POST'])
def reset_simulation(session_id):
    try:
        if session_id not in sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
        # Stop any running simulation
        stop_simulation_thread(session_id)
        
        # Reset the simulation
        simulation = sessions[session_id]
        simulation.reset()
        
        return jsonify({
            "success": True,
            "state": simulation.get_state()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/marl/session/<session_id>/config', methods=['PUT'])
def update_config(session_id):
    try:
        if session_id not in sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
        data = request.get_json()
        simulation = sessions[session_id]
        simulation.update_config(data)
        
        return jsonify({
            "success": True,
            "state": simulation.get_state()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def start_simulation_thread(session_id):
    if session_id in simulation_threads and simulation_threads[session_id].is_alive():
        return  # Already running
    
    def run_simulation():
        simulation = sessions[session_id]
        simulation.set_running(True)
        
        while simulation.is_running():
            try:
                # Step the simulation
                simulation.step()
                
                # Emit update to connected clients
                socketio.emit('simulation_update', {
                    'session_id': session_id,
                    'state': simulation.get_state()
                }, room=f'session_{session_id}')
                
                # Wait based on simulation speed
                time.sleep(simulation.get_speed() / 1000.0)
                
            except Exception as e:
                print(f"Simulation error: {e}")
                break
    
    thread = threading.Thread(target=run_simulation)
    thread.daemon = True
    thread.start()
    simulation_threads[session_id] = thread

def stop_simulation_thread(session_id):
    if session_id in sessions:
        sessions[session_id].set_running(False)
    
    if session_id in simulation_threads:
        thread = simulation_threads[session_id]
        if thread.is_alive():
            thread.join(timeout=1.0)
        del simulation_threads[session_id]

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to MARL backend'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    session_id = data.get('session_id')
    if session_id and session_id in sessions:
        join_room(f'session_{session_id}')
        emit('joined_session', {'session_id': session_id})
        print(f"Client {request.sid} joined session {session_id}")

@socketio.on('leave_session')
def handle_leave_session(data):
    session_id = data.get('session_id')
    if session_id:
        leave_room(f'session_{session_id}')
        emit('left_session', {'session_id': session_id})
        print(f"Client {request.sid} left session {session_id}")

if __name__ == '__main__':
    print("Starting MARL Backend Server...")
    print("Backend will be available at:")
    print("- Local: http://127.0.0.1:5000")
    print("- Network: http://0.0.0.0:5000")
    print("- Health check: http://127.0.0.1:5000/health")
    
    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

