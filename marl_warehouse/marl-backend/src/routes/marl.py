"""
MARL API routes for simulation control
"""

from flask import Blueprint, request, jsonify
from flask_socketio import emit, join_room, leave_room
from src.marl_simulation import simulation_manager
import uuid
import threading
import time

marl_bp = Blueprint('marl', __name__)

# Store active simulation threads
simulation_threads = {}

@marl_bp.route('/session', methods=['POST'])
def create_session():
    """Create new MARL simulation session"""
    try:
        data = request.get_json()
        session_id = str(uuid.uuid4())
        
        config = {
            'width': data.get('width', 15),
            'height': data.get('height', 15),
            'num_agents': data.get('num_agents', 3),
            'max_packages': data.get('max_packages', 5),
            'simulation_speed': data.get('simulation_speed', 500)
        }
        
        initial_state = simulation_manager.create_session(session_id, config)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'initial_state': initial_state
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@marl_bp.route('/session/<session_id>/state', methods=['GET'])
def get_session_state(session_id):
    """Get current state of simulation session"""
    try:
        state = simulation_manager.get_state(session_id)
        return jsonify({
            'success': True,
            'state': state
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@marl_bp.route('/session/<session_id>/step', methods=['POST'])
def step_simulation(session_id):
    """Step simulation forward"""
    try:
        state = simulation_manager.step_simulation(session_id)
        return jsonify({
            'success': True,
            'state': state
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@marl_bp.route('/session/<session_id>/reset', methods=['POST'])
def reset_simulation(session_id):
    """Reset simulation"""
    try:
        state = simulation_manager.reset_simulation(session_id)
        return jsonify({
            'success': True,
            'state': state
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@marl_bp.route('/session/<session_id>/config', methods=['PUT'])
def update_config(session_id):
    """Update simulation configuration"""
    try:
        data = request.get_json()
        simulation_manager.update_config(session_id, data)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@marl_bp.route('/session/<session_id>/control', methods=['POST'])
def control_simulation(session_id):
    """Control simulation (start/stop/training)"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action == 'start':
            simulation_manager.set_running(session_id, True)
            start_simulation_thread(session_id)
        elif action == 'stop':
            simulation_manager.set_running(session_id, False)
            stop_simulation_thread(session_id)
        elif action == 'start_training':
            simulation_manager.set_training(session_id, True)
            simulation_manager.set_running(session_id, True)
            start_simulation_thread(session_id)
        elif action == 'stop_training':
            simulation_manager.set_training(session_id, False)
        
        return jsonify({
            'success': True,
            'message': f'Action {action} executed'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

def start_simulation_thread(session_id):
    """Start simulation thread for real-time updates"""
    if session_id in simulation_threads:
        return  # Already running
    
    def simulation_loop():
        from src.main import socketio
        
        while session_id in simulation_threads:
            try:
                # Check if simulation should continue
                if session_id not in simulation_manager.training_sessions:
                    break
                
                session_state = simulation_manager.training_sessions[session_id]
                if not session_state['is_running']:
                    break
                
                # Step simulation
                state = simulation_manager.step_simulation(session_id)
                
                # Emit state update via WebSocket
                socketio.emit('simulation_update', {
                    'session_id': session_id,
                    'state': state
                }, room=f'session_{session_id}')
                
                # Sleep based on simulation speed
                speed = session_state['config'].get('simulation_speed', 500)
                time.sleep(speed / 1000.0)
                
            except Exception as e:
                print(f"Simulation error for session {session_id}: {e}")
                break
        
        # Clean up
        if session_id in simulation_threads:
            del simulation_threads[session_id]
    
    thread = threading.Thread(target=simulation_loop, daemon=True)
    simulation_threads[session_id] = thread
    thread.start()

def stop_simulation_thread(session_id):
    """Stop simulation thread"""
    if session_id in simulation_threads:
        del simulation_threads[session_id]

# WebSocket events
def register_socketio_events(socketio):
    """Register WebSocket events"""
    
    @socketio.on('join_session')
    def on_join_session(data):
        session_id = data.get('session_id')
        if session_id:
            join_room(f'session_{session_id}')
            emit('joined_session', {'session_id': session_id})
    
    @socketio.on('leave_session')
    def on_leave_session(data):
        session_id = data.get('session_id')
        if session_id:
            leave_room(f'session_{session_id}')
            emit('left_session', {'session_id': session_id})
    
    @socketio.on('disconnect')
    def on_disconnect():
        # Clean up any sessions if needed
        pass

