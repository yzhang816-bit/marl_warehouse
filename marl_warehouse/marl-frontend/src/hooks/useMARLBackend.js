import { useState, useEffect, useCallback, useRef } from 'react'
import { io } from 'socket.io-client'

const API_BASE_URL = 'https://5000-inme1n88910c64cm9fasp-fa0a3a6a.manusvm.computer/api/marl'
const SOCKET_URL = 'https://5000-inme1n88910c64cm9fasp-fa0a3a6a.manusvm.computer'

export const useMARLBackend = () => {
  const [sessionId, setSessionId] = useState(null)
  const [warehouseState, setWarehouseState] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [currentMetrics, setCurrentMetrics] = useState({
    totalReward: 0,
    deliveredPackages: 0,
    coordinationScore: 0.5,
    efficiency: 0.0,
    averageDistance: 0.3
  })
  
  const socketRef = useRef(null)
  
  // Initialize WebSocket connection
  useEffect(() => {
    socketRef.current = io(SOCKET_URL, {
      transports: ['websocket', 'polling']
    })
    
    socketRef.current.on('connect', () => {
      setIsConnected(true)
      console.log('Connected to MARL backend')
    })
    
    socketRef.current.on('disconnect', () => {
      setIsConnected(false)
      console.log('Disconnected from MARL backend')
    })
    
    socketRef.current.on('simulation_update', (data) => {
      if (data.session_id === sessionId) {
        updateStateFromBackend(data.state)
      }
    })
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
    }
  }, [sessionId])
  
  const updateStateFromBackend = useCallback((backendState) => {
    // Convert backend state to frontend format
    const frontendState = {
      width: backendState.warehouse.width,
      height: backendState.warehouse.height,
      agents: backendState.warehouse.agents.map(agent => ({
        id: agent.id,
        position: agent.position,
        carrying: agent.carrying,
        battery: agent.battery,
        lastAction: agent.lastAction
      })),
      packages: backendState.warehouse.packages.map(pkg => ({
        id: pkg.id,
        position: pkg.position,
        destination: pkg.destination,
        priority: pkg.priority,
        delivered: pkg.delivered
      })),
      deliveryZones: backendState.warehouse.deliveryZones,
      obstacles: backendState.warehouse.obstacles
    }
    
    setWarehouseState(frontendState)
    setCurrentMetrics(backendState.metrics)
  }, [])
  
  const createSession = useCallback(async (config = {}) => {
    try {
      const response = await fetch(`${API_BASE_URL}/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          width: config.width || 15,
          height: config.height || 15,
          num_agents: config.num_agents || 3,
          max_packages: config.max_packages || 5,
          simulation_speed: config.simulation_speed || 500
        })
      })
      
      const data = await response.json()
      
      if (data.success) {
        setSessionId(data.session_id)
        updateStateFromBackend(data.initial_state)
        
        // Join WebSocket room
        if (socketRef.current) {
          socketRef.current.emit('join_session', { session_id: data.session_id })
        }
        
        return data.session_id
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      console.error('Failed to create session:', error)
      throw error
    }
  }, [updateStateFromBackend])
  
  const controlSimulation = useCallback(async (action) => {
    if (!sessionId) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action })
      })
      
      const data = await response.json()
      
      if (data.success) {
        switch (action) {
          case 'start':
            setIsRunning(true)
            break
          case 'stop':
            setIsRunning(false)
            break
          case 'start_training':
            setIsTraining(true)
            setIsRunning(true)
            break
          case 'stop_training':
            setIsTraining(false)
            break
        }
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      console.error('Failed to control simulation:', error)
      throw error
    }
  }, [sessionId])
  
  const resetSimulation = useCallback(async () => {
    if (!sessionId) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/reset`, {
        method: 'POST'
      })
      
      const data = await response.json()
      
      if (data.success) {
        updateStateFromBackend(data.state)
        setIsRunning(false)
        setIsTraining(false)
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      console.error('Failed to reset simulation:', error)
      throw error
    }
  }, [sessionId, updateStateFromBackend])
  
  const updateConfig = useCallback(async (config) => {
    if (!sessionId) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      })
      
      const data = await response.json()
      
      if (!data.success) {
        throw new Error(data.error)
      }
    } catch (error) {
      console.error('Failed to update config:', error)
      throw error
    }
  }, [sessionId])
  
  const startSimulation = useCallback(() => controlSimulation('start'), [controlSimulation])
  const stopSimulation = useCallback(() => controlSimulation('stop'), [controlSimulation])
  const startTraining = useCallback(() => controlSimulation('start_training'), [controlSimulation])
  const stopTraining = useCallback(() => controlSimulation('stop_training'), [controlSimulation])
  
  return {
    // State
    sessionId,
    warehouseState,
    currentMetrics,
    isConnected,
    isRunning,
    isTraining,
    
    // Actions
    createSession,
    startSimulation,
    stopSimulation,
    startTraining,
    stopTraining,
    resetSimulation,
    updateConfig
  }
}

