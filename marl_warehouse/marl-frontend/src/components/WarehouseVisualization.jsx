import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const WarehouseVisualization = ({ warehouseState, config, isRunning }) => {
  const canvasRef = useRef(null)
  const [cellSize, setCellSize] = useState(30)
  const [agentTrails, setAgentTrails] = useState({})
  
  const { width, height, agents, packages, deliveryZones, obstacles } = warehouseState
  
  // Update agent trails for trajectory visualization
  useEffect(() => {
    if (config.showTrajectories) {
      setAgentTrails(prev => {
        const newTrails = { ...prev }
        agents.forEach(agent => {
          if (!newTrails[agent.id]) {
            newTrails[agent.id] = []
          }
          newTrails[agent.id].push([...agent.position])
          // Keep only last 20 positions
          if (newTrails[agent.id].length > 20) {
            newTrails[agent.id].shift()
          }
        })
        return newTrails
      })
    } else {
      setAgentTrails({})
    }
  }, [agents, config.showTrajectories])
  
  // Calculate canvas dimensions
  const canvasWidth = Math.min(800, width * cellSize)
  const canvasHeight = Math.min(600, height * cellSize)
  const actualCellSize = Math.min(canvasWidth / width, canvasHeight / height)
  
  // Draw warehouse on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw grid
    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1
    for (let x = 0; x <= width; x++) {
      ctx.beginPath()
      ctx.moveTo(x * actualCellSize, 0)
      ctx.lineTo(x * actualCellSize, height * actualCellSize)
      ctx.stroke()
    }
    for (let y = 0; y <= height; y++) {
      ctx.beginPath()
      ctx.moveTo(0, y * actualCellSize)
      ctx.lineTo(width * actualCellSize, y * actualCellSize)
      ctx.stroke()
    }
    
    // Draw delivery zones
    ctx.fillStyle = '#dbeafe'
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    deliveryZones.forEach(([x, y]) => {
      ctx.fillRect(x * actualCellSize + 2, y * actualCellSize + 2, actualCellSize - 4, actualCellSize - 4)
      ctx.strokeRect(x * actualCellSize + 2, y * actualCellSize + 2, actualCellSize - 4, actualCellSize - 4)
    })
    
    // Draw obstacles
    ctx.fillStyle = '#374151'
    obstacles.forEach(([x, y]) => {
      ctx.fillRect(x * actualCellSize + 1, y * actualCellSize + 1, actualCellSize - 2, actualCellSize - 2)
    })
    
    // Draw agent trails
    if (config.showTrajectories) {
      Object.entries(agentTrails).forEach(([agentId, trail]) => {
        if (trail.length > 1) {
          const colors = ['#ef4444', '#10b981', '#8b5cf6']
          ctx.strokeStyle = colors[parseInt(agentId)] || '#6b7280'
          ctx.lineWidth = 2
          ctx.globalAlpha = 0.6
          
          ctx.beginPath()
          trail.forEach(([x, y], index) => {
            const centerX = x * actualCellSize + actualCellSize / 2
            const centerY = y * actualCellSize + actualCellSize / 2
            if (index === 0) {
              ctx.moveTo(centerX, centerY)
            } else {
              ctx.lineTo(centerX, centerY)
            }
          })
          ctx.stroke()
          ctx.globalAlpha = 1
        }
      })
    }
    
  }, [warehouseState, actualCellSize, agentTrails, config.showTrajectories])
  
  // Agent colors
  const agentColors = [
    { bg: '#ef4444', border: '#dc2626', name: 'Red' },
    { bg: '#10b981', border: '#059669', name: 'Green' },
    { bg: '#8b5cf6', border: '#7c3aed', name: 'Purple' }
  ]
  
  // Package colors by priority
  const getPackageColor = (priority) => {
    switch (priority) {
      case 1: return { bg: '#fbbf24', border: '#f59e0b' }
      case 2: return { bg: '#fb923c', border: '#ea580c' }
      case 3: return { bg: '#f87171', border: '#ef4444' }
      default: return { bg: '#94a3b8', border: '#64748b' }
    }
  }
  
  return (
    <div className="relative bg-white dark:bg-slate-900 rounded-lg border overflow-hidden">
      {/* Canvas for static elements */}
      <canvas
        ref={canvasRef}
        width={width * actualCellSize}
        height={height * actualCellSize}
        className="absolute inset-0"
      />
      
      {/* Animated elements overlay */}
      <div 
        className="relative"
        style={{ 
          width: width * actualCellSize, 
          height: height * actualCellSize 
        }}
      >
        {/* Packages */}
        <AnimatePresence>
          {packages.filter(pkg => !pkg.delivered).map(pkg => {
            const isCarried = agents.some(agent => agent.carrying === pkg.id)
            if (isCarried) return null
            
            const colors = getPackageColor(pkg.priority)
            return (
              <motion.div
                key={`package-${pkg.id}`}
                className="absolute flex items-center justify-center rounded-lg shadow-lg"
                style={{
                  left: pkg.position[0] * actualCellSize + 4,
                  top: pkg.position[1] * actualCellSize + 4,
                  width: actualCellSize - 8,
                  height: actualCellSize - 8,
                  backgroundColor: colors.bg,
                  borderColor: colors.border,
                  borderWidth: 2
                }}
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                exit={{ scale: 0, rotate: 180 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                whileHover={{ scale: 1.1 }}
              >
                <div className="text-white font-bold text-xs">
                  P{pkg.id}
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-white rounded-full flex items-center justify-center">
                  <span className="text-xs font-bold text-slate-700">{pkg.priority}</span>
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
        
        {/* Agents */}
        <AnimatePresence>
          {agents.map((agent, index) => {
            const colors = agentColors[index] || agentColors[0]
            const carriedPackage = agent.carrying !== null ? packages.find(p => p.id === agent.carrying) : null
            
            return (
              <motion.div
                key={`agent-${agent.id}`}
                className="absolute flex items-center justify-center rounded-full shadow-lg cursor-pointer"
                style={{
                  left: agent.position[0] * actualCellSize + 2,
                  top: agent.position[1] * actualCellSize + 2,
                  width: actualCellSize - 4,
                  height: actualCellSize - 4,
                  backgroundColor: colors.bg,
                  borderColor: colors.border,
                  borderWidth: 3,
                  zIndex: 10
                }}
                animate={{
                  left: agent.position[0] * actualCellSize + 2,
                  top: agent.position[1] * actualCellSize + 2,
                  scale: isRunning ? [1, 1.05, 1] : 1
                }}
                transition={{ 
                  type: "spring", 
                  stiffness: 300, 
                  damping: 25,
                  scale: { duration: 0.5, repeat: isRunning ? Infinity : 0 }
                }}
                whileHover={{ scale: 1.1 }}
              >
                <div className="text-white font-bold text-xs">
                  A{agent.id}
                </div>
                
                {/* Battery indicator */}
                <div 
                  className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-6 h-1 bg-gray-300 rounded-full overflow-hidden"
                >
                  <div 
                    className={`h-full transition-all duration-300 ${
                      agent.battery > 50 ? 'bg-green-500' : 
                      agent.battery > 20 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${agent.battery}%` }}
                  />
                </div>
                
                {/* Carried package indicator */}
                {carriedPackage && (
                  <motion.div
                    className="absolute -top-2 -right-2 w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold text-white"
                    style={{
                      backgroundColor: getPackageColor(carriedPackage.priority).bg
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    P
                  </motion.div>
                )}
                
                {/* Action indicator */}
                {agent.lastAction && agent.lastAction !== 'WAIT' && (
                  <motion.div
                    className="absolute -top-6 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-black bg-opacity-75 text-white text-xs rounded"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    {agent.lastAction}
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </AnimatePresence>
        
        {/* Communication lines (if enabled) */}
        {config.showCommunication && (
          <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: 5 }}>
            {agents.map((agent1, i) => 
              agents.slice(i + 1).map((agent2, j) => {
                const distance = Math.sqrt(
                  Math.pow(agent1.position[0] - agent2.position[0], 2) + 
                  Math.pow(agent1.position[1] - agent2.position[1], 2)
                )
                
                // Only show communication lines for nearby agents
                if (distance > 5) return null
                
                return (
                  <motion.line
                    key={`comm-${agent1.id}-${agent2.id}`}
                    x1={agent1.position[0] * actualCellSize + actualCellSize / 2}
                    y1={agent1.position[1] * actualCellSize + actualCellSize / 2}
                    x2={agent2.position[0] * actualCellSize + actualCellSize / 2}
                    y2={agent2.position[1] * actualCellSize + actualCellSize / 2}
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                    opacity="0.6"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.5 }}
                  />
                )
              })
            )}
          </svg>
        )}
      </div>
      
      {/* Legend */}
      <div className="absolute top-4 right-4 bg-white dark:bg-slate-800 rounded-lg shadow-lg p-3 space-y-2 text-xs">
        <div className="font-semibold mb-2">Legend</div>
        
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-100 border-2 border-blue-500 rounded"></div>
          <span>Delivery Zone</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-gray-600 rounded"></div>
          <span>Obstacle</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-yellow-400 border-2 border-yellow-500 rounded-lg"></div>
          <span>Package</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 border-2 border-red-600 rounded-full"></div>
          <span>Agent</span>
        </div>
        
        {config.showTrajectories && (
          <div className="flex items-center space-x-2">
            <div className="w-4 h-1 bg-red-500 opacity-60"></div>
            <span>Trail</span>
          </div>
        )}
      </div>
      
      {/* Status overlay */}
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white rounded-lg px-3 py-2 text-sm">
        <div>Delivered: {packages.filter(p => p.delivered).length}/{packages.length}</div>
        <div>Active: {agents.filter(a => a.battery > 0).length}/{agents.length} agents</div>
      </div>
    </div>
  )
}

export default WarehouseVisualization

