import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Users, Battery, MapPin, Package } from 'lucide-react'

const AgentPanel = ({ agents, packages }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Users className="h-5 w-5" />
          <span>Agent Status</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {agents.map(agent => {
          const carriedPackage = agent.carrying !== null ? packages.find(p => p.id === agent.carrying) : null
          
          return (
            <div key={agent.id} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Agent {agent.id}</span>
                <Badge variant={agent.carrying !== null ? "default" : "secondary"}>
                  {agent.carrying !== null ? 'Carrying' : 'Available'}
                </Badge>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    <MapPin className="h-3 w-3" />
                    <span>Position:</span>
                  </div>
                  <span className="font-mono">({agent.position[0]}, {agent.position[1]})</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    <Battery className="h-3 w-3" />
                    <span>Battery:</span>
                  </div>
                  <span className="font-mono">{agent.battery.toFixed(1)}%</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Last Action:</span>
                  <span className="font-mono text-xs">{agent.lastAction || 'WAIT'}</span>
                </div>
                
                {carriedPackage && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-1">
                      <Package className="h-3 w-3" />
                      <span>Carrying:</span>
                    </div>
                    <span className="font-mono text-xs">Package {carriedPackage.id}</span>
                  </div>
                )}
              </div>
              
              <div className="mt-2">
                <Progress value={agent.battery} className="h-1" />
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

export default AgentPanel

