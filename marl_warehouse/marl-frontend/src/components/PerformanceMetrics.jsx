import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Target, TrendingUp, Package, Users } from 'lucide-react'

const PerformanceMetrics = ({ currentMetrics, warehouseState }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Target className="h-5 w-5" />
          <span>Current Performance</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Total Reward</span>
            <span className="font-mono">{currentMetrics.totalReward.toFixed(1)}</span>
          </div>
          <Progress value={Math.max(0, Math.min(100, (currentMetrics.totalReward + 50) * 2))} />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Coordination Score</span>
            <span className="font-mono">{(currentMetrics.coordinationScore * 100).toFixed(1)}%</span>
          </div>
          <Progress value={currentMetrics.coordinationScore * 100} />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Efficiency</span>
            <span className="font-mono">{(currentMetrics.efficiency * 100).toFixed(1)}%</span>
          </div>
          <Progress value={currentMetrics.efficiency * 100} />
        </div>
        
        <div className="grid grid-cols-2 gap-4 pt-2">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {currentMetrics.deliveredPackages}
            </div>
            <div className="text-xs text-slate-600">Delivered</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {warehouseState.agents.length}
            </div>
            <div className="text-xs text-slate-600">Active Agents</div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-lg font-bold text-orange-600">
              {warehouseState.packages.filter(p => !p.delivered).length}
            </div>
            <div className="text-xs text-slate-600">Pending</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-purple-600">
              {warehouseState.agents.filter(a => a.carrying !== null).length}
            </div>
            <div className="text-xs text-slate-600">In Transit</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default PerformanceMetrics

