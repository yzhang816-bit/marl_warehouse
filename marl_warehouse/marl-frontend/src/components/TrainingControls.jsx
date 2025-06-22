import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Slider } from '@/components/ui/slider.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Brain, Settings, Zap } from 'lucide-react'

const TrainingControls = ({ config, onConfigChange, isTraining, onTrainingToggle }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="h-5 w-5" />
          <span>Learning Parameters</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Learning Rate</Label>
          <Slider
            min={0.0001}
            max={0.01}
            step={0.0001}
            value={[config.learningRate]}
            onValueChange={([value]) => onConfigChange({ ...config, learningRate: value })}
          />
          <div className="text-xs text-slate-600 font-mono">
            {config.learningRate.toFixed(4)}
          </div>
        </div>
        
        <div className="space-y-2">
          <Label>Exploration Rate</Label>
          <Slider
            min={0.01}
            max={0.5}
            step={0.01}
            value={[config.explorationRate]}
            onValueChange={([value]) => onConfigChange({ ...config, explorationRate: value })}
          />
          <div className="text-xs text-slate-600 font-mono">
            {config.explorationRate.toFixed(2)}
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <Label>Reward Shaping</Label>
          <Switch
            checked={config.rewardShaping}
            onCheckedChange={(checked) => onConfigChange({ ...config, rewardShaping: checked })}
          />
        </div>
        
        <div className="flex items-center justify-between">
          <Label>Show Communication</Label>
          <Switch
            checked={config.showCommunication}
            onCheckedChange={(checked) => onConfigChange({ ...config, showCommunication: checked })}
          />
        </div>
        
        <div className="pt-4 border-t">
          <Button 
            onClick={onTrainingToggle}
            variant={isTraining ? "destructive" : "default"}
            className="w-full"
          >
            <Zap className="h-4 w-4 mr-2" />
            {isTraining ? 'Stop Training' : 'Start Training'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export default TrainingControls

