import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Button,
  Slider,
  Grid,
  Switch,
  FormControlLabel,
  Chip,
} from '@mui/material';
import { PlayArrow, Stop, GetApp } from '@mui/icons-material';
import { api } from '../services/api';

interface ControlPanelProps {
  isStreaming: boolean;
  setIsStreaming: (streaming: boolean) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (threshold: number) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  isStreaming,
  setIsStreaming,
  confidenceThreshold,
  setConfidenceThreshold,
}) => {
  const [loading, setLoading] = useState(false);

  const handleStartStop = async () => {
    setLoading(true);
    try {
      if (isStreaming) {
        console.log('🛑 Stopping services...');
        await api.stopStreaming();
        await api.stopProcessing();
        setIsStreaming(false);
        console.log('✅ Services stopped');
      } else {
        console.log('▶️ Starting services...');
        await api.startStreaming();
        await api.startProcessing();
        setIsStreaming(true);
        console.log('✅ Services started');
      }
    } catch (error) {
      console.error('❌ Error toggling streaming:', error);
      // Reset state on error
      setIsStreaming(false);
    } finally {
      setLoading(false);
    }
  };

  const handleThresholdChange = async (event: Event, newValue: number | number[]) => {
    const threshold = Array.isArray(newValue) ? newValue[0] : newValue;
    setConfidenceThreshold(threshold);
    try {
      await api.setThreshold(threshold);
      console.log(`🎯 Threshold set to ${(threshold * 100).toFixed(0)}%`);
    } catch (error) {
      console.error('Error setting threshold:', error);
    }
  };

  const handleExport = () => {
    // In a real implementation, this would trigger a CSV download
    console.log('📊 Exporting results...');
    alert('Export functionality would be implemented here');
  };

  return (
    <Paper sx={{ p: 2, height: '200px' }}>
      <Typography variant="h6" gutterBottom>
        Control Panel
      </Typography>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} sm={6}>
          <Box>
            <Button
              variant="contained"
              color={isStreaming ? 'error' : 'success'}
              startIcon={isStreaming ? <Stop /> : <PlayArrow />}
              onClick={handleStartStop}
              disabled={loading}
              fullWidth
              sx={{ mb: 2 }}
            >
              {loading 
                ? (isStreaming ? 'Stopping...' : 'Starting...') 
                : (isStreaming ? 'Stop Streaming' : 'Start Streaming')
              }
            </Button>
            <Button
              variant="outlined"
              startIcon={<GetApp />}
              onClick={handleExport}
              fullWidth
              disabled={loading}
            >
              Export Results
            </Button>
          </Box>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Box>
            <Typography variant="body2" gutterBottom>
              Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
            </Typography>
            <Slider
              value={confidenceThreshold}
              onChange={handleThresholdChange}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
              sx={{ mb: 2 }}
              disabled={loading}
            />
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="textSecondary">
                Status:
              </Typography>
              <Chip
                label={isStreaming ? 'ACTIVE' : 'STOPPED'}
                color={isStreaming ? 'success' : 'default'}
                size="small"
              />
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ControlPanel;