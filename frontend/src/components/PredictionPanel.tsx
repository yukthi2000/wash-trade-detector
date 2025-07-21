import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {Grid} from '@mui/material';
import type { PredictionResult } from '../services/websocket';

interface PredictionPanelProps {
  latestPrediction: PredictionResult | null;
}

const PredictionPanel: React.FC<PredictionPanelProps> = ({ latestPrediction }) => {
  if (!latestPrediction) {
    return (
      <Paper sx={{ p: 2, height: '200px' }}>
        <Typography variant="h6" gutterBottom>
          Latest Prediction
        </Typography>
        <Typography color="textSecondary">
          Waiting for predictions...
        </Typography>
      </Paper>
    );
  }

  const confidencePercentage = latestPrediction.prediction_probability * 100;

  return (
    <Paper sx={{ p: 2, height: '200px' }}>
      <Typography variant="h6" gutterBottom>
        Latest Prediction
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="textSecondary">
              Prediction
            </Typography>
            <Chip
              label={latestPrediction.predicted_label === 1 ? 'WASH TRADE' : 'NORMAL TRADE'}
              color={latestPrediction.predicted_label === 1 ? 'error' : 'success'}
              sx={{ mt: 1 }}
            />
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="textSecondary">
              True Label
            </Typography>
            <Chip
              label={latestPrediction.true_label === 1 ? 'WASH TRADE' : 'NORMAL TRADE'}
              color={latestPrediction.true_label === 1 ? 'error' : 'success'}
              sx={{ mt: 1 }}
            />
          </Box>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="body2" color="textSecondary">
            Confidence: {confidencePercentage.toFixed(1)}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={confidencePercentage}
            sx={{ mt: 1, height: 8, borderRadius: 4 }}
            color={latestPrediction.agreement ? 'success' : 'error'}
          />
        </Grid>
        <Grid item xs={6}>
          <Typography variant="body2" color="textSecondary">
            Processing Time: {latestPrediction.processing_time_ms.toFixed(2)}ms
          </Typography>
        </Grid>
        <Grid item xs={6}>
          <Chip
            label={latestPrediction.agreement ? 'AGREEMENT' : 'DISAGREEMENT'}
            color={latestPrediction.agreement ? 'success' : 'error'}
            size="small"
          />
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PredictionPanel;