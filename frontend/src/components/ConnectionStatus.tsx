import React, { useState, useEffect } from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { websocketService } from '../services/websocket';

const ConnectionStatus: React.FC = () => {
  const [status, setStatus] = useState({ trades: false, predictions: false });

  useEffect(() => {
    const checkStatus = () => {
      setStatus(websocketService.getConnectionStatus());
    };

    checkStatus();
    const interval = setInterval(checkStatus, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Typography variant="body2" color="textSecondary">
        Connections:
      </Typography>
      <Chip
        label="Trades"
        color={status.trades ? 'success' : 'error'}
        size="small"
      />
      <Chip
        label="Predictions"
        color={status.predictions ? 'success' : 'error'}
        size="small"
      />
    </Box>
  );
};

export default ConnectionStatus;