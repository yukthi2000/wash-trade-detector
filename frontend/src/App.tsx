import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Typography,
  Box,
  AppBar,
  Toolbar,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import { Security } from '@mui/icons-material';
import LiveTradePanel from './components/LiveTradePanel';
import PredictionPanel from './components/PredictionPanel';
import PerformancePanel from './components/PerformancePanel';
import ComparisonPanel from './components/ComparisonPanel';
import ControlPanel from './components/ControlPanel';
import { websocketService, type Trade, type PredictionResult } from './services/websocket';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [predictions, setPredictions] = useState<{ [key: string]: PredictionResult }>({});
  const [latestPrediction, setLatestPrediction] = useState<PredictionResult | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  useEffect(() => {
    // Connect to WebSocket services
    websocketService.connectToTrades((trade: Trade) => {
      setTrades(prev => [...prev.slice(-100), trade]); // Keep last 100 trades
    });

    websocketService.connectToPredictions((prediction: PredictionResult) => {
      setPredictions(prev => ({
        ...prev,
        [prediction.transaction_hash]: prediction
      }));
      setLatestPrediction(prediction);
    });

    return () => {
      websocketService.disconnect();
    };
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Security sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Wash Trade Detection System - Real-time Demo
          </Typography>
          <Typography variant="body2">
            ML Model: Random Forest | Status: {isStreaming ? 'Active' : 'Stopped'}
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
        <Grid container spacing={2}>
          {/* Control Panel */}
          <Grid item xs={12} md={6}>
            <ControlPanel
              isStreaming={isStreaming}
              setIsStreaming={setIsStreaming}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
            />
          </Grid>
          
          {/* Latest Prediction Panel */}
          <Grid item xs={12} md={6}>
            <PredictionPanel latestPrediction={latestPrediction} />
          </Grid>
          
          {/* Live Trade Stream */}
          <Grid item xs={12}>
            <LiveTradePanel trades={trades} predictions={predictions} />
          </Grid>
          
          {/* Performance Metrics */}
          <Grid item xs={12} md={6}>
            <PerformancePanel />
          </Grid>
          
          {/* Comparison Analysis */}
          <Grid item xs={12} md={6}>
            <ComparisonPanel />
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;