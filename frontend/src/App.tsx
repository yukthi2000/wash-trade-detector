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
  
  // Add a Set to track seen transaction hashes
  const [seenTradeHashes, setSeenTradeHashes] = useState<Set<string>>(new Set());

  // SINGLE useEffect for WebSocket connections
  useEffect(() => {
    console.log('🔌 Initializing WebSocket connections...');
    
    websocketService.connectToTrades((trade) => {
      console.log('📦 Trade received:', {
        hash: trade.transactionHash,
        seller: trade.eth_seller.slice(0, 10) + '...',
        buyer: trade.eth_buyer.slice(0, 10) + '...',
        amount: trade.trade_amount_eth,
        wash_label: trade.wash_label
      });
      
      // Check if we've already seen this trade
      setSeenTradeHashes(prevSeen => {
        if (prevSeen.has(trade.transactionHash)) {
          console.log('⚠️ Duplicate trade detected, skipping:', trade.transactionHash);
          return prevSeen;
        }
        
        // Add new trade
        const newSeen = new Set(prevSeen);
        newSeen.add(trade.transactionHash);
        
        setTrades(prev => {
          // Double-check in trades array too
          const exists = prev.some(t => t.transactionHash === trade.transactionHash);
          if (exists) {
            console.log('⚠️ Trade already exists in array, skipping:', trade.transactionHash);
            return prev;
          }
          
          // Keep only last 100 trades to prevent memory issues
          const newTrades = [...prev, trade];
          return newTrades.slice(-100);
        });
        
        return newSeen;
      });
    });
    
    websocketService.connectToPredictions((prediction) => {
      console.log('🔮 Prediction received:', {
        hash: prediction.transaction_hash,
        predicted: prediction.predicted_label,
        probability: prediction.prediction_probability,
        true_label: prediction.true_label,
        agreement: prediction.agreement
      });
      
      setPredictions(prev => {
        const updated = {
          ...prev,
          [prediction.transaction_hash]: prediction
        };
        console.log('📊 Updated predictions object keys:', Object.keys(updated).slice(-5));
        return updated;
      });
      setLatestPrediction(prediction);
    });
    
    // Cleanup function
    return () => {
      console.log('🔌 Cleaning up WebSocket connections...');
      websocketService.disconnect();
    };
  }, []); // Empty dependency array - only run once on mount

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
            ML Model: XGBoost | Status: {isStreaming ? 'Active' : 'Stopped'}
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