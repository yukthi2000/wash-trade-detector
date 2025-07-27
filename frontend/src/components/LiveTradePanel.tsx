import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import type { Trade, PredictionResult } from '../services/websocket';

interface LiveTradePanelProps {
  trades: Trade[];
  predictions: { [key: string]: PredictionResult };
}

const LiveTradePanel: React.FC<LiveTradePanelProps> = ({ trades, predictions }) => {
  // Debug logging
  React.useEffect(() => {
    console.log('🔍 LiveTradePanel updated:', {
      tradesCount: trades.length,
      predictionsCount: Object.keys(predictions).length,
      latestTrade: trades[trades.length - 1]?.transactionHash,
      predictionKeys: Object.keys(predictions).slice(-5) // Last 5 prediction keys
    });
  }, [trades, predictions]);

  const getConfidenceColor = (confidence: string, agreement: boolean) => {
    if (!agreement) return 'error';
    switch (confidence) {
      case 'High': return 'success';
      case 'Medium': return 'warning';
      case 'Low': return 'info';
      default: return 'default';
    }
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  return (
    <Paper sx={{ p: 2, height: '550px', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Live Trade Stream
        <Typography variant="caption" sx={{ ml: 2 }}>
          ({trades.length} trades, {Object.keys(predictions).length} predictions)
        </Typography>
      </Typography>
      <TableContainer sx={{ maxHeight: '450px' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>Hash</TableCell>
              <TableCell>Seller</TableCell>
              <TableCell>Buyer</TableCell>
              <TableCell>Amount (ETH)</TableCell>
              <TableCell>True Label</TableCell>
              <TableCell>ML Prediction</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Agreement</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trades.reverse().map((trade, index) => {
              const prediction = predictions[trade.transactionHash];
              
              // Debug log for each trade
              if (index < 3) { // Only log first 3 to avoid spam
                console.log(`🔍 Trade ${trade.transactionHash}:`, {
                  tradeHash: trade.transactionHash,
                  hasPrediction: !!prediction,
                  predictionHash: prediction?.transaction_hash,
                  hashesMatch: trade.transactionHash === prediction?.transaction_hash
                });
              }
              
              return (
                <TableRow key={trade.transactionHash}>
                  <TableCell>
                    <Typography variant="caption">
                      {formatAddress(trade.transactionHash)}
                    </Typography>
                  </TableCell>
                  <TableCell>{formatAddress(trade.eth_seller)}</TableCell>
                  <TableCell>{formatAddress(trade.eth_buyer)}</TableCell>
                  <TableCell>{trade.trade_amount_eth.toFixed(4)}</TableCell>
                  <TableCell>
                    <Chip
                      label={trade.wash_label === 1 ? 'WASH' : 'NORMAL'}
                      color={trade.wash_label === 1 ? 'error' : 'success'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {prediction ? (
                      <Chip
                        label={prediction.predicted_label === 1 ? 'WASH' : 'NORMAL'}
                        color={prediction.predicted_label === 1 ? 'error' : 'success'}
                        size="small"
                      />
                    ) : (
                      <Chip 
                        label="Pending" 
                        color="default" 
                        size="small"
                        title={`Looking for prediction with hash: ${trade.transactionHash}`}
                      />
                    )}
                  </TableCell>
                  <TableCell>
                    {prediction ? (
                      <Chip
                        label={`${(prediction.prediction_probability * 100).toFixed(1)}%`}
                        color={getConfidenceColor(prediction.confidence_level, prediction.agreement)}
                        size="small"
                      />
                    ) : (
                      <Typography variant="caption" color="textSecondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {prediction ? (
                      <Chip
                        label={prediction.agreement ? '✓' : '✗'}
                        color={prediction.agreement ? 'success' : 'error'}
                        size="small"
                      />
                    ) : (
                      <Typography variant="caption" color="textSecondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default LiveTradePanel;