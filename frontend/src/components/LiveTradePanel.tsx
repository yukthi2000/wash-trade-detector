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
    <Paper sx={{ p: 2, height: '400px', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Live Trade Stream
      </Typography>
      <TableContainer sx={{ maxHeight: '350px' }}>
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
            {trades.slice(-20).reverse().map((trade, index) => {
              const prediction = predictions[trade.transactionHash];
              return (
                <TableRow key={`${trade.transactionHash}-${index}`}>
                  <TableCell>{formatAddress(trade.transactionHash)}</TableCell>
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
                      <Chip label="Pending" color="default" size="small" />
                    )}
                  </TableCell>
                  <TableCell>
                    {prediction && (
                      <Chip
                        label={`${(prediction.prediction_probability * 100).toFixed(1)}%`}
                        color={getConfidenceColor(prediction.confidence_level, prediction.agreement)}
                        size="small"
                      />
                    )}
                  </TableCell>
                  <TableCell>
                    {prediction && (
                      <Chip
                        label={prediction.agreement ? '✓' : '✗'}
                        color={prediction.agreement ? 'success' : 'error'}
                        size="small"
                      />
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