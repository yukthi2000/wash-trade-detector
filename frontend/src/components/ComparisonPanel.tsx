import React, { useEffect, useState } from 'react';
import {
  Paper,
  Typography,
  Grid,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
} from '@mui/material';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { api } from '../services/api';
import type { ComparisonBreakdown } from '../services/websocket';

ChartJS.register(ArcElement, Tooltip, Legend);

const ComparisonPanel: React.FC = () => {
  const [comparison, setComparison] = useState<ComparisonBreakdown | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchComparison = async () => {
      try {
        const response = await api.getComparison();
        setComparison(response.data);
      } catch (error) {
        console.error('Error fetching comparison:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchComparison();
    const interval = setInterval(fetchComparison, 2000);

    return () => clearInterval(interval);
  }, []);

  if (loading || !comparison) {
    return (
      <Paper sx={{ p: 2, height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Paper>
    );
  }

  const total = comparison.both_wash + comparison.both_normal + comparison.only_true_wash + comparison.only_ml_wash;
  const agreementRate = total > 0 ? ((comparison.both_wash + comparison.both_normal) / total * 100) : 0;

  const chartData = {
    labels: [
      'Both Agree (Wash)',
      'Both Agree (Normal)',
      'Only True Label (Wash)',
      'Only ML Detected (Wash)',
    ],
    datasets: [
      {
        data: [
          comparison.both_wash,
          comparison.both_normal,
          comparison.only_true_wash,
          comparison.only_ml_wash,
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)', // Red for wash agreements
          'rgba(75, 192, 192, 0.8)', // Green for normal agreements
          'rgba(255, 205, 86, 0.8)', // Yellow for true only
          'rgba(153, 102, 255, 0.8)', // Purple for ML only
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const comparisonData = [
    {
      category: 'Both methods agree (Wash)',
      trueLabel: 'WASH',
      mlPrediction: 'WASH',
      count: comparison.both_wash,
      color: 'error',
    },
    {
      category: 'Both methods agree (Normal)',
      trueLabel: 'NORMAL',
      mlPrediction: 'NORMAL',
      count: comparison.both_normal,
      color: 'success',
    },
    {
      category: 'Only True Label found wash',
      trueLabel: 'WASH',
      mlPrediction: 'NORMAL',
      count: comparison.only_true_wash,
      color: 'warning',
    },
    {
      category: 'Only ML found wash',
      trueLabel: 'NORMAL',
      mlPrediction: 'WASH',
      count: comparison.only_ml_wash,
      color: 'info',
    },
  ];

  return (
    <Paper sx={{ p: 2, height: '400px' }}>
      <Typography variant="h6" gutterBottom>
        Comparison Analysis
      </Typography>
      <Grid container spacing={2} sx={{ height: '350px' }}>
        <Grid item xs={6}>
          <Box sx={{ height: '200px' }}>
            <Typography variant="subtitle2" align="center" gutterBottom>
              Agreement Breakdown
            </Typography>
            <Doughnut
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom',
                    labels: {
                      font: {
                        size: 10,
                      },
                    },
                  },
                },
              }}
            />
          </Box>
          <Box textAlign="center" sx={{ mt: 2 }}>
            <Typography variant="h4" color="primary">
              {agreementRate.toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Overall Agreement Rate
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <TableContainer sx={{ maxHeight: '320px' }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Category</TableCell>
                  <TableCell>True</TableCell>
                  <TableCell>ML</TableCell>
                  <TableCell>Count</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {comparisonData.map((row, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                        {row.category}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={row.trueLabel}
                        color={row.trueLabel === 'WASH' ? 'error' : 'success'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={row.mlPrediction}
                        color={row.mlPrediction === 'WASH' ? 'error' : 'success'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="h6" color={`${row.color}.main`}>
                        {row.count}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ComparisonPanel;