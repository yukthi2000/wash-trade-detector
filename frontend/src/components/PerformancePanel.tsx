import React, { useEffect, useState } from 'react';
import {
  Paper,
  Typography,
  Grid,
  Box,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import { api } from '../services/api';
import type { PerformanceMetrics, EnsembleStats } from '../services/websocket';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const PerformancePanel: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [ensembleStats, setEnsembleStats] = useState<EnsembleStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [serviceStatus, setServiceStatus] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setError(null);
        
        // Check service status first
        const status = await api.checkServices();
        setServiceStatus(status);
        
        if (status.mlService.status === 'offline') {
          setError('ML Prediction Service is offline');
          return;
        }
        
        // Fetch metrics and ensemble stats
        const [metricsResponse, ensembleResponse] = await Promise.allSettled([
          api.getMetrics(),
          api.getEnsembleStats()
        ]);
        
        if (metricsResponse.status === 'fulfilled') {
          setMetrics(metricsResponse.value.data);
        }
        
        if (ensembleResponse.status === 'fulfilled') {
          setEnsembleStats(ensembleResponse.value.data);
        }
        
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to fetch performance data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Paper sx={{ p: 2, height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2, height: '400px' }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        {serviceStatus && (
          <Box>
            <Typography variant="h6" gutterBottom>Service Status</Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <Chip 
                label={`Trade Service: ${serviceStatus.tradeService.status}`}
                color={serviceStatus.tradeService.status === 'healthy' ? 'success' : 'error'}
              />
              <Chip 
                label={`ML Service: ${serviceStatus.mlService.status}`}
                color={serviceStatus.mlService.status === 'healthy' ? 'success' : 'error'}
              />
            </Box>
            {serviceStatus.mlService.ensemble_status && (
              <Chip 
                label={`Ensemble: ${serviceStatus.mlService.ensemble_status}`}
                color={serviceStatus.mlService.ensemble_status === 'enabled' ? 'success' : 'warning'}
              />
            )}
          </Box>
        )}
      </Paper>
    );
  }

  if (!metrics) {
    return (
      <Paper sx={{ p: 2, height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography>No performance data available</Typography>
      </Paper>
    );
  }

  const performanceData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    datasets: [
      {
        data: [
          metrics.accuracy * 100,
          metrics.precision * 100,
          metrics.recall * 100,
          metrics.f1_score * 100,
        ],
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 99, 132, 0.8)',
          'rgba(255, 205, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(75, 192, 192, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const predictionData = {
    labels: ['Wash Trades', 'Normal Trades'],
    datasets: [
      {
        data: [metrics.wash_predictions, metrics.normal_predictions],
        backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(75, 192, 192, 0.8)'],
        borderColor: ['rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)'],
        borderWidth: 1,
      },
    ],
  };

  // Ensemble method breakdown chart
  const ensembleMethodData = ensembleStats ? {
    labels: ['Basic Only', 'Advanced', 'Full Ensemble'],
    datasets: [
      {
        data: [
          ensembleStats.basic_only_predictions,
          ensembleStats.advanced_predictions,
          ensembleStats.ensemble_predictions,
        ],
        backgroundColor: [
          'rgba(255, 206, 84, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
        borderColor: [
          'rgba(255, 206, 84, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  } : null;

  return (
    <Paper sx={{ p: 2, height: '500px' }}>
      <Typography variant="h6" gutterBottom>
        Enhanced Real-time Performance
        {serviceStatus?.mlService.ensemble_status === 'enabled' && (
          <Chip label="Ensemble Active" color="success" size="small" sx={{ ml: 1 }} />
        )}
      </Typography>
      <Grid container spacing={2} sx={{ height: '450px' }}>
        <Grid item xs={4}>
          <Box sx={{ height: '150px' }}>
            <Typography variant="subtitle2" align="center">
              Performance Metrics
            </Typography>
            <Bar
              data={performanceData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                  },
                },
                plugins: {
                  legend: {
                    display: false,
                  },
                },
              }}
            />
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ height: '150px' }}>
            <Typography variant="subtitle2" align="center">
              Prediction Distribution
            </Typography>
            <Doughnut
              data={predictionData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom',
                  },
                },
              }}
            />
          </Box>
        </Grid>
        {ensembleMethodData && (
          <Grid item xs={4}>
            <Box sx={{ height: '150px' }}>
              <Typography variant="subtitle2" align="center">
                Ensemble Methods Used
              </Typography>
              <Doughnut
                data={ensembleMethodData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                    },
                  },
                }}
              />
            </Box>
          </Grid>
        )}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={2}>
              <Box textAlign="center">
                <Typography variant="h4" color="primary">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Accuracy
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={2}>
              <Box textAlign="center">
                <Typography variant="h4" color="secondary">
                  {(metrics.agreement_rate * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Agreement Rate
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={2}>
              <Box textAlign="center">
                <Typography variant="h4" color="success.main">
                  {metrics.total_predictions}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Total Predictions
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={2}>
              <Box textAlign="center">
                <Typography variant="h4" color="error.main">
                  {metrics.wash_predictions}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Wash Detected
                </Typography>
              </Box>
            </Grid>
            {metrics.avg_processing_time_ms && (
              <Grid item xs={2}>
                <Box textAlign="center">
                  <Typography variant="h4" color="info.main">
                    {metrics.avg_processing_time_ms.toFixed(1)}ms
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Avg Processing
                  </Typography>
                </Box>
              </Grid>
            )}
            {ensembleStats && (
              <Grid item xs={2}>
                <Box textAlign="center">
                  <Typography variant="h4" color="warning.main">
                    {Object.values(ensembleStats.cache_size).reduce((a, b) => a + b, 0)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Cache Size
                  </Typography>
                </Box>
              </Grid>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PerformancePanel;