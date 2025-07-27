import axios from 'axios';
import type { PerformanceMetrics, ComparisonBreakdown, EnsembleStats } from '../services/websocket';

const TRADE_API_BASE = 'http://localhost:8001';
const ML_API_BASE = 'http://localhost:8002';

export const api = {
  // Trade Ingestion Service
  startStreaming: () => axios.post(`${TRADE_API_BASE}/start-streaming`),
  stopStreaming: () => axios.post(`${TRADE_API_BASE}/stop-streaming`),
  
  // Enhanced ML Prediction Service
  startProcessing: () => axios.post(`${ML_API_BASE}/start-processing`),
  stopProcessing: () => axios.post(`${ML_API_BASE}/stop-processing`),
  setThreshold: (threshold: number) => axios.post(`${ML_API_BASE}/set-threshold/${threshold}`),
  
  // Enhanced metrics endpoints
  getMetrics: (): Promise<{ data: PerformanceMetrics }> => 
    axios.get(`${ML_API_BASE}/metrics`),
  
  getComparison: (): Promise<{ data: ComparisonBreakdown }> => 
    axios.get(`${ML_API_BASE}/comparison`),
  
  // New ensemble-specific endpoint
  getEnsembleStats: (): Promise<{ data: EnsembleStats }> => 
    axios.get(`${ML_API_BASE}/ensemble-stats`),
  
  // Health check
  getHealth: () => axios.get(`${ML_API_BASE}/health`),
  
  // Service status check
  checkServices: async () => {
    try {
      const [tradeHealth, mlHealth] = await Promise.allSettled([
        axios.get(`${TRADE_API_BASE}/health`).catch(() => ({ data: { status: 'offline' } })),
        axios.get(`${ML_API_BASE}/health`).catch(() => ({ data: { status: 'offline' } }))
      ]);
      
      return {
        tradeService: tradeHealth.status === 'fulfilled' ? tradeHealth.value.data : { status: 'offline' },
        mlService: mlHealth.status === 'fulfilled' ? mlHealth.value.data : { status: 'offline' }
      };
    } catch (error) {
      return {
        tradeService: { status: 'offline' },
        mlService: { status: 'offline' }
      };
    }
  }
};