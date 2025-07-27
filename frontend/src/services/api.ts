import axios from 'axios';
import type { PerformanceMetrics, ComparisonBreakdown } from '../services/websocket';

const TRADE_API_BASE = 'http://localhost:8001';
const ML_API_BASE = 'http://localhost:8002';

export const api = {
  // Trade Ingestion Service
  startStreaming: () => axios.post(`${TRADE_API_BASE}/start-streaming`),
  stopStreaming: () => axios.post(`${TRADE_API_BASE}/stop-streaming`),
  
  // ML Prediction Service
  startProcessing: () => axios.post(`${ML_API_BASE}/start-processing`),
  stopProcessing: () => axios.post(`${ML_API_BASE}/stop-processing`),
  setThreshold: (threshold: number) => axios.post(`${ML_API_BASE}/set-threshold/${threshold}`),
  
  // Health checks
  getHealth: () => axios.get(`${ML_API_BASE}/health`),
  getModelInfo: () => axios.get(`${ML_API_BASE}/model-info`),
  
  getMetrics: (): Promise<{ data: PerformanceMetrics }> => 
    axios.get(`${ML_API_BASE}/metrics`),
  
  getComparison: (): Promise<{ data: ComparisonBreakdown }> => 
    axios.get(`${ML_API_BASE}/comparison`),
};