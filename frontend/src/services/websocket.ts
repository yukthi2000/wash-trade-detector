import io, { Socket } from 'socket.io-client';

export interface Trade {
  eth_seller: string;
  eth_buyer: string;
  date: string;
  cut: number;
  blockNumber: number;
  timestamp: number;
  transactionHash: string;
  ether: number;
  token: number;
  trade_amount_eth: number;
  trade_amount_dollar: number;
  trade_amount_token: number;
  token_price_in_eth: number;
  eth_buyer_id: number;
  eth_seller_id: number;
  wash_label: number;
}

export interface PredictionResult {
  transaction_hash: string;
  true_label: number;
  predicted_label: number;
  prediction_probability: number;
  processing_time_ms: number;
  agreement: boolean;
  confidence_level: string;
  prediction_method?: string; // New field
  individual_predictions?: Record<string, number>; // New field
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  total_predictions: number;
  wash_predictions: number;
  normal_predictions: number;
  agreement_rate: number;
  // Enhanced metrics
  auc_roc?: number;
  avg_precision?: number;
  avg_processing_time_ms?: number;
  prediction_method_breakdown?: Record<string, number>;
}

export interface ComparisonBreakdown {
  both_wash: number;
  both_normal: number;
  only_true_wash: number;
  only_ml_wash: number;
}

export interface EnsembleStats {
  total_predictions: number;
  basic_only_predictions: number;
  advanced_predictions: number;
  ensemble_predictions: number;
  cache_size: Record<string, number>;
  avg_processing_time_ms: number;
}

class WebSocketService {
  private tradeSocket: WebSocket | null = null;
  private predictionSocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  
  connectToTrades(onMessage: (trade: Trade) => void, onError?: (error: Event) => void) {
    try {
      this.tradeSocket = new WebSocket('ws://localhost:8001/ws/trades');
      
      this.tradeSocket.onopen = () => {
        console.log('Connected to trade WebSocket');
        this.reconnectAttempts = 0;
      };
      
      this.tradeSocket.onmessage = (event) => {
        try {
          const trade = JSON.parse(event.data);
          onMessage(trade);
        } catch (error) {
          console.error('Error parsing trade data:', error);
        }
      };
      
      this.tradeSocket.onerror = (error) => {
        console.error('Trade WebSocket error:', error);
        if (onError) onError(error);
      };
      
      this.tradeSocket.onclose = () => {
        console.log('Trade WebSocket closed');
        this.attemptReconnect('trades', onMessage, onError);
      };
    } catch (error) {
      console.error('Failed to connect to trade WebSocket:', error);
      if (onError) onError(error as Event);
    }
  }
  
  connectToPredictions(onMessage: (prediction: PredictionResult) => void, onError?: (error: Event) => void) {
    try {
      this.predictionSocket = new WebSocket('ws://localhost:8002/ws/predictions');
      
      this.predictionSocket.onopen = () => {
        console.log('Connected to prediction WebSocket');
        this.reconnectAttempts = 0;
      };
      
      this.predictionSocket.onmessage = (event) => {
        try {
          const prediction = JSON.parse(event.data);
          onMessage(prediction);
        } catch (error) {
          console.error('Error parsing prediction data:', error);
        }
      };
      
      this.predictionSocket.onerror = (error) => {
        console.error('Prediction WebSocket error:', error);
        if (onError) onError(error);
      };
      
      this.predictionSocket.onclose = () => {
        console.log('Prediction WebSocket closed');
        this.attemptReconnect('predictions', onMessage, onError);
      };
    } catch (error) {
      console.error('Failed to connect to prediction WebSocket:', error);
      if (onError) onError(error as Event);
    }
  }
  
  private attemptReconnect(type: 'trades' | 'predictions', onMessage: any, onError?: (error: Event) => void) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect to ${type} WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        if (type === 'trades') {
          this.connectToTrades(onMessage, onError);
        } else {
          this.connectToPredictions(onMessage, onError);
        }
      }, this.reconnectDelay);
    } else {
      console.error(`Max reconnection attempts reached for ${type} WebSocket`);
    }
  }
  
  disconnect() {
    if (this.tradeSocket) {
      this.tradeSocket.close();
      this.tradeSocket = null;
    }
    if (this.predictionSocket) {
      this.predictionSocket.close();
      this.predictionSocket = null;
    }
    this.reconnectAttempts = 0;
  }
  
  isConnected(): { trades: boolean; predictions: boolean } {
    return {
      trades: this.tradeSocket?.readyState === WebSocket.OPEN,
      predictions: this.predictionSocket?.readyState === WebSocket.OPEN
    };
  }
}

export const websocketService = new WebSocketService();