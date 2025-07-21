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
}

export interface ComparisonBreakdown {
  both_wash: number;
  both_normal: number;
  only_true_wash: number;
  only_ml_wash: number;
}

class WebSocketService {
  private tradeSocket: WebSocket | null = null;
  private predictionSocket: WebSocket | null = null;
  
  connectToTrades(onMessage: (trade: Trade) => void) {
    this.tradeSocket = new WebSocket('ws://localhost:8001/ws/trades');
    
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
    };
  }
  
  connectToPredictions(onMessage: (prediction: PredictionResult) => void) {
    this.predictionSocket = new WebSocket('ws://localhost:8002/ws/predictions');
    
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
    };
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
  }
}

export const websocketService = new WebSocketService();