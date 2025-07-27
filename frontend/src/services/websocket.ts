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
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: number | null = null;
  
  connectToTrades(onMessage: (trade: Trade) => void) {
    try {
      console.log('Connecting to trade WebSocket...');
      this.tradeSocket = new WebSocket('ws://localhost:8001/ws/trades');
      
      this.tradeSocket.onopen = () => {
        console.log('✅ Trade WebSocket connected');
        this.reconnectAttempts = 0;
        // Send a test message to keep connection alive
        if (this.tradeSocket) {
          this.tradeSocket.send('ping');
        }
      };
      
      this.tradeSocket.onmessage = (event) => {
        try {
          const trade = JSON.parse(event.data);
          console.log('📦 Received trade:', trade.transactionHash);
          onMessage(trade);
        } catch (error) {
          console.error('Error parsing trade data:', error);
        }
      };
      
      this.tradeSocket.onerror = (error) => {
        console.error('❌ Trade WebSocket error:', error);
      };
      
      this.tradeSocket.onclose = (event) => {
        console.log('🔌 Trade WebSocket closed:', event.code, event.reason);
        this.attemptReconnect('trades', onMessage);
      };
    } catch (error) {
      console.error('Failed to connect to trade WebSocket:', error);
    }
  }
  
  connectToPredictions(onMessage: (prediction: PredictionResult) => void) {
    try {
      console.log('Connecting to prediction WebSocket...');
      this.predictionSocket = new WebSocket('ws://localhost:8002/ws/predictions');
      
      this.predictionSocket.onopen = () => {
        console.log('✅ Prediction WebSocket connected');
        this.reconnectAttempts = 0;
        // Send a test message to keep connection alive
        if (this.predictionSocket) {
          this.predictionSocket.send('ping');
        }
      };
      
      this.predictionSocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Ignore ping messages
          if (data.type === 'ping') {
            return;
          }
          console.log('🔮 Received prediction:', data.transaction_hash);
          onMessage(data);
        } catch (error) {
          console.error('Error parsing prediction data:', error);
        }
      };
      
      this.predictionSocket.onerror = (error) => {
        console.error('❌ Prediction WebSocket error:', error);
      };
      
      this.predictionSocket.onclose = (event) => {
        console.log('🔌 Prediction WebSocket closed:', event.code, event.reason);
        this.attemptReconnect('predictions', onMessage);
      };
    } catch (error) {
      console.error('Failed to connect to prediction WebSocket:', error);
    }
  }
  
  private attemptReconnect(type: 'trades' | 'predictions', onMessage: any) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`🔄 Attempting to reconnect ${type} WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      this.reconnectTimeout = setTimeout(() => {
        if (type === 'trades') {
          this.connectToTrades(onMessage);
        } else {
          this.connectToPredictions(onMessage);
        }
      }, 2000 * this.reconnectAttempts);
    } else {
      console.error(`❌ Max reconnection attempts reached for ${type} WebSocket`);
    }
  }
  
  disconnect() {
    console.log('🔌 Disconnecting WebSockets...');
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
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
}

export const websocketService = new WebSocketService();