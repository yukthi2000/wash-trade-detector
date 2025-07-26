// Export interfaces first
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
  prediction_method?: string; // Added for ensemble tracking
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
  basic_ensemble_count?: number; // Added for ensemble tracking
  full_ensemble_count?: number;  // Added for ensemble tracking
}

export interface ComparisonBreakdown {
  both_wash: number;
  both_normal: number;
  only_true_wash: number;
  only_ml_wash: number;
}

// WebSocket Service Class
class WebSocketService {
  private tradeSocket: WebSocket | null = null;
  private predictionSocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connectToTrades(onMessage: (trade: Trade) => void, onError?: (error: Event) => void) {
    this.connectWithRetry(
      'ws://localhost:8001/ws/trades',
      (socket) => {
        this.tradeSocket = socket;
        socket.onmessage = (event) => {
          try {
            const data = event.data;
            
            // Skip ping/pong messages
            if (data === 'pong' || data === 'ping') {
              return;
            }
            
            const trade = JSON.parse(data);
            onMessage(trade);
          } catch (error) {
            console.error('Error parsing trade data:', error);
            console.error('Raw data:', event.data);
          }
        };
      },
      onError
    );
  }

  connectToPredictions(onMessage: (prediction: PredictionResult) => void, onError?: (error: Event) => void) {
    this.connectWithRetry(
      'ws://localhost:8002/ws/predictions',
      (socket) => {
        this.predictionSocket = socket;
        socket.onmessage = (event) => {
          try {
            const data = event.data;
            
            // Skip ping/pong messages
            if (data === 'pong' || data === 'ping') {
              return;
            }
            
            const prediction = JSON.parse(data);
            console.log('Received prediction:', prediction);
            onMessage(prediction);
          } catch (error) {
            console.error('Error parsing prediction data:', error);
            console.error('Raw data:', event.data);
          }
        };
      },
      onError
    );
  }

  private connectWithRetry(
    url: string,
    onConnect: (socket: WebSocket) => void,
    onError?: (error: Event) => void
  ) {
    const connect = () => {
      console.log(`Connecting to ${url}...`);
      const socket = new WebSocket(url);

      socket.onopen = () => {
        console.log(`Connected to ${url}`);
        this.reconnectAttempts = 0;
        onConnect(socket);
        
        // Send periodic ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send('ping');
          } else {
            clearInterval(pingInterval);
          }
        }, 30000);
      };

      socket.onclose = (event) => {
        console.log(`Connection to ${url} closed:`, event.code, event.reason);
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`Reconnecting to ${url} (attempt ${this.reconnectAttempts})...`);
          setTimeout(connect, this.reconnectDelay * this.reconnectAttempts);
        } else {
          console.error(`Failed to connect to ${url} after ${this.maxReconnectAttempts} attempts`);
        }
      };

      socket.onerror = (error) => {
        console.error(`WebSocket error for ${url}:`, error);
        if (onError) {
          onError(error);
        }
      };
    };

    connect();
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

  // Check connection status
  getConnectionStatus() {
    return {
      trades: this.tradeSocket?.readyState === WebSocket.OPEN,
      predictions: this.predictionSocket?.readyState === WebSocket.OPEN
    };
  }

  // Send message to specific socket
  sendToTrades(message: string) {
    if (this.tradeSocket?.readyState === WebSocket.OPEN) {
      this.tradeSocket.send(message);
    }
  }

  sendToPredictions(message: string) {
    if (this.predictionSocket?.readyState === WebSocket.OPEN) {
      this.predictionSocket.send(message);
    }
  }
}

// Export the service instance
export const websocketService = new WebSocketService();

// Default export for the service
export default websocketService;