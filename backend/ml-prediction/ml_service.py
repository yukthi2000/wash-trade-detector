import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any, Tuple
import time

logger = logging.getLogger(__name__)

class MLPredictionService:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'cut', 'blockNumber', 'timestamp', 'ether', 'token',
            'trade_amount_eth', 'trade_amount_dollar', 'trade_amount_token',
            'token_price_in_eth', 'eth_buyer_id', 'eth_seller_id'
        ]
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("ML model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Create DataFrame with required features
            features = {}
            for col in self.feature_columns:
                if col in trade_data:
                    features[col] = [trade_data[col]]
                else:
                    # Handle missing features with defaults
                    features[col] = [0.0]
            
            df = pd.DataFrame(features)
            
            # Apply scaling
            if self.scaler:
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df),
                    columns=df.columns
                )
                return df_scaled
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def predict(self, trade_data: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Make prediction on trade data
        Returns: (predicted_label, probability, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Prepare features
            features_df = self.prepare_features(trade_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            prediction_proba = self.model.predict_proba(features_df)[0]
            
            # Get probability for wash trade (class 1)
            wash_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return int(prediction), float(wash_probability), processing_time
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            processing_time = (time.time() - start_time) * 1000
            return 0, 0.0, processing_time
    
    def get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.8 or probability <= 0.2:
            return "High"
        elif probability >= 0.6 or probability <= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def calculate_metrics(self, true_labels: list, predicted_labels: list) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(true_labels) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        try:
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }