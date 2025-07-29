import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any, Tuple
import time
import joblib

logger = logging.getLogger(__name__)

class MLPredictionService:
    def __init__(self, model_path: str):
        self.model = None
        # Updated feature columns to match your training data
        self.feature_columns = [
            'cut', 'blockNumber', 'timestamp', 'trade_amount_eth', 
            'trade_amount_dollar', 'trade_amount_token', 'token_price_in_eth', 
            'eth_buyer_id', 'eth_seller_id'
        ]
        
        try:
            # Try multiple loading methods for compatibility
            self.model = self._load_model_with_fallback(model_path)
            logger.info("Random Forest model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_model_with_fallback(self, model_path: str):
        """Try multiple methods to load the model"""
        loading_methods = [
            # Method 1: Standard pickle
            lambda path: pickle.load(open(path, 'rb')),
            # Method 2: Joblib (recommended for sklearn models)
            lambda path: joblib.load(path),
            # Method 3: Pickle with different protocol
            lambda path: pickle.load(open(path, 'rb'), encoding='latin1'),
            # Method 4: Pickle with bytes mode
            lambda path: pickle.load(open(path, 'rb'), fix_imports=True, encoding='bytes'),
        ]
        
        for i, method in enumerate(loading_methods):
            try:
                logger.info(f"Trying loading method {i+1}...")
                model = method(model_path)
                logger.info(f"Successfully loaded model using method {i+1}")
                return model
            except Exception as e:
                logger.warning(f"Loading method {i+1} failed: {e}")
                continue
        
        raise Exception("All loading methods failed")
    
    def prepare_features(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction - Random Forest doesn't need scaling"""
        try:
            # Create DataFrame with required features
            features = {}
            for col in self.feature_columns:
                if col in trade_data:
                    features[col] = [trade_data[col]]
                else:
                    # Handle missing features with defaults
                    if col in ['cut', 'blockNumber', 'timestamp']:
                        features[col] = [0.0]
                    elif col in ['trade_amount_eth', 'trade_amount_dollar', 'trade_amount_token', 'token_price_in_eth']:
                        features[col] = [0.0]
                    elif col in ['eth_buyer_id', 'eth_seller_id']:
                        features[col] = [0]
                    else:
                        features[col] = [0.0]
            
            df = pd.DataFrame(features)
            
            # Ensure correct data types
            for col in ['eth_buyer_id', 'eth_seller_id']:
                df[col] = df[col].astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def predict(self, trade_data: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Make prediction on trade data using Random Forest
        Returns: (predicted_label, probability, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Prepare features (no scaling needed for Random Forest)
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