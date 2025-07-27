import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any, Tuple
import time
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class MLPredictionService:
    def __init__(self, model_path: str):
        self.model = None
        self.best_params = {}
        self.results = {}
        self.feature_columns = [
            'cut', 'blockNumber', 'timestamp', 
            'trade_amount_eth', 'trade_amount_dollar', 'trade_amount_token',
            'token_price_in_eth', 'eth_buyer_id', 'eth_seller_id'
        ]
        
        try:
            logger.info(f"Loading XGBoost model from: {model_path}")
            
            # Suppress the XGBoost version warning during loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
            logger.info(f"Loaded data type: {type(model_data)}")
                
            # Handle the case where model is stored directly (your case)
            if hasattr(model_data, 'predict') and hasattr(model_data, 'predict_proba'):
                # Model is stored directly as XGBClassifier
                logger.info("Model stored directly as XGBClassifier object")
                self.model = model_data
                self.best_params = {}
                self.results = {}
                
                # Try to get some model info if available
                if hasattr(self.model, 'get_params'):
                    self.best_params = self.model.get_params()
                    
            elif isinstance(model_data, dict):
                # If it's a dictionary (fallback)
                logger.info(f"Model stored as dictionary with keys: {list(model_data.keys())}")
                self.model = model_data.get('model')
                self.best_params = model_data.get('best_params', {})
                self.results = model_data.get('results', {})
                
                if self.model is None:
                    raise ValueError("No 'model' key found in dictionary")
            else:
                raise ValueError(f"Unexpected model data type: {type(model_data)}")
            
            # Verify model has required methods
            if not hasattr(self.model, 'predict'):
                raise ValueError("Model does not have 'predict' method")
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model does not have 'predict_proba' method")
            
            logger.info(f"✅ XGBoost model loaded successfully!")
            logger.info(f"✅ Model type: {type(self.model)}")
            logger.info(f"✅ Model class: {self.model.__class__.__name__}")
            
            # Test prediction with dummy data to ensure it works
            dummy_data = pd.DataFrame([[0.0] * len(self.feature_columns)], columns=self.feature_columns)
            test_pred = self.model.predict(dummy_data)
            test_proba = self.model.predict_proba(dummy_data)
            logger.info(f"✅ Model test successful - prediction shape: {test_pred.shape}, proba shape: {test_proba.shape}")
                           
        except Exception as e:
            logger.error(f"❌ Error loading XGBoost model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def prepare_features(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for XGBoost prediction (no scaling needed)"""
        try:
            # Create DataFrame with required features
            features = {}
            
            logger.debug(f"Input trade data keys: {list(trade_data.keys())}")
            
            for col in self.feature_columns:
                if col in trade_data:
                    # Convert to appropriate type
                    value = trade_data[col]
                    if value is None:
                        value = 0
                    
                    if col in ['eth_buyer_id', 'eth_seller_id']:
                        features[col] = [int(float(value))]
                    else:
                        features[col] = [float(value)]
                else:
                    # Handle missing features with defaults
                    logger.debug(f"Missing feature '{col}', using default value")
                    if col in ['eth_buyer_id', 'eth_seller_id']:
                        features[col] = [0]
                    else:
                        features[col] = [0.0]
            
            df = pd.DataFrame(features)
            
            # Ensure correct data types
            for col in self.feature_columns:
                if col in ['eth_buyer_id', 'eth_seller_id']:
                    df[col] = df[col].astype('int64')
                else:
                    df[col] = df[col].astype('float64')
            
            # Ensure column order matches training
            df = df[self.feature_columns]
            
            logger.debug(f"Prepared features shape: {df.shape}")
            logger.debug(f"Prepared features dtypes: {df.dtypes.to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            logger.error(f"Trade data received: {trade_data}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, trade_data: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Make prediction on trade data using XGBoost
        Returns: (predicted_label, probability, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Prepare features (no scaling for XGBoost)
            features_df = self.prepare_features(trade_data)
            
            logger.debug(f"Making prediction with features shape: {features_df.shape}")
            
            # Make prediction with XGBoost
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = self.model.predict(features_df)[0]
                prediction_proba = self.model.predict_proba(features_df)[0]
            
            # Get probability for wash trade (class 1)
            wash_probability = float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.debug(f"✅ XGBoost prediction: {int(prediction)}, probability: {wash_probability:.4f}, time: {processing_time:.2f}ms")
            
            return int(prediction), wash_probability, processing_time
            
        except Exception as e:
            logger.error(f"❌ Error making XGBoost prediction: {e}")
            logger.error(f"Model type: {type(self.model)}")
            if 'features_df' in locals():
                logger.error(f"Features shape: {features_df.shape}")
                logger.error(f"Features dtypes: {features_df.dtypes}")
                logger.error(f"Features values: {features_df.iloc[0].to_dict()}")
            import traceback
            logger.error(traceback.format_exc())
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'XGBoost',
            'model_class': str(type(self.model)),
            'parameters': self.best_params,
            'performance': self.results,
            'features': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'loaded': self.model is not None
        }
    
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