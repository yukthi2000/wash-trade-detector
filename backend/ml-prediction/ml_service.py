import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import logging
from typing import Dict, Any, Tuple, Optional
import time
import redis
from datetime import datetime
from collections import defaultdict
import asyncio
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

logger = logging.getLogger(__name__)

class HistoricalCache:
    """Manages historical data cache for advanced feature engineering"""
    
    def __init__(self, redis_client=None, cache_days=30):
        self.redis_client = redis_client
        self.cache_days = cache_days
        self.local_cache = {
            'address_stats': {},
            'token_stats': {},
            'recent_trades': defaultdict(list),
            'last_update': {}
        }
        
    def add_trade(self, trade_data: Dict):
        """Add new trade to cache"""
        try:
            seller = trade_data.get('eth_seller', '')
            buyer = trade_data.get('eth_buyer', '')
            token = trade_data.get('token', '')
            timestamp = trade_data.get('timestamp', 0)
            
            # Update recent trades for time gap calculation
            self.local_cache['recent_trades'][seller].append({
                'timestamp': timestamp,
                'hash': trade_data.get('transactionHash', '')
            })
            
            # Keep only recent trades (last 1000 per address)
            if len(self.local_cache['recent_trades'][seller]) > 1000:
                self.local_cache['recent_trades'][seller] = \
                    self.local_cache['recent_trades'][seller][-1000:]
            
            # Update address counts
            if seller not in self.local_cache['address_stats']:
                self.local_cache['address_stats'][seller] = {'count': 0, 'first_seen': timestamp}
            if buyer not in self.local_cache['address_stats']:
                self.local_cache['address_stats'][buyer] = {'count': 0, 'first_seen': timestamp}
                
            self.local_cache['address_stats'][seller]['count'] += 1
            self.local_cache['address_stats'][buyer]['count'] += 1
            
            # Update token stats
            if token not in self.local_cache['token_stats']:
                self.local_cache['token_stats'][token] = {
                    'prices': [],
                    'mean': 0,
                    'std': 1,
                    'median': 0
                }
            
            price = trade_data.get('token_price_in_eth', 0)
            if price > 0:
                self.local_cache['token_stats'][token]['prices'].append(price)
                
                # Keep only recent prices (last 1000)
                if len(self.local_cache['token_stats'][token]['prices']) > 1000:
                    self.local_cache['token_stats'][token]['prices'] = \
                        self.local_cache['token_stats'][token]['prices'][-1000:]
                
                # Recalculate token statistics
                prices = self.local_cache['token_stats'][token]['prices']
                self.local_cache['token_stats'][token]['mean'] = np.mean(prices)
                self.local_cache['token_stats'][token]['std'] = np.std(prices) or 1
                self.local_cache['token_stats'][token]['median'] = np.median(prices)
                
        except Exception as e:
            logger.error(f"Error adding trade to cache: {str(e)}")
    
    def get_address_frequency(self, address: str) -> int:
        """Get address frequency from cache"""
        return self.local_cache['address_stats'].get(address, {}).get('count', 1)
    
    def get_address_velocity(self, address: str) -> float:
        """Calculate address velocity"""
        try:
            trades = self.local_cache['recent_trades'].get(address, [])
            if len(trades) < 2:
                return 0.0
            
            timestamps = [t['timestamp'] for t in trades]
            time_span = (max(timestamps) - min(timestamps)) / 3600  # hours
            return len(trades) / (time_span + 1)
        except:
            return 0.0
    
    def get_time_gap(self, seller: str, current_timestamp: int) -> float:
        """Get time gap since last trade"""
        try:
            trades = self.local_cache['recent_trades'].get(seller, [])
            if not trades:
                return 3600  # Default 1 hour
            
            last_timestamp = max(t['timestamp'] for t in trades)
            return current_timestamp - last_timestamp
        except:
            return 3600
    
    def get_token_stats(self, token: str) -> Dict:
        """Get token price statistics"""
        default_stats = {'mean': 0.001, 'std': 0.001, 'median': 0.001}
        return self.local_cache['token_stats'].get(token, default_stats)

class MLPredictionService:
    def __init__(self, model_path: str, scaler_path: str = None):
        """Initialize with ensemble models"""
        self.ensemble_package = None
        self.models = {}
        self.scaler_basic = None
        self.meta_learner = None
        self.basic_features = None
        self.advanced_features = None
        self.cache = HistoricalCache()
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'basic_only': 0,
            'advanced': 0,
            'ensemble': 0,
            'processing_times': [],
            'last_method': 'ensemble'
        }
        
        # Legacy feature columns for backward compatibility
        self.feature_columns = [
            'cut', 'blockNumber', 'timestamp', 'ether', 'token',
            'trade_amount_eth', 'trade_amount_dollar', 'trade_amount_token',
            'token_price_in_eth', 'eth_buyer_id', 'eth_seller_id'
        ]
        
        try:
            # Load ensemble package
            self.ensemble_package = joblib.load(model_path)
            
            # Extract individual models
            self.models = {
                'neural_network_basic': self.ensemble_package['neural_network_basic'],
                'xgboost_basic': self.ensemble_package['xgboost_basic'],
                'random_forest_advanced': self.ensemble_package['random_forest_advanced'],
                'xgboost_advanced': self.ensemble_package['xgboost_advanced']
            }
            
            # Extract other components
            self.scaler_basic = self.ensemble_package['scaler_basic']
            self.meta_learner = self.ensemble_package['meta_learner']
            self.basic_features = self.ensemble_package['basic_features']
            self.advanced_features = self.ensemble_package['advanced_features']
            
            logger.info("Ensemble ML models loaded successfully")
            logger.info(f"Basic features: {len(self.basic_features)}")
            logger.info(f"Advanced features: {len(self.advanced_features)}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
            # Fallback to single model if available
            try:
                if scaler_path:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Fallback: Single model loaded")
                else:
                    raise
            except:
                raise Exception(f"Failed to load any models: {e}")
    
    def _engineer_basic_features(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """Engineer basic features from trade data - returns DataFrame with proper column names"""
        try:
            # Create basic feature vector
            basic_dict = {}
            for feature in self.basic_features:
                value = trade_data.get(feature, 0)
                basic_dict[feature] = float(value)
            
            # Create DataFrame with proper column names
            df = pd.DataFrame([basic_dict])
            
            # Handle missing values
            df = df.fillna(0.0)
            df = df.replace([np.inf, -np.inf], 0.0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering basic features: {str(e)}")
            # Return default DataFrame with proper column names
            default_dict = {feature: 0.0 for feature in self.basic_features}
            return pd.DataFrame([default_dict])
    
    def _engineer_advanced_features(self, trade_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Engineer advanced features using cached historical data - returns DataFrame with proper column names"""
        try:
            # Extract trade information
            seller = trade_data.get('eth_seller', '')
            buyer = trade_data.get('eth_buyer', '')
            token = trade_data.get('token', '')
            timestamp = trade_data.get('timestamp', 0)
            
            # Get cached statistics
            seller_freq = self.cache.get_address_frequency(seller)
            buyer_freq = self.cache.get_address_frequency(buyer)
            seller_velocity = self.cache.get_address_velocity(seller)
            buyer_velocity = self.cache.get_address_velocity(buyer)
            time_gap = self.cache.get_time_gap(seller, timestamp)
            token_stats = self.cache.get_token_stats(token)
            
            # Create datetime features
            try:
                dt = datetime.fromtimestamp(timestamp)
                hour = dt.hour
                day_of_week = dt.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
            except:
                hour = 12
                day_of_week = 1
                is_weekend = 0
            
            # Calculate derived features
            address_frequency_ratio = seller_freq / (buyer_freq + 1)
            
            # Price deviation features
            price = trade_data.get('token_price_in_eth', 0.001)
            price_deviation_from_mean = (price - token_stats['mean']) / (token_stats['std'] + 1e-8)
            price_deviation_from_median = abs(price - token_stats['median']) / (token_stats['median'] + 1e-8)
            
            # Round number detection
            def is_suspicious_round(value, threshold=0.1):
                if value == 0:
                    return 0
                try:
                    log_val = np.log10(abs(value))
                    return int(abs(log_val - round(log_val)) < threshold)
                except:
                    return 0
            
            eth_suspicious_round = is_suspicious_round(trade_data.get('trade_amount_eth', 0))
            dollar_suspicious_round = is_suspicious_round(trade_data.get('trade_amount_dollar', 0))
            token_suspicious_round = is_suspicious_round(trade_data.get('trade_amount_token', 0))
            
            # Time-based features
            is_rapid_trade = 1 if time_gap < 300 else 0
            is_very_rapid_trade = 1 if time_gap < 60 else 0
            
            # Volume z-scores (using simple approximation)
            eth_amount = trade_data.get('trade_amount_eth', 0)
            dollar_amount = trade_data.get('trade_amount_dollar', 0)
            token_amount = trade_data.get('trade_amount_token', 0)
            
            eth_volume_zscore = (eth_amount - 0.1) / 1.0  # Simplified
            dollar_volume_zscore = (dollar_amount - 100) / 1000  # Simplified
            token_volume_zscore = (token_amount - 1000) / 10000  # Simplified
            
            is_volume_anomaly = 1 if (abs(eth_volume_zscore) > 3 or 
                                    abs(dollar_volume_zscore) > 3 or 
                                    abs(token_volume_zscore) > 3) else 0
            
            # Interaction features
            frequency_velocity_interaction = seller_freq * seller_velocity
            price_volume_interaction = price * token_amount
            time_volume_interaction = time_gap * eth_amount
            
            # Basic derived features
            eth_to_dollar_ratio = eth_amount / (dollar_amount + 1e-8)
            token_to_eth_ratio = token_amount / (eth_amount + 1e-8)
            log_trade_amount_eth = np.log1p(eth_amount)
            log_trade_amount_dollar = np.log1p(dollar_amount)
            log_trade_amount_token = np.log1p(token_amount)
            
            # Create feature dictionary in correct order
            advanced_dict = {}
            advanced_values = [
                eth_amount,
                dollar_amount,
                token_amount,
                price,
                trade_data.get('blockNumber', 0),
                timestamp,
                eth_to_dollar_ratio,
                token_to_eth_ratio,
                log_trade_amount_eth,
                log_trade_amount_dollar,
                log_trade_amount_token,
                hour,
                day_of_week,
                is_weekend,
                seller_freq,
                buyer_freq,
                address_frequency_ratio,
                seller_velocity,
                buyer_velocity,
                price_deviation_from_mean,
                price_deviation_from_median,
                eth_suspicious_round,
                dollar_suspicious_round,
                token_suspicious_round,
                time_gap,
                is_rapid_trade,
                is_very_rapid_trade,
                eth_volume_zscore,
                dollar_volume_zscore,
                token_volume_zscore,
                is_volume_anomaly,
                frequency_velocity_interaction,
                price_volume_interaction,
                time_volume_interaction
            ]
            
            # Map values to feature names
            for i, feature_name in enumerate(self.advanced_features):
                advanced_dict[feature_name] = float(advanced_values[i])
            
            # Create DataFrame with proper column names
            df = pd.DataFrame([advanced_dict])
            
            # Handle missing values
            df = df.fillna(0.0)
            df = df.replace([np.inf, -np.inf], 0.0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering advanced features: {str(e)}")
            return None
    
    def prepare_features(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        try:
            # Use basic features for legacy compatibility
            basic_df = self._engineer_basic_features(trade_data)
            
            # Apply scaling if available
            if self.scaler_basic:
                # Create scaled DataFrame with proper column names
                scaled_values = self.scaler_basic.transform(basic_df)
                df_scaled = pd.DataFrame(scaled_values, columns=basic_df.columns)
                return df_scaled
            
            return basic_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return default DataFrame
            default_dict = {feature: 0.0 for feature in self.basic_features}
            return pd.DataFrame([default_dict])
    
    def predict(self, trade_data: Dict[str, Any]) -> Tuple[int, float, float]:
        """
        Enhanced prediction using hybrid ensemble approach
        Returns: (predicted_label, probability, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Check if ensemble models are available
            if not self.ensemble_package:
                # Fallback to legacy single model
                return self._predict_legacy(trade_data, start_time)
            
            # Step 1: Always get basic prediction (fast)
            basic_features_df = self._engineer_basic_features(trade_data)
            basic_features_scaled_df = pd.DataFrame(
                self.scaler_basic.transform(basic_features_df),
                columns=basic_features_df.columns
            )
            
            # Get basic model predictions
            nn_basic_proba = self.models['neural_network_basic'].predict_proba(basic_features_scaled_df)[0, 1]
            xgb_basic_proba = self.models['xgboost_basic'].predict_proba(basic_features_df)[0, 1]
            
            individual_predictions = {
                'neural_network_basic': float(nn_basic_proba),
                'xgboost_basic': float(xgb_basic_proba)
            }
            
            # Step 2: Try to get advanced prediction
            advanced_features_df = self._engineer_advanced_features(trade_data)
            
            if advanced_features_df is not None:
                # Get advanced model predictions
                rf_adv_proba = self.models['random_forest_advanced'].predict_proba(advanced_features_df)[0, 1]
                xgb_adv_proba = self.models['xgboost_advanced'].predict_proba(advanced_features_df)[0, 1]
                
                individual_predictions.update({
                    'random_forest_advanced': float(rf_adv_proba),
                    'xgboost_advanced': float(xgb_adv_proba)
                })
                
                # Step 3: Use meta-learner for final ensemble prediction
                meta_features = np.array([[nn_basic_proba, xgb_basic_proba, rf_adv_proba, xgb_adv_proba]])
                final_probability = self.meta_learner.predict_proba(meta_features)[0, 1]
                prediction_method = "ensemble"
                self.prediction_stats['ensemble'] += 1
                
            else:
                # Fallback to basic models average
                final_probability = (nn_basic_proba + xgb_basic_proba) / 2
                prediction_method = "basic_only"
                self.prediction_stats['basic_only'] += 1
            
            # Final prediction
            predicted_label = 1 if final_probability > 0.5 else 0
            
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            self.prediction_stats['total_predictions'] += 1
            self.prediction_stats['processing_times'].append(processing_time)
            self.prediction_stats['last_method'] = prediction_method
            
            # Keep only recent processing times
            if len(self.prediction_stats['processing_times']) > 1000:
                self.prediction_stats['processing_times'] = \
                    self.prediction_stats['processing_times'][-1000:]
            
            # Update cache with new trade (async)
            try:
                self.cache.add_trade(trade_data)
            except Exception as e:
                logger.error(f"Error updating cache: {str(e)}")
            
            return predicted_label, float(final_probability), processing_time
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            processing_time = (time.time() - start_time) * 1000
            return 0, 0.0, processing_time
    
    def _predict_legacy(self, trade_data: Dict[str, Any], start_time: float) -> Tuple[int, float, float]:
        """Legacy prediction method for fallback"""
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
            logger.error(f"Error making legacy prediction: {e}")
            processing_time = (time.time() - start_time) * 1000
            return 0, 0.0, processing_time
    
    def get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.9 or probability <= 0.1:
            return "High"
        elif probability >= 0.7 or probability <= 0.3:
            return "Medium"
        else:
            return "Low"
    
    def calculate_metrics(self, true_labels: list, predicted_labels: list) -> Dict[str, float]:
        """Enhanced metrics calculation"""
        if len(true_labels) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'avg_precision': 0.0
            }
        
        try:
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            
            # Additional metrics if probabilities are available
            auc_roc = 0.0
            avg_precision = 0.0
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
                'avg_precision': float(avg_precision)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'avg_precision': 0.0
            }
    
    def get_prediction_stats(self) -> Dict:
        """Get enhanced prediction statistics"""
        avg_processing_time = 0.0
        if self.prediction_stats['processing_times']:
            avg_processing_time = np.mean(self.prediction_stats['processing_times'])
        
        return {
            **self.prediction_stats,
            'avg_processing_time_ms': avg_processing_time,
            'cache_size': {
                'addresses': len(self.cache.local_cache['address_stats']),
                'tokens': len(self.cache.local_cache['token_stats']),
                'recent_trades': sum(len(trades) for trades in self.cache.local_cache['recent_trades'].values())
            }
        }