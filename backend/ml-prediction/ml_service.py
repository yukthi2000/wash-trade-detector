import numpy as np
import pandas as pd
import joblib
import time
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("ml_service")
logger.setLevel(logging.INFO)

class EnsembleMLPredictionService:
    def __init__(self, model_path):
        # Load ensemble model package
        package = joblib.load(model_path)
        self.neural_network_basic = package['neural_network_basic']
        self.xgboost_basic = package['xgboost_basic']
        self.random_forest_advanced = package['random_forest_advanced']
        self.xgboost_advanced = package['xgboost_advanced']
        self.scaler_basic = package['scaler_basic']
        self.meta_learner = package['meta_learner']
        self.basic_features = package['basic_features']
        self.advanced_features = package['advanced_features']

        # Global medians for missing advanced features
        self.global_adv_medians = package.get('global_adv_medians', None)
        if self.global_adv_medians is None:
            logger.warning("No global medians found in model package. Using 0 for all advanced features.")
            self.global_adv_medians = {feat: 0 for feat in self.advanced_features}

    def _prepare_basic_features(self, trade_data: Dict[str, Any]) -> np.ndarray:
        """Prepare and scale basic features as numpy array."""
        data = {feat: trade_data.get(feat, 0) for feat in self.basic_features}
        df = pd.DataFrame([data])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        return self.scaler_basic.transform(df)




    def _prepare_advanced_features(self, trade_data: Dict[str, Any]) -> np.ndarray:
        """Prepare advanced features as numpy array, filling missing with medians."""
        data = {}
        for feat in self.advanced_features:
            value = trade_data.get(feat, None)
            if value is None or value == '' or value == np.inf or value == -np.inf or pd.isnull(value):
                value = self.global_adv_medians.get(feat, 0)
            data[feat] = value
        df = pd.DataFrame([data])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        return df.values

    def predict(self, trade_data: Dict[str, Any]) -> Tuple[int, float, float, str]:
        """Predict using all four models and the stacking meta-learner (always 'full_ensemble')."""
        start_time = time.time()
        try:
            # Prepare features
            basic_features = self._prepare_basic_features(trade_data)
            advanced_features = self._prepare_advanced_features(trade_data)

            # Four base model probabilities
            nn_proba = self.neural_network_basic.predict_proba(basic_features)[0, 1]
            xgb_basic_proba = self.xgboost_basic.predict_proba(basic_features)[0, 1]
            rf_proba = self.random_forest_advanced.predict_proba(advanced_features)[0, 1]
            xgb_advanced_proba = self.xgboost_advanced.predict_proba(advanced_features)[0, 1]

            # Meta features for stacking
            meta_features = np.array([[nn_proba, xgb_basic_proba, rf_proba, xgb_advanced_proba]])
            final_probability = self.meta_learner.predict_proba(meta_features)[0, 1]
            final_prediction = int(final_probability > 0.5)
            processing_time = (time.time() - start_time) * 1000

            return final_prediction, float(final_probability), processing_time, "full_ensemble"

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            processing_time = (time.time() - start_time) * 1000
            return 0, 0.0, processing_time, "error"

    def calculate_metrics(self, y_true, y_pred):
        """Returns a metrics dict compatible with /metrics endpoint."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics
