from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Optional

Base = declarative_base()

class PredictionResult(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_hash = Column(String, index=True)
    true_label = Column(Integer)  # 0 or 1
    predicted_label = Column(Integer)  # 0 or 1
    prediction_probability = Column(Float)  # 0.0 to 1.0
    processing_time_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Enhanced fields for ensemble tracking
    prediction_method = Column(String, default="ensemble")  # "basic_only", "advanced", "ensemble"
    confidence_level = Column(String, default="Medium")  # "High", "Medium", "Low"
    individual_predictions = Column(Text)  # JSON string of individual model predictions
    
    # Trade features for analysis
    trade_amount_eth = Column(Float)
    trade_amount_dollar = Column(Float)
    eth_buyer_id = Column(Integer)
    eth_seller_id = Column(Integer)

class PredictionResponse(BaseModel):
    transaction_hash: str
    true_label: int
    predicted_label: int
    prediction_probability: float
    processing_time_ms: float
    agreement: bool
    confidence_level: str  # "High", "Medium", "Low"
    prediction_method: Optional[str] = "ensemble"  # New field
    individual_predictions: Optional[Dict] = {}  # New field
    
class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    wash_predictions: int
    normal_predictions: int
    agreement_rate: float
    
    # Enhanced metrics
    auc_roc: Optional[float] = 0.0
    avg_precision: Optional[float] = 0.0
    avg_processing_time_ms: Optional[float] = 0.0
    prediction_method_breakdown: Optional[Dict] = {}
    
class ComparisonBreakdown(BaseModel):
    both_wash: int  # True=1, Pred=1
    both_normal: int  # True=0, Pred=0
    only_true_wash: int  # True=1, Pred=0
    only_ml_wash: int  # True=0, Pred=1

class EnsembleStats(BaseModel):
    """New model for ensemble-specific statistics"""
    total_predictions: int
    basic_only_predictions: int
    advanced_predictions: int
    ensemble_predictions: int
    cache_size: Dict
    avg_processing_time_ms: float