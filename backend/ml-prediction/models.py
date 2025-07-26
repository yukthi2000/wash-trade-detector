from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class PredictionResult(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_hash = Column(String, index=True)
    true_label = Column(Integer)  # 0 or 1
    predicted_label = Column(Integer)  # 0 or 1
    prediction_probability = Column(Float)  # 0.0 to 1.0
    processing_time_ms = Column(Float)
    prediction_method = Column(String)  # "basic_ensemble", "full_ensemble", "error"
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Trade features for analysis
    trade_amount_eth = Column(Float)
    trade_amount_dollar = Column(Float)
    eth_buyer_id = Column(Integer)
    eth_seller_id = Column(Integer)
    eth_seller = Column(String)  # Added for address tracking
    eth_buyer = Column(String)   # Added for address tracking

class PredictionResponse(BaseModel):
    transaction_hash: str
    true_label: int
    predicted_label: int
    prediction_probability: float
    processing_time_ms: float
    prediction_method: str  # Added to track which ensemble method was used
    agreement: bool
    confidence_level: str  # "High", "Medium", "Low"
    
class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    wash_predictions: int
    normal_predictions: int
    agreement_rate: float
    basic_ensemble_count: int  # Added
    full_ensemble_count: int   # Added
    
class ComparisonBreakdown(BaseModel):
    both_wash: int  # True=1, Pred=1
    both_normal: int  # True=0, Pred=0
    only_true_wash: int  # True=1, Pred=0
    only_ml_wash: int  # True=0, Pred=1