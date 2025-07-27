from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis
import asyncio
import json
import logging
from ml_service import MLPredictionService
from models import Base, PredictionResult, PredictionResponse, PerformanceMetrics, ComparisonBreakdown, EnsembleStats
from typing import List, Dict
import uvicorn
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced ML Prediction Service with Ensemble Learning")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Enhanced ML Service with Ensemble
try:
    ml_service = MLPredictionService(
        model_path="../models/enhanced_ensemble_wash_trade_detector.pkl"  # Updated path
    )
    logger.info("Enhanced ensemble ML service initialized")
except Exception as e:
    logger.error(f"Failed to initialize enhanced ML service: {e}")
    # Fallback to original model
    try:
        ml_service = MLPredictionService(
            model_path="../models/Random_Forest_phase1.pkl",
            scaler_path="../models/scaler_phase1.pkl"
        )
        logger.info("Fallback: Original ML service initialized")
    except Exception as e2:
        logger.error(f"Failed to initialize any ML service: {e2}")
        ml_service = None

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Processing control
processing_active = False
confidence_threshold = 0.5

@app.get("/")
async def root():
    return {
        "message": "Enhanced ML Prediction Service with Ensemble Learning", 
        "status": "running",
        "ensemble_enabled": ml_service.ensemble_package is not None if ml_service else False
    }

@app.post("/start-processing")
async def start_processing():
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())
    return {"message": "Enhanced processing started"}

@app.post("/stop-processing")
async def stop_processing():
    global processing_active
    processing_active = False
    return {"message": "Processing stopped"}

@app.post("/set-threshold/{threshold}")
async def set_threshold(threshold: float):
    global confidence_threshold
    confidence_threshold = max(0.0, min(1.0, threshold))
    return {"message": f"Threshold set to {confidence_threshold}"}

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_trades():
    """Enhanced trade processing with ensemble predictions"""
    global processing_active
    logger.info("Starting enhanced trade processing with ensemble learning...")
    
    while processing_active:
        try:
            if not redis_client or not ml_service:
                await asyncio.sleep(1.0)
                continue
            
            # Get trade from queue
            trade_json = redis_client.rpop("trade_queue")
            if not trade_json:
                await asyncio.sleep(0.1)
                continue
            
            trade_data = json.loads(trade_json)
            
            # Make enhanced prediction
            predicted_label, probability, processing_time = ml_service.predict(trade_data)
            
            # Apply threshold
            if probability >= confidence_threshold:
                final_prediction = 1
            else:
                final_prediction = 0
            
            # Get additional prediction info
            confidence_level = ml_service.get_confidence_level(probability)
            prediction_stats = ml_service.get_prediction_stats()
            
            # Store enhanced result in database
            db = SessionLocal()
            try:
                result = PredictionResult(
                    transaction_hash=trade_data.get('transactionHash', ''),
                    true_label=trade_data.get('wash_label', 0),
                    predicted_label=final_prediction,
                    prediction_probability=probability,
                    processing_time_ms=processing_time,
                    prediction_method=prediction_stats.get('last_method', 'ensemble'),
                    confidence_level=confidence_level,
                    individual_predictions=json.dumps({}),  # Could store individual model predictions
                    trade_amount_eth=trade_data.get('trade_amount_eth', 0.0),
                    trade_amount_dollar=trade_data.get('trade_amount_dollar', 0.0),
                    eth_buyer_id=trade_data.get('eth_buyer_id', 0),
                    eth_seller_id=trade_data.get('eth_seller_id', 0)
                )
                db.add(result)
                db.commit()
            finally:
                db.close()
            
            # Create enhanced response
            response = PredictionResponse(
                transaction_hash=trade_data.get('transactionHash', ''),
                true_label=trade_data.get('wash_label', 0),
                predicted_label=final_prediction,
                prediction_probability=probability,
                processing_time_ms=processing_time,
                agreement=trade_data.get('wash_label', 0) == final_prediction,
                confidence_level=confidence_level,
                prediction_method=prediction_stats.get('last_method', 'ensemble'),
                individual_predictions={}
            )
            
            # Fixed: Use model_dump_json() instead of deprecated json()
            await manager.broadcast(response.model_dump_json())
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            await asyncio.sleep(1.0)

@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics(db: Session = Depends(get_db)):
    """Get enhanced performance metrics"""
    try:
        results = db.query(PredictionResult).all()
        
        if not results:
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                total_predictions=0, wash_predictions=0, normal_predictions=0,
                agreement_rate=0.0, auc_roc=0.0, avg_precision=0.0,
                avg_processing_time_ms=0.0, prediction_method_breakdown={}
            )
        
        true_labels = [r.true_label for r in results]
        predicted_labels = [r.predicted_label for r in results]
        
        metrics = ml_service.calculate_metrics(true_labels, predicted_labels)
        
        total_predictions = len(results)
        wash_predictions = sum(predicted_labels)
        normal_predictions = total_predictions - wash_predictions
        agreements = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
        agreement_rate = agreements / total_predictions if total_predictions > 0 else 0.0
        
        # Enhanced metrics
        avg_processing_time = sum(r.processing_time_ms for r in results) / total_predictions
        
        # Prediction method breakdown
        method_breakdown = defaultdict(int)
        for r in results:
            method_breakdown[r.prediction_method or 'unknown'] += 1
        
        return PerformanceMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            total_predictions=total_predictions,
            wash_predictions=wash_predictions,
            normal_predictions=normal_predictions,
            agreement_rate=agreement_rate,
            auc_roc=metrics.get('auc_roc', 0.0),
            avg_precision=metrics.get('avg_precision', 0.0),
            avg_processing_time_ms=avg_processing_time,
            prediction_method_breakdown=dict(method_breakdown)
        )
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return PerformanceMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            total_predictions=0, wash_predictions=0, normal_predictions=0,
            agreement_rate=0.0, auc_roc=0.0, avg_precision=0.0,
            avg_processing_time_ms=0.0, prediction_method_breakdown={}
        )

@app.get("/comparison", response_model=ComparisonBreakdown)
async def get_comparison(db: Session = Depends(get_db)):
    """Get comparison breakdown"""
    try:
        results = db.query(PredictionResult).all()
        
        both_wash = sum(1 for r in results if r.true_label == 1 and r.predicted_label == 1)
        both_normal = sum(1 for r in results if r.true_label == 0 and r.predicted_label == 0)
        only_true_wash = sum(1 for r in results if r.true_label == 1 and r.predicted_label == 0)
        only_ml_wash = sum(1 for r in results if r.true_label == 0 and r.predicted_label == 1)
        
        return ComparisonBreakdown(
            both_wash=both_wash,
            both_normal=both_normal,
            only_true_wash=only_true_wash,
            only_ml_wash=only_ml_wash
        )
        
    except Exception as e:
        logger.error(f"Error getting comparison: {e}")
        return ComparisonBreakdown(
            both_wash=0, both_normal=0, only_true_wash=0, only_ml_wash=0
        )

@app.get("/ensemble-stats", response_model=EnsembleStats)
async def get_ensemble_stats():
    """Get ensemble-specific statistics"""
    try:
        if not ml_service:
            return EnsembleStats(
                total_predictions=0, basic_only_predictions=0,
                advanced_predictions=0, ensemble_predictions=0,
                cache_size={}, avg_processing_time_ms=0.0
            )
        
        stats = ml_service.get_prediction_stats()
        
        return EnsembleStats(
            total_predictions=stats['total_predictions'],
            basic_only_predictions=stats['basic_only'],
            advanced_predictions=stats['advanced'],
            ensemble_predictions=stats['ensemble'],
            cache_size=stats['cache_size'],
            avg_processing_time_ms=stats['avg_processing_time_ms']
        )
        
    except Exception as e:
        logger.error(f"Error getting ensemble stats: {e}")
        return EnsembleStats(
            total_predictions=0, basic_only_predictions=0,
            advanced_predictions=0, ensemble_predictions=0,
            cache_size={}, avg_processing_time_ms=0.0
        )

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    ensemble_status = "enabled" if (ml_service and ml_service.ensemble_package) else "disabled"
    cache_status = "active" if (ml_service and ml_service.cache) else "inactive"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ensemble_status": ensemble_status,
        "cache_status": cache_status,
        "processing_active": processing_active
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced ML Prediction Service with Ensemble Learning started")
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Keeping your original port