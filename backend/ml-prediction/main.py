from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis
import asyncio
import json
import logging
from ml_service import EnsembleMLPredictionService  # Updated import
from models import Base, PredictionResult, PredictionResponse, PerformanceMetrics, ComparisonBreakdown
from typing import List, Dict
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ensemble ML Prediction Service")

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

# Redis connection with error handling
def get_redis_client():
    try:
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        logger.info("Connected to Redis")
        return client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

redis_client = get_redis_client()

# Enhanced ML Service initialization
def initialize_ml_service():
    ensemble_paths = [
     
        "enhanced_ensemble_wash_trade_detector.pkl",
    ]
    
    for ensemble_path in ensemble_paths:
        if os.path.exists(ensemble_path):
            try:
                service = EnsembleMLPredictionService(ensemble_path)
                logger.info(f"Ensemble ML service initialized with: {ensemble_path}")
                return service
            except Exception as e:
                logger.error(f"Failed to load ensemble model {ensemble_path}: {e}")
                continue
    
    logger.error("Could not initialize Ensemble ML service - no valid model files found")
    return None

ml_service = initialize_ml_service()

# WebSocket connections (same as before)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Prediction client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Prediction client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending prediction message: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Processing control
processing_active = False
confidence_threshold = 0.5

@app.get("/")
async def root():
    return {"message": "Ensemble ML Prediction Service", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "redis_connected": redis_client is not None,
        "ml_service_loaded": ml_service is not None,
        "active_connections": len(manager.active_connections),
        "processing": processing_active,
        "confidence_threshold": confidence_threshold,
        "service_type": "ensemble"  # Added
    }



def get_confidence_level(prob: float) -> str:
    if prob >= 0.85:
        return "Very High"
    elif prob >= 0.7:
        return "High"
    elif prob >= 0.5:
        return "Medium"
    elif prob >= 0.3:
        return "Low"
    else:
        return "Very Low"

# Updated process_trades function
async def process_trades():
    """Process trades from Redis queue and make ensemble predictions"""
    global processing_active
    logger.info("Starting ensemble trade processing...")
    
    while processing_active:
        try:
            if not redis_client:
                logger.warning("Redis not available, retrying...")
                await asyncio.sleep(5.0)
                continue
                
            if not ml_service:
                logger.warning("Ensemble ML service not available, retrying...")
                await asyncio.sleep(5.0)
                continue
            
            # Get trade from queue
            trade_json = redis_client.rpop("trade_queue")
            if not trade_json:
                await asyncio.sleep(0.1)
                continue
            
            trade_data = json.loads(trade_json)
            logger.info(f"Processing trade: {trade_data.get('transactionHash', 'unknown')[:10]}...")
            
            # Make ensemble prediction
            predicted_label, probability, processing_time, prediction_method = ml_service.predict(trade_data)
            
            # Apply threshold
            final_prediction = 1 if probability >= confidence_threshold else 0
            
            # Store result in database
            db = SessionLocal()
            try:
                result = PredictionResult(
                    transaction_hash=trade_data.get('transactionHash', ''),
                    true_label=trade_data.get('wash_label', 0),
                    predicted_label=final_prediction,
                    prediction_probability=probability,
                    processing_time_ms=processing_time,
                    prediction_method=prediction_method,  # Added
                    trade_amount_eth=trade_data.get('trade_amount_eth', 0.0),
                    trade_amount_dollar=trade_data.get('trade_amount_dollar', 0.0),
                    eth_buyer_id=trade_data.get('eth_buyer_id', 0),
                    eth_seller_id=trade_data.get('eth_seller_id', 0),
                    eth_seller=trade_data.get('eth_seller', ''),  # Added
                    eth_buyer=trade_data.get('eth_buyer', '')    # Added
                )
                db.add(result)
                db.commit()
                logger.info(f"Stored prediction for {trade_data.get('transactionHash', 'unknown')[:10]} using {prediction_method}")
            except Exception as e:
                logger.error(f"Database error: {e}")
                db.rollback()
            finally:
                db.close()
            
            # Create response
            response = PredictionResponse(
                transaction_hash=trade_data.get('transactionHash', ''),
                true_label=trade_data.get('wash_label', 0),
                predicted_label=final_prediction,
                prediction_probability=probability,
                processing_time_ms=processing_time,
                prediction_method=prediction_method,  # Added
                agreement=trade_data.get('wash_label', 0) == final_prediction,
                # confidence_level=ml_service.get_confidence_level(probability)
                confidence_level=get_confidence_level(probability)  
            )
            
            # Broadcast to WebSocket clients
            if manager.active_connections:
                await manager.broadcast(response.json())
                logger.info(f"Broadcasted prediction: {final_prediction} (prob: {probability:.3f}, method: {prediction_method})")
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            await asyncio.sleep(1.0)

# Updated metrics endpoint
@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics(db: Session = Depends(get_db)):
    """Get current performance metrics with ensemble breakdown"""
    try:
        results = db.query(PredictionResult).all()
        
        if not results:
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                total_predictions=0, wash_predictions=0, normal_predictions=0,
                agreement_rate=0.0, basic_ensemble_count=0, full_ensemble_count=0
            )
        
        true_labels = [r.true_label for r in results]
        predicted_labels = [r.predicted_label for r in results]
        
        if ml_service:
            metrics = ml_service.calculate_metrics(true_labels, predicted_labels)
        else:
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        total_predictions = len(results)
        wash_predictions = sum(predicted_labels)
        normal_predictions = total_predictions - wash_predictions
        agreements = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
        agreement_rate = agreements / total_predictions if total_predictions > 0 else 0.0
        
        # Count ensemble method usage
        basic_ensemble_count = sum(1 for r in results if r.prediction_method == "basic_ensemble")
        full_ensemble_count = sum(1 for r in results if r.prediction_method == "full_ensemble")
        
        return PerformanceMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            total_predictions=total_predictions,
            wash_predictions=wash_predictions,
            normal_predictions=normal_predictions,
            agreement_rate=agreement_rate,
            basic_ensemble_count=basic_ensemble_count,
            full_ensemble_count=full_ensemble_count
        )
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return PerformanceMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            total_predictions=0, wash_predictions=0, normal_predictions=0,
            agreement_rate=0.0, basic_ensemble_count=0, full_ensemble_count=0
        )

# Add new endpoint for ensemble statistics
@app.get("/ensemble-stats")
async def get_ensemble_stats(db: Session = Depends(get_db)):
    """Get ensemble method usage statistics"""
    try:
        results = db.query(PredictionResult).all()
        
        stats = {
            "total_predictions": len(results),
            "basic_ensemble_count": sum(1 for r in results if r.prediction_method == "basic_ensemble"),
            "full_ensemble_count": sum(1 for r in results if r.prediction_method == "full_ensemble"),
            "error_count": sum(1 for r in results if r.prediction_method == "error"),
            "avg_processing_time_basic": 0.0,
            "avg_processing_time_full": 0.0
        }
        
        basic_times = [r.processing_time_ms for r in results if r.prediction_method == "basic_ensemble"]
        full_times = [r.processing_time_ms for r in results if r.prediction_method == "full_ensemble"]
        
        if basic_times:
            stats["avg_processing_time_basic"] = sum(basic_times) / len(basic_times)
        if full_times:
            stats["avg_processing_time_full"] = sum(full_times) / len(full_times)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting ensemble stats: {e}")
        return {
            "total_predictions": 0,
            "basic_ensemble_count": 0,
            "full_ensemble_count": 0,
            "error_count": 0,
            "avg_processing_time_basic": 0.0,
            "avg_processing_time_full": 0.0
        }

# Keep all other endpoints the same (start-processing, stop-processing, etc.)
@app.post("/start-processing")
async def start_processing():
    global processing_active
    if not processing_active:
        processing_active = True
        asyncio.create_task(process_trades())
    return {"message": "Ensemble processing started", "status": processing_active}

@app.post("/stop-processing")
async def stop_processing():
    global processing_active
    processing_active = False
    return {"message": "Ensemble processing stopped", "status": processing_active}

@app.post("/set-threshold/{threshold}")
async def set_threshold(threshold: float):
    global confidence_threshold
    confidence_threshold = max(0.0, min(1.0, threshold))
    return {"message": f"Threshold set to {confidence_threshold}", "threshold": confidence_threshold}

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Prediction WebSocket error: {e}")
        manager.disconnect(websocket)

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

@app.on_event("startup")
async def startup_event():
    logger.info("Ensemble ML Prediction Service started")
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)