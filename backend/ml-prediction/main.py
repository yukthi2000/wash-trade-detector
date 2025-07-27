from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis
import asyncio
import json
import logging
from ml_service import MLPredictionService
from models import Base, PredictionResult, PredictionResponse, PerformanceMetrics, ComparisonBreakdown
from typing import List, Dict
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XGBoost ML Prediction Service")

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

# ML Service with XGBoost model
def initialize_ml_service():
    # Updated paths for XGBoost model
    model_paths = [
        "XGBoost_phase1.pkl",

  
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                service = MLPredictionService(model_path)
                logger.info(f"XGBoost ML service initialized with model: {model_path}")
                model_info = service.get_model_info()
                logger.info(f"Model info: {model_info}")
                return service
            except Exception as e:
                logger.error(f"Failed to load XGBoost model {model_path}: {e}")
                continue
    
    logger.error("Could not initialize XGBoost ML service - no valid model files found")
    logger.info("Expected model file: xgb_final.pkl")
    return None

ml_service = initialize_ml_service()

# WebSocket connections
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
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Processing control
processing_active = False
confidence_threshold = 0.5

@app.get("/")
async def root():
    model_info = ml_service.get_model_info() if ml_service else {"model_type": "Not loaded"}
    return {
        "message": "XGBoost ML Prediction Service", 
        "status": "running",
        "model": model_info
    }

@app.get("/health")
async def health():
    model_info = ml_service.get_model_info() if ml_service else None
    return {
        "status": "healthy",
        "redis_connected": redis_client is not None,
        "ml_service_loaded": ml_service is not None,
        "model_info": model_info,
        "active_connections": len(manager.active_connections),
        "processing": processing_active,
        "confidence_threshold": confidence_threshold
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if ml_service:
        return ml_service.get_model_info()
    else:
        return {"error": "Model not loaded"}

@app.post("/start-processing")
async def start_processing():
    global processing_active
    if not processing_active:
        processing_active = True
        asyncio.create_task(process_trades())
    return {"message": "XGBoost processing started", "status": processing_active}

@app.post("/stop-processing")
async def stop_processing():
    global processing_active
    processing_active = False
    return {"message": "XGBoost processing stopped", "status": processing_active}

@app.post("/set-threshold/{threshold}")
async def set_threshold(threshold: float):
    global confidence_threshold
    confidence_threshold = max(0.0, min(1.0, threshold))
    return {"message": f"Threshold set to {confidence_threshold}", "threshold": confidence_threshold}

@app.post("/test-prediction")
async def test_prediction(trade_data: dict):
    """Test endpoint for making a single prediction"""
    if not ml_service:
        return {"error": "ML service not available"}
    
    try:
        predicted_label, probability, processing_time = ml_service.predict(trade_data)
        return {
            "predicted_label": predicted_label,
            "probability": probability,
            "processing_time_ms": processing_time,
            "confidence_level": ml_service.get_confidence_level(probability),
            "model_type": "XGBoost"
        }
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Prediction WebSocket error: {e}")
        manager.disconnect(websocket)

async def process_trades():
    """Process trades from Redis queue and make XGBoost predictions"""
    global processing_active
    logger.info("Starting XGBoost trade processing...")
    
    while processing_active:
        try:
            if not redis_client:
                logger.warning("Redis not available, retrying...")
                await asyncio.sleep(5.0)
                continue
                
            if not ml_service:
                logger.warning("XGBoost ML service not available, retrying...")
                await asyncio.sleep(5.0)
                continue
            
            # Get trade from queue
            trade_json = redis_client.rpop("trade_queue")
            if not trade_json:
                await asyncio.sleep(0.1)
                continue
            
            trade_data = json.loads(trade_json)
            logger.info(f"Processing trade with XGBoost: {trade_data.get('transactionHash', 'unknown')[:10]}...")
            
            # Make XGBoost prediction
            predicted_label, probability, processing_time = ml_service.predict(trade_data)
            
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
                    trade_amount_eth=trade_data.get('trade_amount_eth', 0.0),
                    trade_amount_dollar=trade_data.get('trade_amount_dollar', 0.0),
                    eth_buyer_id=trade_data.get('eth_buyer_id', 0),
                    eth_seller_id=trade_data.get('eth_seller_id', 0)
                )
                db.add(result)
                db.commit()
                logger.info(f"Stored XGBoost prediction for {trade_data.get('transactionHash', 'unknown')[:10]}")
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
                agreement=trade_data.get('wash_label', 0) == final_prediction,
                confidence_level=ml_service.get_confidence_level(probability)
            )
            
            # Broadcast to WebSocket clients
            if manager.active_connections:
                await manager.broadcast(response.model_dump_json())
                logger.info(f"Broadcasted XGBoost prediction: {final_prediction} (prob: {probability:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing trade with XGBoost: {e}")
            await asyncio.sleep(1.0)

# Keep all your existing endpoints (metrics, comparison, etc.)
@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics(db: Session = Depends(get_db)):
    """Get current performance metrics"""
    try:
        results = db.query(PredictionResult).all()
        
        if not results:
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                total_predictions=0, wash_predictions=0, normal_predictions=0,
                agreement_rate=0.0
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
        
        return PerformanceMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            total_predictions=total_predictions,
            wash_predictions=wash_predictions,
            normal_predictions=normal_predictions,
            agreement_rate=agreement_rate
        )
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return PerformanceMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            total_predictions=0, wash_predictions=0, normal_predictions=0,
            agreement_rate=0.0
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

@app.on_event("startup")
async def startup_event():
    logger.info("XGBoost ML Prediction Service started")
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)