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
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Random Forest ML Prediction Service")

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

# Random Forest ML Service
try:
    # Try different possible paths for your Random Forest model
    model_paths = [
        "Random_Forest_phase1.pkl",
        "../Random_Forest_phase1.pkl", 
        "./models/Random_Forest_phase1.pkl",
        "../models/Random_Forest_phase1.pkl"
    ]
    
    ml_service = None
    for path in model_paths:
        try:
            ml_service = MLPredictionService(model_path=path)
            logger.info(f"Random Forest model loaded from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if ml_service is None:
        logger.error("Could not find Random Forest model file in any expected location")
        
except Exception as e:
    logger.error(f"Failed to initialize Random Forest ML service: {e}")
    ml_service = None

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Processing control
processing_active = False
confidence_threshold = 0.5

@app.get("/")
async def root():
    return {
        "message": "Random Forest ML Prediction Service", 
        "status": "running",
        "model": "Random Forest" if ml_service else "Not loaded",
        "processing": processing_active,
        "expected_performance": {
            "accuracy": "98.02%",
            "f1_score": "97.54%",
            "auc_roc": "99.74%"
        }
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if ml_service and ml_service.model:
        return {
            "model_type": "Random Forest",
            "features": ml_service.feature_columns,
            "feature_count": len(ml_service.feature_columns),
            "scaling_required": False,
            "training_performance": {
                "accuracy": 0.9802,
                "precision": 0.9719,
                "recall": 0.9789,
                "f1_score": 0.9754,
                "auc_roc": 0.9974
            }
        }
    return {"error": "Model not loaded"}

@app.post("/start-processing")
async def start_processing():
    global processing_active
    if not ml_service:
        return {"error": "ML service not available"}
    
    if not processing_active:
        processing_active = True
        asyncio.create_task(process_trades())
        logger.info("Random Forest processing started")
        return {"message": "Random Forest processing started", "status": "success"}
    else:
        return {"message": "Processing already active", "status": "already_running"}

@app.post("/stop-processing")
async def stop_processing():
    global processing_active
    processing_active = False
    logger.info("Random Forest processing stopped")
    return {"message": "Random Forest processing stopped", "status": "success"}

@app.post("/set-threshold/{threshold}")
async def set_threshold(threshold: float):
    global confidence_threshold
    confidence_threshold = max(0.0, min(1.0, threshold))
    logger.info(f"Threshold set to {confidence_threshold}")
    return {"message": f"Confidence threshold set to {confidence_threshold}", "threshold": confidence_threshold}

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive - just wait for any message
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text('{"type": "ping"}')
            except Exception:
                break
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

async def process_trades():
    """Process trades from Redis queue and make Random Forest predictions"""
    global processing_active
    logger.info("Starting Random Forest trade processing...")
    
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
            transaction_hash = trade_data.get('transactionHash', '')
            logger.info(f"Processing trade: {transaction_hash}")
            
            # Make Random Forest prediction
            predicted_label, probability, processing_time = ml_service.predict(trade_data)
            
            # Apply threshold
            final_prediction = 1 if probability >= confidence_threshold else 0
            
            # Store result in database
            db = SessionLocal()
            try:
                result = PredictionResult(
                    transaction_hash=transaction_hash,
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
                logger.info(f"Stored prediction: {final_prediction} (prob: {probability:.3f}) for hash: {transaction_hash}")
            finally:
                db.close()
            
            # Create response
            response = PredictionResponse(
                transaction_hash=transaction_hash,
                true_label=trade_data.get('wash_label', 0),
                predicted_label=final_prediction,
                prediction_probability=probability,
                processing_time_ms=processing_time,
                agreement=trade_data.get('wash_label', 0) == final_prediction,
                confidence_level=ml_service.get_confidence_level(probability)
            )
            
            # Debug log the response
            logger.info(f"Broadcasting prediction for hash: {response.transaction_hash}")
            
            # Broadcast to WebSocket clients
            await manager.broadcast(response.model_dump_json())
            logger.info(f"Broadcasted prediction to {len(manager.active_connections)} clients")
            
        except Exception as e:
            logger.error(f"Error processing trade with Random Forest: {e}")
            await asyncio.sleep(1.0)

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
        
        metrics = ml_service.calculate_metrics(true_labels, predicted_labels) if ml_service else {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
        }
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)