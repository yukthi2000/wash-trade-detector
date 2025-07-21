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

app = FastAPI(title="ML Prediction Service")

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

# ML Service
try:
    ml_service = MLPredictionService(
        model_path="../models/Random_Forest_phase1.pkl",
        scaler_path="../models/scaler_phase1.pkl"
    )
except Exception as e:
    logger.error(f"Failed to initialize ML service: {e}")
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
    return {"message": "ML Prediction Service", "status": "running"}

@app.post("/start-processing")
async def start_processing():
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())
    return {"message": "Processing started"}

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
    """Process trades from Redis queue and make predictions"""
    global processing_active
    logger.info("Starting trade processing...")
    
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
            
            # Make prediction
            predicted_label, probability, processing_time = ml_service.predict(trade_data)
            
            # Apply threshold
            if probability >= confidence_threshold:
                final_prediction = 1
            else:
                final_prediction = 0
            
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
            await manager.broadcast(response.json())
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
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
        
        metrics = ml_service.calculate_metrics(true_labels, predicted_labels)
        
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
    logger.info("ML Prediction Service started")
    global processing_active
    processing_active = True
    asyncio.create_task(process_trades())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)