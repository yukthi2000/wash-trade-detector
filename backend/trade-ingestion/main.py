from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import redis
import asyncio
import json
import logging
from trade_generator import TradeGenerator
from models import Trade
from typing import List
import uvicorn
# In your trade ingestion main.py
import uuid
import random
from datetime import datetime

# Add a set to track generated transaction hashes
generated_hashes = set()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trade Ingestion Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Trade generator
trade_generator = TradeGenerator('test_data_5pct_stratified.csv')  # Add CSV path here if available

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

# Streaming control
streaming_active = False

@app.get("/")
async def root():
    return {"message": "Trade Ingestion Service", "status": "running"}

@app.post("/start-streaming")
async def start_streaming():
    global streaming_active
    streaming_active = True
    asyncio.create_task(stream_trades())
    return {"message": "Streaming started"}

@app.post("/stop-streaming")
async def stop_streaming():
    global streaming_active
    streaming_active = False
    return {"message": "Streaming stopped"}

@app.websocket("/ws/trades")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# async def stream_trades():
#     """Stream trades to WebSocket clients and Redis queue"""
#     global streaming_active
#     logger.info("Starting trade streaming...")
    
#     while streaming_active:
#         try:
#             # Generate trade
#             trade = trade_generator.generate_realistic_trade()
#             trade_json = trade.to_json()
            
#             # Send to Redis queue
#             if redis_client:
#                 redis_client.lpush("trade_queue", trade_json)
            
#             # Broadcast to WebSocket clients
#             await manager.broadcast(trade_json)
            
#             # Wait before next trade (adjust for desired frequency)
#             await asyncio.sleep(1.0)  # 1 trade per second
            
#         except Exception as e:
#             logger.error(f"Error in trade streaming: {e}")
#             await asyncio.sleep(1.0)





async def stream_trades():
    """Stream trades to WebSocket clients and Redis queue"""
    global streaming_active
    logger.info("Starting trade streaming...")
    
    while streaming_active:
        try:
            # Generate trade
            trade = trade_generator.generate_realistic_trade()
            
            # Ensure unique transaction hash
            max_attempts = 5
            attempts = 0
            while trade.transactionHash in generated_hashes and attempts < max_attempts:
                logger.warning(f"Duplicate hash detected: {trade.transactionHash}, regenerating...")
                trade = trade_generator.generate_realistic_trade()
                attempts += 1
            
            if attempts >= max_attempts:
                # Generate a completely new hash
                trade.transactionHash = "0x" + "".join(random.choices("0123456789abcdef", k=64))
                logger.info(f"Generated new unique hash: {trade.transactionHash}")
            
            # Add to seen hashes
            generated_hashes.add(trade.transactionHash)
            
            # Keep only recent hashes to prevent memory issues
            if len(generated_hashes) > 10000:
                # Remove oldest half
                hashes_list = list(generated_hashes)
                generated_hashes.clear()
                generated_hashes.update(hashes_list[-5000:])
            
            trade_json = trade.to_json()
            
            logger.info(f"Streaming trade: {trade.transactionHash}")
            
            # Send to Redis queue
            if redis_client:
                redis_client.lpush("trade_queue", trade_json)
            
            # Broadcast to WebSocket clients
            await manager.broadcast(trade_json)
            
            # Wait before next trade (adjust for desired frequency)
            await asyncio.sleep(1.0)  # 1 trade per second
            
        except Exception as e:
            logger.error(f"Error in trade streaming: {e}")
            await asyncio.sleep(1.0)
# Remove or comment out the auto-start section:
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Trade Ingestion Service started")
#     # Auto-start streaming
#     global streaming_active
#     streaming_active = True
#     asyncio.create_task(stream_trades())

# Replace with:
@app.on_event("startup")
async def startup_event():
    logger.info("Trade Ingestion Service started - waiting for manual start")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)