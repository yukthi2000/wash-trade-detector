from pydantic import BaseModel
from typing import Optional
import json

class Trade(BaseModel):
    eth_seller: str
    eth_buyer: str
    date: str
    cut: float
    blockNumber: int
    timestamp: int
    transactionHash: str
    ether: float
    token: float
    trade_amount_eth: float
    trade_amount_dollar: float
    trade_amount_token: float
    token_price_in_eth: float
    eth_buyer_id: int
    eth_seller_id: int
    wash_label: int  # 0=Normal, 1=Wash Trade
    
    def to_json(self):
        return json.dumps(self.dict())
    
    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))