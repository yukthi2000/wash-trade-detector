import random
import time
from datetime import datetime, timedelta
from models import Trade
import pandas as pd
import numpy as np

class TradeGenerator:
    def __init__(self, csv_file_path: str = None):
        self.csv_file_path = csv_file_path
        self.csv_data = None
        self.csv_index = 0
        
        if csv_file_path:
            try:
                self.csv_data = pd.read_csv(csv_file_path)
                print(f"Loaded {len(self.csv_data)} trades from CSV")
            except Exception as e:
                print(f"Error loading CSV: {e}")
                self.csv_data = None
    
    def generate_realistic_trade(self) -> Trade:
        """Generate a realistic cryptocurrency trade"""
        if self.csv_data is not None and self.csv_index < len(self.csv_data):
            # Use CSV data
            row = self.csv_data.iloc[self.csv_index]
            self.csv_index += 1
            
            trade = Trade(
                eth_seller=str(row.get('eth_seller', self._generate_address())),
                eth_buyer=str(row.get('eth_buyer', self._generate_address())),
                date=str(row.get('date', datetime.now().strftime('%Y-%m-%d'))),
                cut=float(row.get('cut', random.uniform(1000000, 2000000))),
                blockNumber=int(row.get('blockNumber', random.randint(9000000, 10000000))),
                timestamp=int(row.get('timestamp', int(time.time()))),
                transactionHash=str(row.get('transactionHash', self._generate_tx_hash())),
                ether=float(row.get('ether', 0.0)),
                token=float(row.get('token', random.uniform(1, 1000))),
                trade_amount_eth=float(row.get('trade_amount_eth', random.uniform(0.1, 10.0))),
                trade_amount_dollar=float(row.get('trade_amount_dollar', random.uniform(100, 10000))),
                trade_amount_token=float(row.get('trade_amount_token', random.uniform(1, 1000))),
                token_price_in_eth=float(row.get('token_price_in_eth', random.uniform(0.001, 0.1))),
                eth_buyer_id=int(row.get('eth_buyer_id', random.randint(1, 100000))),
                eth_seller_id=int(row.get('eth_seller_id', random.randint(1, 100000))),
                wash_label=int(row.get('wash_label', random.choices([0, 1], weights=[0.85, 0.15])[0]))
            )
            return trade
        else:
            # Generate synthetic data
            return self._generate_synthetic_trade()
    
    def _generate_synthetic_trade(self) -> Trade:
        """Generate synthetic trade data"""
        # Create wash trade patterns (15% of trades)
        is_wash = random.random() < 0.15
        
        if is_wash:
            # Wash trade characteristics
            buyer_id = random.randint(1, 1000)
            seller_id = buyer_id + random.randint(1, 5)  # Related accounts
            trade_amount = random.uniform(0.5, 2.0)  # Smaller amounts
        else:
            # Normal trade
            buyer_id = random.randint(1, 100000)
            seller_id = random.randint(1, 100000)
            trade_amount = random.uniform(0.1, 10.0)
        
        trade_amount_dollar = trade_amount * random.uniform(1500, 2500)  # ETH price
        token_amount = random.uniform(10, 1000)
        token_price = trade_amount / token_amount
        
        return Trade(
            eth_seller=self._generate_address(),
            eth_buyer=self._generate_address(),
            date=datetime.now().strftime('%Y-%m-%d'),
            cut=random.uniform(1000000, 2000000),
            blockNumber=random.randint(9000000, 10000000),
            timestamp=int(time.time()),
            transactionHash=self._generate_tx_hash(),
            ether=0.0,
            token=random.uniform(1, 1000),
            trade_amount_eth=trade_amount,
            trade_amount_dollar=trade_amount_dollar,
            trade_amount_token=token_amount,
            token_price_in_eth=token_price,
            eth_buyer_id=buyer_id,
            eth_seller_id=seller_id,
            wash_label=1 if is_wash else 0
        )
    
    def _generate_address(self) -> str:
        """Generate a realistic Ethereum address"""
        return "0x" + "".join(random.choices("0123456789abcdef", k=40))
    
    def _generate_tx_hash(self) -> str:
        """Generate a realistic transaction hash"""
        return "0x" + "".join(random.choices("0123456789abcdef", k=64))