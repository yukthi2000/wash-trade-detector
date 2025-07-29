import random
import time
import uuid
from datetime import datetime, timedelta
from models import Trade
import pandas as pd
import numpy as np

class TradeGenerator:
    def __init__(self, csv_file_path: str = None):
        self.csv_file_path = csv_file_path
        self.csv_data = None
        self.csv_index = 0
        self.used_hashes = set()  # Track used hashes
        
        if csv_file_path:
            try:
                self.csv_data = pd.read_csv(csv_file_path)
                print(f"Loaded {len(self.csv_data)} trades from CSV")
                # Clean the data upon loading
                self._clean_csv_data()
            except Exception as e:
                print(f"Error loading CSV: {e}")
                self.csv_data = None
    
    def _clean_csv_data(self):
        """Clean and preprocess the CSV data"""
        if self.csv_data is None:
            return
        
        print("Cleaning CSV data...")
        
        # Handle hexadecimal values in ether and token columns
        def convert_hex_to_float(value):
            if pd.isna(value):
                return 0.0
            if isinstance(value, str) and value.startswith('0x'):
                try:
                    # Convert hex to int, then to float
                    return float(int(value, 16))
                except ValueError:
                    return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        # Clean ether and token columns
        self.csv_data['ether'] = self.csv_data['ether'].apply(convert_hex_to_float)
        self.csv_data['token'] = self.csv_data['token'].apply(convert_hex_to_float)
        
        # Handle wash_label NaN values - assign random labels based on realistic distribution
        if 'wash_label' in self.csv_data.columns:
            # Fill NaN values with random labels (85% normal, 15% wash)
            nan_mask = self.csv_data['wash_label'].isna()
            random_labels = np.random.choice([0, 1], size=nan_mask.sum(), p=[0.85, 0.15])
            self.csv_data.loc[nan_mask, 'wash_label'] = random_labels
        else:
            # If wash_label column doesn't exist, create it
            self.csv_data['wash_label'] = np.random.choice([0, 1], size=len(self.csv_data), p=[0.85, 0.15])
        
        # Handle infinite values in token_price_in_eth
        if 'token_price_in_eth' in self.csv_data.columns:
            # Replace inf values with reasonable random prices
            inf_mask = np.isinf(self.csv_data['token_price_in_eth'])
            self.csv_data.loc[inf_mask, 'token_price_in_eth'] = np.random.uniform(0.001, 0.1, inf_mask.sum())
        
        # Ensure all numeric columns are properly typed
        numeric_columns = [
            'cut', 'blockNumber', 'timestamp', 'ether', 'token', 
            'trade_amount_eth', 'trade_amount_dollar', 'trade_amount_token', 
            'token_price_in_eth', 'eth_buyer_id', 'eth_seller_id', 'wash_label'
        ]
        
        for col in numeric_columns:
            if col in self.csv_data.columns:
                self.csv_data[col] = pd.to_numeric(self.csv_data[col], errors='coerce').fillna(0)
        
        print(f"Data cleaning complete. Shape: {self.csv_data.shape}")
        print(f"Wash label distribution: {self.csv_data['wash_label'].value_counts().to_dict()}")
    
    def _generate_unique_tx_hash(self) -> str:
        """Generate a unique transaction hash"""
        max_attempts = 10
        for _ in range(max_attempts):
            tx_hash = "0x" + "".join(random.choices("0123456789abcdef", k=64))
            if tx_hash not in self.used_hashes:
                self.used_hashes.add(tx_hash)
                # Keep memory usage reasonable
                if len(self.used_hashes) > 50000:
                    # Remove oldest half
                    hashes_list = list(self.used_hashes)
                    self.used_hashes.clear()
                    self.used_hashes.update(hashes_list[-25000:])
                return tx_hash
        
        # Fallback: use timestamp + random
        timestamp = str(int(time.time() * 1000000))  # microseconds
        random_part = "".join(random.choices("0123456789abcdef", k=40))
        return f"0x{timestamp[-24:]}{random_part}"
    
    def generate_realistic_trade(self) -> Trade:
        """Generate a realistic cryptocurrency trade"""
        if self.csv_data is not None and self.csv_index < len(self.csv_data):
            # Use CSV data but ensure unique hash
            row = self.csv_data.iloc[self.csv_index]
            self.csv_index += 1
            
            # If we've reached the end, reset to beginning
            if self.csv_index >= len(self.csv_data):
                self.csv_index = 0
                print("Reached end of CSV data, restarting from beginning")
            
            try:
                trade = Trade(
                    eth_seller=str(row.get('eth_seller', self._generate_address())),
                    eth_buyer=str(row.get('eth_buyer', self._generate_address())),
                    date=str(row.get('date', datetime.now().strftime('%Y-%m-%d'))),
                    cut=float(row.get('cut', random.uniform(1000000, 2000000))),
                    blockNumber=int(row.get('blockNumber', random.randint(9000000, 10000000))),
                    timestamp=int(row.get('timestamp', int(time.time()))),
                    transactionHash=self._generate_unique_tx_hash(),  # Always generate unique
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
            except Exception as e:
                print(f"Error creating trade from CSV row {self.csv_index-1}: {e}")
                print(f"Row data: {row.to_dict()}")
                # Fallback to synthetic trade
                return self._generate_synthetic_trade()
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
            transactionHash=self._generate_unique_tx_hash(),  # Use unique generator
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
    
    def get_stats(self):
        """Get statistics about the loaded data"""
        if self.csv_data is not None:
            return {
                "total_trades": len(self.csv_data),
                "current_index": self.csv_index,
                "wash_trades": int(self.csv_data['wash_label'].sum()),
                "normal_trades": int((self.csv_data['wash_label'] == 0).sum()),
                "columns": list(self.csv_data.columns)
            }
        return {"status": "No CSV data loaded"}