import time
import uuid
from typing import Dict


from .base_connector import BaseConnector


class MockExchange(BaseConnector):
    def __init__(self):
        # balances simples
        self.balances = {'USDT': 100000.0, 'BTC': 1.0}
        self.orders: Dict[str, dict] = {}


    def get_balance(self):
        return self.balances


    def get_ticker(self, symbol: str):
        # simbolo ejemplo: 'BTC/USDT'
        base, quote = symbol.split('/')
        # ticker simulado
        return {'symbol': symbol, 'bid': 50000.0, 'ask': 50001.0, 'last': 50000.5, 'timestamp': int(time.time()*1000)}


    def create_order(self, symbol: str, side: str, type: str, amount: float, price: float = None):
        oid = str(uuid.uuid4())
        # Simula fill inmediato
        ticker = self.get_ticker(symbol)
        fill_price = ticker['ask'] if side == 'buy' else ticker['bid']
        filled_cost = amount * fill_price


        # Ajusta balances simplificados
        base, quote = symbol.split('/')
        if side == 'buy':
            self.balances[base] = self.balances.get(base, 0) + amount
            self.balances[quote] = self.balances.get(quote, 0) - filled_cost
        else:
            self.balances[base] = self.balances.get(base, 0) - amount
            self.balances[quote] = self.balances.get(quote, 0) + filled_cost


        order = {'id': oid, 'symbol': symbol, 'side': side, 'type': type, 'amount': amount, 'price': fill_price, 'status': 'closed', 'filled': amount}
        self.orders[oid] = order
        return order


    def fetch_order(self, order_id: str):
        return self.orders.get(order_id)