# src/order_manager.py

class OrderManager:
    def __init__(self, connector):
        self.connector = connector

    def create_order(self, symbol, side, type, amount, price=None):
        """
        Crea una orden usando el conector (CCXT o MockExchange)
        """
        return self.connector.create_order(symbol, side, type, amount, price)
