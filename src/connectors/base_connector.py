from abc import ABC, abstractmethod


class BaseConnector(ABC):
    @abstractmethod
    def get_balance(self):
        pass


    @abstractmethod
    def get_ticker(self, symbol: str):
       pass


    @abstractmethod
    def create_order(self, symbol: str, side: str, type: str, amount: float, price: float = None):
        pass


    @abstractmethod
    def fetch_order(self, order_id: str):
        pass