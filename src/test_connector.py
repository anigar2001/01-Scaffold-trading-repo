from connectors.ccxt_connector import CCXTConnector

def test_connector():
    connector = CCXTConnector()
    balance = connector.get_balance()
    print("Balance:", balance)
    ticker = connector.get_ticker('BTC/USDT')
    print("Ticker BTC/USDT:", ticker)
    order = connector.create_order('BTC/USDT', 'buy', 'market', 0.001)
    print("Order result:", order)

if __name__ == "__main__":
    test_connector()
