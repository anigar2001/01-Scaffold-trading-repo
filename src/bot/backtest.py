# src/bot/backtest.py
def simple_backtest(df, signals, amount=0.001, fee=0.00075):
    cash = 100000.0
    btc = 0.0
    trades = []
    for i,row in df.iterrows():
        sig = signals[i]  # 'buy'/'sell'/'hold'
        price = row['close']
        if sig == 'buy' and cash > price*amount:
            cash -= price*amount*(1+fee)
            btc += amount
            trades.append({'timestamp':row['timestamp'],'side':'buy','price':price})
        elif sig == 'sell' and btc >= amount:
            cash += price*amount*(1-fee)
            btc -= amount
            trades.append({'timestamp':row['timestamp'],'side':'sell','price':price})
    final_value = cash + btc * df['close'].iloc[-1]
    return trades, final_value
