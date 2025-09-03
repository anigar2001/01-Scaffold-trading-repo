import pandas as pd
import math


class SimpleBacktester:
    """
    Backtester sencillo con comisiones, slippage y sizing por volatilidad.
    - fee_bps: comisión ida+vuelta por lado en puntos básicos (10 = 0.10%).
    - slippage_bps: deslizamiento aplicado al precio de entrada/salida.
    - risk_frac: fracción de capital a arriesgar por operación usando sizing por vol.
    - vol_window: ventana para volatilidad (desv. estándar de rendimientos log).
    """

    def __init__(self, price_series: pd.Series, starting_cash: float = 10000.0,
                 fee_bps: float = 0.0, slippage_bps: float = 0.0,
                 risk_frac: float = 1.0, vol_window: int = 60):
        self.prices = price_series.astype(float)
        self.cash = float(starting_cash)
        self.position = 0.0
        self.history = []
        self.fee_bps = float(fee_bps)
        self.slip_bps = float(slip_bps) if (slip_bps := slippage_bps) is not None else 0.0
        self.risk_frac = float(risk_frac)
        self.vol_window = int(vol_window)

        # Precalcula volatilidad (desv std de rendimientos log) para sizing
        self._vol = self._compute_volatility(self.prices, self.vol_window)

    @staticmethod
    def _compute_volatility(prices: pd.Series, window: int) -> pd.Series:
        rets = (prices.astype(float).pct_change().apply(lambda x: math.log(1+x)))
        return rets.rolling(window, min_periods=1).std().fillna(0.0)

    def _apply_slippage(self, price: float, side: str) -> float:
        bps = self.slip_bps / 10_000.0
        if side == 'buy':
            return price * (1 + bps)
        return price * (1 - bps)

    def _fee_cost(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10_000.0)

    def _size_by_vol(self, idx: int, price: float) -> float:
        vol = float(self._vol.iloc[idx])
        if vol <= 0:
            return self.cash * self.risk_frac
        # monto en efectivo tal que riesgo ~ risk_frac del capital (heurístico)
        target_cash = min(self.cash * self.risk_frac, self.cash)
        return max(0.0, target_cash)

    def buy_at_index(self, idx, amount: float | None = None):
        raw_price = float(self.prices.iloc[idx])
        price = self._apply_slippage(raw_price, 'buy')
        if amount is None:
            amount = self._size_by_vol(idx, price)
        amount = min(amount, self.cash)  # no apalancamiento
        if amount <= 0:
            return
        qty = amount / price
        fee = self._fee_cost(amount)
        self.position += qty
        self.cash -= (amount + fee)
        self.history.append({'idx': idx, 'action': 'buy', 'price': price, 'qty': qty, 'fee': fee})

    def sell_at_index(self, idx, qty: float | None = None):
        raw_price = float(self.prices.iloc[idx])
        price = self._apply_slippage(raw_price, 'sell')
        if qty is None:
            qty = self.position
        qty = min(qty, self.position)
        if qty <= 0:
            return
        notional = qty * price
        fee = self._fee_cost(notional)
        self.position -= qty
        self.cash += (notional - fee)
        self.history.append({'idx': idx, 'action': 'sell', 'price': price, 'qty': qty, 'fee': fee})

    def equity(self, current_price: float) -> float:
        return self.cash + self.position * float(current_price)
