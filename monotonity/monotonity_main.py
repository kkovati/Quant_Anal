"""
Backtrader + vectorbt/Optuna proof-of-concept for the Monotonicity Strategy,
converted from monotonity_strategy.pinescript.

Pipeline:
1. Load BTC-USD OHLCV data via data_mining/financial_data.py.
2. Pick one random 7-day window, once, using a fixed seed (reused for every
   optimization run below).
3. Coarse-search the `lookback` parameter with vectorbt (fast, vectorized
   approximation of the strategy).
4. Fine-search all parameters (lookback, tp_pct, sl_pct) with Optuna,
   driving the exact Backtrader engine, maximizing ROI.
"""
import random
from datetime import timedelta

import backtrader as bt
import numpy as np
import optuna
import pandas as pd
import vectorbt as vbt

from data_mining.financial_data import get_financial_data

RANDOM_SEED = 42


class MonotonicityStrategy(bt.Strategy):
    """
    Direct Backtrader port of monotonity_strategy.pinescript.

    Monotonicity index (needs `lookback` bars of history):
        mono_index = count(close[j] < close, for j in 1..lookback) / lookback
    -> 0.0 : current close is the lowest in the window (extreme low)
    -> 1.0 : current close is the highest in the window (extreme high)

    State machine (only updated while flat):
        0  idle          -> primes to  1 when mono_index <= 0.0
                          -> primes to -1 when mono_index >= 1.0
        1  primed long   -> enters long  once mono_index crosses OVER  0.5
       -1  primed short  -> enters short once mono_index crosses UNDER 0.5
    Entering resets state to 0 (matches Pine's `trade_state := 0` on entry).

    Exit is a fixed TP/SL bracket set at the entry price (mirrors
    strategy.exit(..., limit=avg_price*(1±tp_pct), stop=avg_price*(1∓sl_pct))).
    """

    params = dict(
        lookback=60,
        tp_pct=0.3,
        sl_pct=0.2,
    )

    def __init__(self):
        self.trade_state = 0
        self._mono_index = None
        self._prev_mono_index = None
        self.order = None  # tracks the pending/active entry bracket

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            if order == self.order:
                self.order = None

    def _compute_mono_index(self):
        lookback = self.p.lookback
        if len(self.data) <= lookback:
            return None
        current_close = self.data.close[0]
        count = sum(1 for j in range(1, lookback + 1) if self.data.close[-j] < current_close)
        return count / lookback

    def next(self):
        self._prev_mono_index = self._mono_index
        self._mono_index = self._compute_mono_index()

        # Pine only updates the state machine / looks for entries while flat.
        if self.position or self.order or self._mono_index is None:
            return

        if self.trade_state == 0:
            if self._mono_index <= 0.0:
                self.trade_state = 1
            elif self._mono_index >= 1.0:
                self.trade_state = -1

        if self._prev_mono_index is None:
            return

        crossover = self._prev_mono_index <= 0.5 < self._mono_index
        crossunder = self._prev_mono_index >= 0.5 > self._mono_index
        price = self.data.close[0]

        if self.trade_state == 1 and crossover:
            self.order = self.buy_bracket(
                exectype=bt.Order.Market,
                stopprice=price * (1 - self.p.sl_pct / 100),
                limitprice=price * (1 + self.p.tp_pct / 100),
            )[0]
            self.trade_state = 0
        elif self.trade_state == -1 and crossunder:
            self.order = self.sell_bracket(
                exectype=bt.Order.Market,
                stopprice=price * (1 + self.p.sl_pct / 100),
                limitprice=price * (1 - self.p.tp_pct / 100),
            )[0]
            self.trade_state = 0


def pick_random_week(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one random 7-day window, once, using a fixed seed."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, format='ISO8601')
    df = df.sort_index()

    random.seed(RANDOM_SEED)
    min_date = df.index.min()
    max_date = df.index.max() - timedelta(days=7)
    random_days = random.randint(0, (max_date - min_date).days)
    start = min_date + timedelta(days=random_days)
    end = start + timedelta(days=7)
    return df.loc[start:end]


def run_backtrader(df: pd.DataFrame, lookback: int, tp_pct: float, sl_pct: float) -> float:
    """Run one Backtrader backtest, return ROI in %."""
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(MonotonicityStrategy, lookback=lookback, tp_pct=tp_pct, sl_pct=sl_pct)
    # Pine's default_qty_type=strategy.percent_of_equity, default_qty_value=100.
    # 99% leaves a small cash buffer so orders aren't rejected on rounding.
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99)
    start_cash = 10_000.0
    cerebro.broker.setcash(start_cash)
    cerebro.run()
    end_cash = cerebro.broker.getvalue()
    return (end_cash - start_cash) / start_cash * 100.0


def _compute_mono_index_series(close: pd.Series, lookback: int) -> pd.Series:
    """Vectorized rolling monotonicity index (see MonotonicityStrategy docstring)."""
    count = pd.Series(0.0, index=close.index)
    for j in range(1, lookback + 1):
        count = count.add((close.shift(j) < close).astype(float), fill_value=0.0)
    mono_index = count / lookback
    mono_index.iloc[:lookback] = np.nan
    return mono_index


def _simulate_signals(close: pd.Series, mono_index: pd.Series, tp_pct: float, sl_pct: float):
    """
    Stateful replication of the Pine state machine + a close-price
    approximation of the TP/SL exit (used only so the state machine knows
    when the position is "flat" again; the actual PnL/ROI is computed by
    vbt.Portfolio.from_signals via tp_stop/sl_stop, which uses OHLC).
    """
    c = close.to_numpy()
    m = mono_index.to_numpy()
    n = len(c)
    long_entries = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)

    trade_state = 0
    position = 0  # 0 flat, 1 long, -1 short
    entry_price = np.nan
    prev_mono = np.nan

    for i in range(n):
        cur_mono = m[i]
        price = c[i]

        if position == 1:
            change = (price - entry_price) / entry_price * 100
            if change >= tp_pct or change <= -sl_pct:
                position = 0
        elif position == -1:
            change = (entry_price - price) / entry_price * 100
            if change >= tp_pct or change <= -sl_pct:
                position = 0

        if position == 0 and not np.isnan(cur_mono):
            if trade_state == 0:
                if cur_mono <= 0.0:
                    trade_state = 1
                elif cur_mono >= 1.0:
                    trade_state = -1

            if not np.isnan(prev_mono):
                crossover = prev_mono <= 0.5 < cur_mono
                crossunder = prev_mono >= 0.5 > cur_mono
                if trade_state == 1 and crossover:
                    long_entries[i] = True
                    trade_state = 0
                    position = 1
                    entry_price = price
                elif trade_state == -1 and crossunder:
                    short_entries[i] = True
                    trade_state = 0
                    position = -1
                    entry_price = price

        prev_mono = cur_mono

    idx = close.index
    return pd.Series(long_entries, index=idx), pd.Series(short_entries, index=idx)


def vectorbt_search(df: pd.DataFrame, lookback_range, tp_pct: float = 0.3, sl_pct: float = 0.2):
    """Coarse grid search over `lookback` using vectorbt (tp/sl held fixed)."""
    close = df["close"]
    best_roi, best_params = -np.inf, None
    for lookback in lookback_range:
        mono_index = _compute_mono_index_series(close, lookback)
        long_entries, short_entries = _simulate_signals(close, mono_index, tp_pct, sl_pct)
        if not (long_entries.any() or short_entries.any()):
            continue
        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=False,
            short_entries=short_entries,
            short_exits=False,
            tp_stop=tp_pct / 100,
            sl_stop=sl_pct / 100,
            init_cash=10_000.0,
        )
        roi = pf.total_return() * 100.0
        if roi > best_roi:
            best_roi, best_params = roi, lookback
    return best_params, best_roi


def optuna_search(df: pd.DataFrame, n_trials: int = 30):
    """Fine-tuned search over all parameters using Optuna, maximizing ROI."""

    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int("lookback", 10, 80, step=5)
        tp_pct = trial.suggest_float("tp_pct", 0.05, 2.0)
        sl_pct = trial.suggest_float("sl_pct", 0.05, 2.0)
        return run_backtrader(df, lookback, tp_pct, sl_pct)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def main():
    df = get_financial_data(ticker='BTC-USD', days=60, interval='hourly')
    df.columns = [str(c).lower() for c in df.columns]

    week_df = pick_random_week(df)
    print(f"Selected week: {week_df.index.min()} - {week_df.index.max()}")

    vbt_lookback, vbt_roi = vectorbt_search(week_df, range(5, 60, 5))
    print(f"vectorbt best lookback: {vbt_lookback}, ROI: {vbt_roi:.2f}%")

    optuna_params, optuna_roi = optuna_search(week_df)
    print(f"Optuna best params: {optuna_params}, ROI: {optuna_roi:.2f}%")


if __name__ == "__main__":
    main()