"""
Backtrader + Optuna proof-of-concept for the Monotonicity Strategy,
converted from monotonity_strategy.pinescript.

Pipeline:
1. Load BTC-USD OHLCV data via data_mining/financial_data.py.
2. Pick one random 7-day window, once, using a fixed seed (reused for every
   optimization run below).
3. Search all parameters (lookback, tp_pct, sl_pct) with Optuna, driving the
   exact Backtrader engine, maximizing ROI.
"""
import random
from datetime import timedelta

import backtrader as bt
import optuna
import pandas as pd

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


def optuna_search(df: pd.DataFrame, n_trials: int = 30):
    """Fine-tuned search over all parameters using Optuna, maximizing ROI."""

    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int("lookback", 10, 80, step=5)
        tp_pct = trial.suggest_float("tp_pct", 0.05, 2.0)
        sl_pct = trial.suggest_float("sl_pct", 0.05, 2.0)
        return run_backtrader(df, lookback, tp_pct, sl_pct)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def main():
    df = get_financial_data(ticker='BTC-USD', days=60, interval='hourly')
    df.columns = [str(c).lower() for c in df.columns]

    week_df = pick_random_week(df)
    print(f"Selected week: {week_df.index.min()} - {week_df.index.max()}")

    optuna_params, optuna_roi = optuna_search(week_df, n_trials=1000)
    print(f"Optuna best params: {optuna_params}, ROI: {optuna_roi:.2f}%")


if __name__ == "__main__":
    main()