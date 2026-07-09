"""
Backtrader + Optuna proof-of-concept for the Monotonicity Strategy,
converted from monotonity_strategy.pinescript.

Pipeline (repeated for several "cycles"):
1. Load BTC-USD OHLCV data via data_mining/financial_data.py.
2. Pick two consecutive random 7-day windows (week1, week2) using a fixed
   seed, so the whole sequence of cycles is reproducible.
3. Search all parameters (lookback, tp_pct, sl_pct) with Optuna, driving the
   exact Backtrader engine, maximizing ROI on week1 only (in-sample).
4. Re-run a single Backtrader backtest on week2 with those same best params,
   with no re-optimization, to get an out-of-sample ROI.
5. After all cycles finish, print a summary table of every cycle's window,
   best params, and both ROIs.
"""
import random
from dataclasses import dataclass
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


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df sorted by a proper (UTC) DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, format='ISO8601')
    return df.sort_index()


def pick_random_two_weeks(df: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pick two consecutive, non-overlapping random 7-day windows from `df`.

    `df` must already have a sorted DatetimeIndex (see `_ensure_datetime_index`).
    `rng` is a `random.Random` instance owned by the caller: reusing the same
    instance across repeated calls advances its internal state so each call
    returns a *different* pair of weeks, while re-creating `rng` from
    RANDOM_SEED reproduces the exact same sequence of picks every run.
    """
    min_date = df.index.min()
    max_date = df.index.max() - timedelta(days=14)
    random_days = rng.randint(0, (max_date - min_date).days)
    start = min_date + timedelta(days=random_days)
    mid = start + timedelta(days=7)
    end = mid + timedelta(days=7)

    week1 = df[(df.index >= start) & (df.index < mid)]
    week2 = df[(df.index >= mid) & (df.index < end)]
    return week1, week2


@dataclass
class BacktestStats:
    """Results of one Backtrader run."""
    roi: float          # % return on starting cash
    win_rate: float      # % of *closed* trades that were winners; NaN if none closed
    num_trades: int       # number of closed trades (the win_rate denominator)
    sharpe: float          # daily, non-annualized Sharpe ratio; NaN if not computable


def run_backtrader(df: pd.DataFrame, lookback: int, tp_pct: float, sl_pct: float) -> BacktestStats:
    """Run one Backtrader backtest, return ROI/win-rate/Sharpe stats."""
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(MonotonicityStrategy, lookback=lookback, tp_pct=tp_pct, sl_pct=sl_pct)
    # Pine's default_qty_type=strategy.percent_of_equity, default_qty_value=100.
    # 99% leaves a small cash buffer so orders aren't rejected on rounding.
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    # Sharpe needs several return samples to get a non-zero stddev; with only
    # ~7 days of hourly bars per window, TimeFrame.Days (~7 samples) is about
    # the finest grain that still yields a usable (if noisy) ratio.
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=False,
    )
    start_cash = 10_000.0
    cerebro.broker.setcash(start_cash)
    strat = cerebro.run()[0]
    end_cash = cerebro.broker.getvalue()
    roi = (end_cash - start_cash) / start_cash * 100.0

    trade_analysis = strat.analyzers.trades.get_analysis()
    num_trades = trade_analysis.get("total", {}).get("closed", 0)
    won_trades = trade_analysis.get("won", {}).get("total", 0)
    win_rate = (won_trades / num_trades * 100.0) if num_trades else float("nan")

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    if sharpe is None:
        sharpe = float("nan")

    return BacktestStats(roi=roi, win_rate=win_rate, num_trades=num_trades, sharpe=sharpe)


def optuna_search(df: pd.DataFrame, n_trials: int = 30):
    """Fine-tuned search over all parameters using Optuna, maximizing ROI."""

    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int("lookback", 10, 200, step=1)
        tp_pct = trial.suggest_float("tp_pct", 0.05, 2.0)
        sl_pct = trial.suggest_float("sl_pct", 0.05, 2.0)
        return run_backtrader(df, lookback, tp_pct, sl_pct).roi

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def _format_params(params: dict) -> str:
    """Render a best-params dict as a compact, human-readable string."""
    return ", ".join(
        f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in params.items()
    )


def run_optimize_test_cycles(df: pd.DataFrame, n_cycles: int = 5, n_trials: int = 300) -> pd.DataFrame:
    """
    Repeat the "pick 2 weeks -> optimize on week1 -> test on week2" cycle
    `n_cycles` times and collect the results.

    Each cycle:
      1. Picks two consecutive random 7-day windows (week1, week2).
      2. Runs the Optuna search on week1 (in-sample) -> best_params.
      3. Re-runs a single Backtrader backtest on week1 *and* on week2 with
         those best_params (no re-optimization) to collect full stats
         (roi, win_rate, num_trades, sharpe) for both, in- and out-of-sample.

    Returns a DataFrame with one row per cycle: start_date (week1 start),
    end_date (week2 end), best_params (dict), and the week1_*/week2_* stats.
    """
    df = _ensure_datetime_index(df)
    rng = random.Random(RANDOM_SEED)

    rows = []
    for i in range(n_cycles):
        week1, week2 = pick_random_two_weeks(df, rng)
        print(f"\n=== Cycle {i + 1}/{n_cycles}: optimizing on {week1.index.min()} - {week1.index.max()} ===")

        best_params, _ = optuna_search(week1, n_trials=n_trials)
        week1_stats = run_backtrader(week1, **best_params)
        week2_stats = run_backtrader(week2, **best_params)

        print(
            f"Best params: {best_params}\n"
            f"  Week1 (in-sample):     ROI {week1_stats.roi:6.2f}% | "
            f"win rate {week1_stats.win_rate:5.1f}% ({week1_stats.num_trades} trades) | "
            f"Sharpe {week1_stats.sharpe:.2f}\n"
            f"  Week2 (out-of-sample): ROI {week2_stats.roi:6.2f}% | "
            f"win rate {week2_stats.win_rate:5.1f}% ({week2_stats.num_trades} trades) | "
            f"Sharpe {week2_stats.sharpe:.2f}"
        )

        rows.append({
            "start_date": week1.index.min(),
            "end_date": week2.index.max(),
            "best_params": best_params,
            "week1_roi": week1_stats.roi,
            "week1_win_rate": week1_stats.win_rate,
            "week1_trades": week1_stats.num_trades,
            "week1_sharpe": week1_stats.sharpe,
            "week2_roi": week2_stats.roi,
            "week2_win_rate": week2_stats.win_rate,
            "week2_trades": week2_stats.num_trades,
            "week2_sharpe": week2_stats.sharpe,
        })

    return pd.DataFrame(rows)


def _format_win_rate(win_rate: float, num_trades: int) -> str:
    """Render win-rate + sample size, e.g. '60.0% (5)'; 'n/a' if no closed trades."""
    if not num_trades:
        return "n/a (0)"
    return f"{win_rate:.1f}% ({num_trades})"


def _format_sharpe(sharpe: float) -> str:
    """Render a Sharpe ratio, or 'n/a' if it wasn't computable (e.g. no variance)."""
    return "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}"


def print_results_table(results: pd.DataFrame) -> None:
    """Pretty-print the per-cycle results as a single summary table."""
    table = pd.DataFrame({
        "start_date": results["start_date"],
        "end_date": results["end_date"],
        "best_params": results["best_params"].apply(_format_params),
        "week1_roi_%": results["week1_roi"].map(lambda v: f"{v:.2f}"),
        "week1_win_rate": [
            _format_win_rate(wr, n) for wr, n in zip(results["week1_win_rate"], results["week1_trades"])
        ],
        "week1_sharpe": results["week1_sharpe"].map(_format_sharpe),
        "week2_roi_%": results["week2_roi"].map(lambda v: f"{v:.2f}"),
        "week2_win_rate": [
            _format_win_rate(wr, n) for wr, n in zip(results["week2_win_rate"], results["week2_trades"])
        ],
        "week2_sharpe": results["week2_sharpe"].map(_format_sharpe),
    })
    print("\n=== Optimize (week1) / Out-of-sample test (week2) results ===")
    print(table.to_string(index=False))


def main():
    df = get_financial_data(ticker='BTC-USD', days=60, interval='hourly')
    df.columns = [str(c).lower() for c in df.columns]

    results = run_optimize_test_cycles(df, n_cycles=20, n_trials=1000)
    print_results_table(results)


if __name__ == "__main__":
    main()
