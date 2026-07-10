"""
Backtrader + Optuna proof-of-concept for the Monotonicity Strategy,
converted from monotonity_strategy.pinescript.

Pipeline (repeated for several "cycles"):
1. Load BTC-USD OHLCV data via data_mining/financial_data.py.
2. Pick three consecutive periods using a fixed seed (warmup, optimization,
   test), so the whole sequence of cycles is reproducible. `warmup` is long
   enough to cover the mono index's largest possible lookback.
3. Search all parameters (lookback, tp_pct, sl_pct) with Optuna, running
   Backtrader over pd.concat([warmup, optimization]) with warmup_bars set so
   the mono index/state machine is primed during `warmup` but no trades are
   opened (and no stats collected) until `optimization` starts, maximizing
   ROI over that in-sample window only.
   tp_pct is always sampled higher than sl_pct, and any trial whose win rate
   is above MAX_WIN_RATE_PCT is pruned so it can never become "best", even
   if its ROI looks great.
4. Re-run a single Backtrader backtest over pd.concat([warmup, optimization,
   test]) with those same best params (no re-optimization) and warmup_bars
   covering warmup+optimization, so trading/stats are scoped to `test` only,
   giving an out-of-sample ROI.
5. After all cycles finish, print a summary table of every cycle's window,
   best params, and both ROIs.
"""
import os
import random
from dataclasses import dataclass, field
from datetime import timedelta, datetime

import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from data_mining.financial_data import get_financial_data

matplotlib.use("Agg")  # headless rendering for saving PNGs

RANDOM_SEED = 42
MAX_WIN_RATE_PCT = 80.0  # trials whose win rate exceeds this are pruned (never "best")
MIN_TRADES = 5  # trials with fewer closed trades than this are pruned (never "best")

# --- Monotonicity index lookback search range --------------------------------
LOOKBACK_MIN = 10
LOOKBACK_MAX = 200

# --- TP/SL search ranges (in %) -----------------------------------------------
SL_PCT_MIN = 0.05
SL_PCT_MAX = 0.9
TP_SL_MIN_MARGIN = 0.05  # tp_pct is always sampled at least this much above sl_pct
TP_PCT_MAX = 1.0

# --- Period lengths (in hours) ------------------------------------------------
# The mono index needs `lookback` bars of prior history before it produces its
# first value, so a dedicated warmup period (long enough for the *largest*
# lookback Optuna can try) is placed right before the optimization period.
# This way the optimization/test periods themselves are never "wasted" on
# warmup and can generate signals/trades right from their first bar.
WARMUP_PERIOD_HOURS = LOOKBACK_MAX + 2
OPTIMIZATION_PERIOD_HOURS = 4 * 7 * 24  # in hours, in-sample
TEST_PERIOD_HOURS = 4 * 7 * 24  # in hours, out-of-sample

# --- Cycle / Optuna run sizes --------------------------------------------------
N_CYCLES = 20
N_TRIALS = 1000

# --- Trade plot capture ---------------------------------------------------------
PRE_TRADE_BARS = 10   # bars of OHLC/mono-index history to include before entry
POST_TRADE_BARS = 10  # bars of OHLC/mono-index history to include after exit
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


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

    `warmup_bars`: the strategy keeps computing the mono index and updating
    the trade_state machine on every bar (so it's fully "warmed up" by the
    time it matters), but it won't open any *new* trades while
    len(self.data) <= warmup_bars. This lets a caller feed in extra leading
    history purely to prime the indicator/state without that history's bars
    contributing any trades to the resulting stats.
    """

    params = dict(
        lookback=60,
        tp_pct=0.3,
        sl_pct=0.2,
        warmup_bars=0,
    )

    def __init__(self):
        self.trade_state = 0
        self._mono_index = None
        self._prev_mono_index = None
        self.order = None  # tracks the pending/active entry bracket

        # Full per-bar mono-index history (1 entry per `next()` call, in the
        # same order/indexing as the data feed itself), so trade plots can
        # later slice out OHLC + mono-index around any trade's bar range.
        self.mono_index_history: list[float | None] = []

        # Per-trade capture: entry bar index (1-based, matching len(self.data)
        # at the time the position was opened) keyed by trade.ref, and the
        # list of completed trades' info dicts.
        self._trade_entry_bars: dict[int, int] = {}
        self.trades_info: list[dict] = []

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            if order == self.order:
                self.order = None

    def notify_trade(self, trade):
        if trade.justopened:
            self._trade_entry_bars[trade.ref] = len(self.data)
        if trade.isclosed:
            entry_bar = self._trade_entry_bars.pop(trade.ref, None)
            exit_bar = len(self.data)
            entry_value = abs(trade.value) or 1.0
            pnl_pct = trade.pnlcomm / entry_value * 100.0
            self.trades_info.append({
                "index": len(self.trades_info) + 1,
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "direction": "long" if trade.long else "short",
                "win": trade.pnlcomm > 0,
                "pnl_pct": pnl_pct,
            })

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
        self.mono_index_history.append(self._mono_index)

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

        # Still within the warmup window: keep the state machine ticking
        # over (above) so it's primed correctly, but don't act on it yet.
        if len(self.data) <= self.p.warmup_bars:
            return

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


def pick_random_periods(df: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pick three consecutive, non-overlapping, plain segments from `df`:

        [warmup][optimization][test]

    Callers are responsible for actually using `warmup` to prime the mono
    index/state machine (see MonotonicityStrategy's `warmup_bars` param):
      - to optimize: run on pd.concat([warmup, optimization]) with
        warmup_bars=len(warmup), so trading is only allowed once the
        `optimization` segment starts.
      - to test: run on pd.concat([warmup, optimization, test]) with
        warmup_bars=len(warmup) + len(optimization), so trading is only
        allowed once the `test` segment starts.

    `df` must already have a sorted DatetimeIndex (see `_ensure_datetime_index`).
    `rng` is a `random.Random` instance owned by the caller: reusing the same
    instance across repeated calls advances its internal state so each call
    returns a *different* triple of periods, while re-creating `rng` from
    RANDOM_SEED reproduces the exact same sequence of picks every run.
    """
    total_hours = WARMUP_PERIOD_HOURS + OPTIMIZATION_PERIOD_HOURS + TEST_PERIOD_HOURS
    min_date = df.index.min()
    max_date = df.index.max() - timedelta(hours=total_hours)
    random_hours = rng.randint(0, int((max_date - min_date).total_seconds() // 3600))
    warmup_start = min_date + timedelta(hours=random_hours)
    opt_start = warmup_start + timedelta(hours=WARMUP_PERIOD_HOURS)
    test_start = opt_start + timedelta(hours=OPTIMIZATION_PERIOD_HOURS)
    test_end = test_start + timedelta(hours=TEST_PERIOD_HOURS)

    warmup = df[(df.index >= warmup_start) & (df.index < opt_start)]
    optimization = df[(df.index >= opt_start) & (df.index < test_start)]
    test = df[(df.index >= test_start) & (df.index < test_end)]
    return warmup, optimization, test


@dataclass
class BacktestStats:
    """Results of one Backtrader run."""
    roi: float          # % return on starting cash
    win_rate: float      # % of *closed* trades that were winners; NaN if none closed
    num_trades: int       # number of closed trades (the win_rate denominator)
    sharpe: float          # daily, non-annualized Sharpe ratio; NaN if not computable
    trades_info: list = field(default_factory=list)         # per-trade dicts (entry/exit bar, direction, win, pnl_pct)
    mono_index_history: list = field(default_factory=list)  # mono-index value per bar (same indexing as `df`)


def run_backtrader(
    df: pd.DataFrame, lookback: int, tp_pct: float, sl_pct: float, warmup_bars: int = 0
) -> BacktestStats:
    """Run one Backtrader backtest, return ROI/win-rate/Sharpe stats.

    `df` may include leading history purely to warm up the mono index/state
    machine: pass `warmup_bars=len(that leading history)` so the strategy
    still processes it bar-by-bar (priming its indicator/state) but won't
    open any trades until past it, keeping ROI/win_rate/num_trades/sharpe
    scoped to the bars after the warmup.
    """
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(
        MonotonicityStrategy, lookback=lookback, tp_pct=tp_pct, sl_pct=sl_pct, warmup_bars=warmup_bars
    )
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

    return BacktestStats(
        roi=roi, win_rate=win_rate, num_trades=num_trades, sharpe=sharpe,
        trades_info=list(strat.trades_info), mono_index_history=list(strat.mono_index_history),
    )


def optuna_search(df: pd.DataFrame, n_trials: int = 30, warmup_bars: int = 0):
    """
    Fine-tuned search over all parameters using Optuna, maximizing ROI.

    `df` may include leading warmup history; see `run_backtrader`'s
    `warmup_bars` docstring -- it's passed straight through here.

    Two constraints are enforced directly in the objective, not just as a
    post-hoc filter, so they hold for every trial Optuna considers:
      - tp_pct is always sampled strictly higher than sl_pct (sl_pct is
        drawn first, then tp_pct is drawn from (sl_pct, 2.0]).
      - any trial whose win rate is above MAX_WIN_RATE_PCT, or whose number
        of closed trades is below MIN_TRADES, is pruned via
        `optuna.TrialPruned`, which excludes it from `study.best_*` -- so
        the reported "best" is always the highest ROI *among* parameter
        sets with win rate <= MAX_WIN_RATE_PCT and at least MIN_TRADES
        closed trades, never a high-win-rate or low-sample-size one.
    """

    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int("lookback", LOOKBACK_MIN, LOOKBACK_MAX, step=1)
        sl_pct = trial.suggest_float("sl_pct", SL_PCT_MIN, SL_PCT_MAX)
        tp_pct = trial.suggest_float("tp_pct", sl_pct + TP_SL_MIN_MARGIN, TP_PCT_MAX)

        stats = run_backtrader(df, lookback, tp_pct, sl_pct, warmup_bars=warmup_bars)
        if stats.win_rate > MAX_WIN_RATE_PCT:
            raise optuna.TrialPruned()
        if stats.num_trades < MIN_TRADES:
            raise optuna.TrialPruned()
        return stats.roi

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {n_trials} trials were pruned (win rate above {MAX_WIN_RATE_PCT:.0f}% "
            f"or fewer than {MIN_TRADES} closed trades) -- try increasing n_trials."
        )
    return study.best_params, study.best_value


def _format_params(params: dict) -> str:
    """Render a best-params dict as a compact, human-readable string."""
    return ", ".join(
        f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in params.items()
    )


def run_optimize_test_cycles(
    df: pd.DataFrame, n_cycles: int = N_CYCLES, n_trials: int = N_TRIALS, run_dir: str | None = None
) -> pd.DataFrame:
    """
    Repeat the "pick warmup/optimization/test periods -> optimize -> test"
    cycle `n_cycles` times and collect the results.

    Each cycle:
      1. Picks a warmup period, an optimization period, and a test period,
         all consecutive and non-overlapping (see `pick_random_periods`).
      2. Runs the Optuna search on pd.concat([warmup, optimization]), with
         warmup_bars=len(warmup) so the mono index/state machine is primed
         over the warmup bars but no trades open until the optimization
         segment starts -> best_params.
      3. Re-runs a single Backtrader backtest on that same
         warmup+optimization data (warmup_bars=len(warmup)) *and* on
         pd.concat([warmup, optimization, test]) (warmup_bars=len(warmup) +
         len(optimization), so trading is only allowed once the `test`
         segment starts) with those best_params (no re-optimization) to
         collect full stats (roi, win_rate, num_trades, sharpe) for both,
         in- and out-of-sample.

    Returns a DataFrame with one row per cycle: start_date (optimization
    period start), end_date (test period end), best_params (dict), and the
    optim_*/test_* stats (optim = optimization period, test = test period).

    If `run_dir` is given, a `Cycle {i+1}` subfolder is created inside it for
    every cycle, and a PNG (price + monotonicity-index) is saved for every
    closed trade (both from the optimization and the test period), named
    `<optim|test>_{trade_index}_<win|lose>_<long|short>_<pnl>percent.png`.
    """
    df = _ensure_datetime_index(df)
    rng = random.Random(RANDOM_SEED)

    rows = []
    for i in range(n_cycles):
        warmup, optimization, test = pick_random_periods(df, rng)
        warmup_bars = len(warmup)
        print(f"\n=== Cycle {i + 1}/{n_cycles}: optimizing on {optimization.index.min()} - {optimization.index.max()} ===")

        optim_data = pd.concat([warmup, optimization])
        test_data = pd.concat([warmup, optimization, test])

        best_params, _ = optuna_search(optim_data, n_trials=n_trials, warmup_bars=warmup_bars)
        optim_stats = run_backtrader(optim_data, **best_params, warmup_bars=warmup_bars)
        test_stats = run_backtrader(test_data, **best_params, warmup_bars=warmup_bars + len(optimization))

        print(
            f"Best params: {best_params}\n"
            f"  Optim: ROI {optim_stats.roi:6.2f}% | "
            f"win rate {optim_stats.win_rate:5.1f}% ({optim_stats.num_trades} trades) | "
            f"Sharpe {optim_stats.sharpe:.2f}\n"
            f"  Test:  ROI {test_stats.roi:6.2f}% | "
            f"win rate {test_stats.win_rate:5.1f}% ({test_stats.num_trades} trades) | "
            f"Sharpe {test_stats.sharpe:.2f}"
        )

        if run_dir is not None:
            cycle_dir = os.path.join(run_dir, f"Cycle {i + 1}")
            save_all_trade_plots(
                optim_data, optim_stats, cycle_dir, PRE_TRADE_BARS, POST_TRADE_BARS, file_prefix="optim_"
            )
            save_all_trade_plots(
                test_data, test_stats, cycle_dir, PRE_TRADE_BARS, POST_TRADE_BARS, file_prefix="test_"
            )

        rows.append({
            "start_date": optimization.index.min(),
            "end_date": test.index.max(),
            "best_params": best_params,
            "optim_roi": optim_stats.roi,
            "optim_win_rate": optim_stats.win_rate,
            "optim_trades": optim_stats.num_trades,
            "optim_sharpe": optim_stats.sharpe,
            "test_roi": test_stats.roi,
            "test_win_rate": test_stats.win_rate,
            "test_trades": test_stats.num_trades,
            "test_sharpe": test_stats.sharpe,
        })

    return pd.DataFrame(rows)


def _sanitize_filename(name: str) -> str:
    """Strip characters that aren't safe in filenames."""
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


def save_trade_plot(
    df: pd.DataFrame,
    mono_index_history: list,
    trade: dict,
    pre_bars: int,
    post_bars: int,
    out_dir: str,
    file_prefix: str = "",
) -> None:
    """
    Render and save a two-panel (price + monotonicity index) chart for one
    trade, in the same dark-themed style as monotonity_index.py, covering
    `pre_bars` bars before entry through `post_bars` bars after exit.

    `df` is the exact DataFrame the strategy ran on (1-based bar indices in
    `trade` correspond 1:1 to `df.iloc[bar - 1]`). `mono_index_history` is
    the per-bar mono-index list captured from the strategy (same indexing,
    0-based, i.e. `mono_index_history[bar - 1]`).
    """
    entry_bar = trade["entry_bar"]
    exit_bar = trade["exit_bar"]
    if entry_bar is None or exit_bar is None:
        return

    start_idx = max(0, entry_bar - 1 - pre_bars)
    end_idx = min(len(df), exit_bar + post_bars)  # exclusive upper bound

    ohlc_slice = df.iloc[start_idx:end_idx]
    mono_slice = np.array(
        [mono_index_history[i] if i < len(mono_index_history) else np.nan for i in range(start_idx, end_idx)],
        dtype=float,
    )
    x = np.arange(len(ohlc_slice))
    entry_x = (entry_bar - 1) - start_idx
    exit_x = (exit_bar - 1) - start_idx

    close = ohlc_slice["close"].to_numpy()

    win_lose = "win" if trade["win"] else "lose"
    direction = trade["direction"]
    trade_idx = trade.get("index")
    pnl_pct = trade["pnl_pct"]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle(
        f"Monotonicity Analysis - Trade #{trade_idx} | {direction.upper()} | "
        f"{win_lose.upper()} | PnL {pnl_pct:+.2f}% | entry bar {entry_bar} -> exit bar {exit_bar}",
        fontsize=13, fontweight='bold', color='white',
    )

    # --- Price subplot ---
    ax1.plot(x, close, label='Price', color='#00BFFF', linewidth=1.2)
    ax1.axvline(entry_x, color='#00FF88', linestyle='--', linewidth=1.2, label='Entry')
    ax1.axvline(exit_x, color='#FF4444', linestyle='--', linewidth=1.2, label='Exit')
    ax1.set_title('Price', color='white')
    ax1.set_ylabel('Price', color='white')
    ax1.legend(facecolor='#222222', edgecolor='gray')
    ax1.grid(True, which='major', color='#444444', linestyle='-', linewidth=0.8, alpha=0.9)
    ax1.grid(True, which='minor', color='#333333', linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.minorticks_on()
    ax1.set_facecolor('#111111')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555555')

    # --- Monotonicity subplot ---
    ax2.plot(x, mono_slice, label='Monotonicity Index', color='#FFA500', linewidth=1.4)
    ax2.axhline(0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.6, label='Neutral (0.5)')
    ax2.axhline(0.75, color='#00FF88', linestyle=':', linewidth=0.8, alpha=0.6, label='Strong up (0.75)')
    ax2.axhline(0.25, color='#FF6666', linestyle=':', linewidth=0.8, alpha=0.6, label='Strong down (0.25)')
    ax2.axvline(entry_x, color='#00FF88', linestyle='--', linewidth=1.2, label='Entry')
    ax2.axvline(exit_x, color='#FF4444', linestyle='--', linewidth=1.2, label='Exit')
    ax2.fill_between(x, mono_slice, 0.5, where=(mono_slice >= 0.5), alpha=0.15, color='#00FF88')
    ax2.fill_between(x, mono_slice, 0.5, where=(mono_slice < 0.5), alpha=0.15, color='#FF4444')
    ax2.set_title('Monotonicity Index', color='white')
    ax2.set_ylabel('Index Value', color='white')
    ax2.set_xlabel('Bar', color='white')
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor='#222222', edgecolor='gray', ncol=2, fontsize=8)
    ax2.grid(True, which='major', color='#444444', linestyle='-', linewidth=0.8, alpha=0.9)
    ax2.grid(True, which='minor', color='#333333', linestyle='--', linewidth=0.5, alpha=0.6)
    ax2.minorticks_on()
    ax2.set_facecolor('#111111')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555555')

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    pnl_str = f"{trade['pnl_pct']:.2f}".replace("-", "neg").replace(".", "p")
    filename = _sanitize_filename(f"{file_prefix}{trade_idx}_{win_lose}_{direction}_{pnl_str}percent.png")
    fig.savefig(os.path.join(out_dir, filename), dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_all_trade_plots(
    df: pd.DataFrame, stats: BacktestStats, out_dir: str, pre_bars: int, post_bars: int, file_prefix: str = ""
) -> None:
    """Save a trade plot for every closed trade recorded in `stats`."""
    for trade in stats.trades_info:
        save_trade_plot(df, stats.mono_index_history, trade, pre_bars, post_bars, out_dir, file_prefix=file_prefix)


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
        "optim_roi_%": results["optim_roi"].map(lambda v: f"{v:.2f}"),
        "optim_win_rate": [
            _format_win_rate(wr, n) for wr, n in zip(results["optim_win_rate"], results["optim_trades"])
        ],
        "optim_sharpe": results["optim_sharpe"].map(_format_sharpe),
        "test_roi_%": results["test_roi"].map(lambda v: f"{v:.2f}"),
        "test_win_rate": [
            _format_win_rate(wr, n) for wr, n in zip(results["test_win_rate"], results["test_trades"])
        ],
        "test_sharpe": results["test_sharpe"].map(_format_sharpe),
    })
    print("\n=== Optimize / Test results ===")
    print(table.to_string(index=False))


def main():
    df = get_financial_data(ticker='BTC-USD', days=730, interval='hourly')
    df.columns = [str(c).lower() for c in df.columns]

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to: {run_dir}")

    results = run_optimize_test_cycles(df, n_cycles=N_CYCLES, n_trials=N_TRIALS, run_dir=run_dir)
    print_results_table(results)


if __name__ == "__main__":
    main()
