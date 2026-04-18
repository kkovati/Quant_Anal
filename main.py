from collections import Counter
import math
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import random
from tqdm import tqdm


INITIAL_BUDGET = 1000
COMMISSION = 0.001  # 0.1% per trade


class Ohcl:
    IN = 0
    OUT_HI = 1
    OUT_LO = 2
    OUT_BOTH = 3

    def __init__(self, open_, high, low, close):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        assert high >= open_ >= low and high >= close >= low, f"Wrong Ohcl {open_, high, low, close}"

    def is_out_of_bounds(self, hi, lo):
        above_upper_bound = max(self.open, self.high, self.low, self.close) > hi
        below_lower_bound = min(self.open, self.high, self.low, self.close) < lo
        if above_upper_bound and below_lower_bound:
            return self.OUT_BOTH
        if above_upper_bound:
            return self.OUT_HI
        if below_lower_bound:
            return self.OUT_LO
        return self.IN


class Timeseries:
    def __init__(self, path: str):
        # self.df = pd.read_csv(
        #     "data/Bitcoin_Historical_Data/btcusd_1-min_data.csv",
        #     parse_dates=["Timestamp"],
        #     index_col="Timestamp"
        # )  # noqa

        self.df = pd.read_csv(path)  # noqa

        # print(df.head())

    def __iter__(self):
        return self

    def __next__(self):
        """
        VERY SLOW!
        Usage:
        for idx, ohcl in ts:
            print(idx, ohcl.open, ohcl.high, ohcl.low, ohcl.close)
        """
        idx, row = next(self.df.iterrows())  # noqa
        # return idx, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
        return idx, Ohcl(row["Open"], row["High"], row["Low"], row["Close"])


def log_uniform(min_val=0.1, max_val=1000):
    # Sample uniformly in log space
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    log_sample = random.uniform(log_min, log_max)
    return 10 ** log_sample


def generate_random_ch_breakout_trader(mode):
    if mode == "independent":
        params = {
            'win_percentage': log_uniform(0.1, 10),
            'lose_percentage': log_uniform(0.1, 10),
            'first_position_size_percentage': log_uniform(1, 100),
            'position_increment_percentage': log_uniform(10, 1000),
            'next_direction_strategy': random.choice(['always_up', 'always_down', 'alternate']),
            'optimistic': random.choice([True, False])
        }
    elif mode == "correlated":
        win_percentage = log_uniform(0.1, 10)
        win_lose_ratio = log_uniform(1, 20)
        if random.choice([True, False]):
            win_lose_ratio = 1 / win_lose_ratio
        lose_percentage = win_percentage * win_lose_ratio
        params = {
            'win_percentage': win_percentage,
            'lose_percentage': lose_percentage,
            'first_position_size_percentage': log_uniform(1, 100),
            'position_increment_percentage': log_uniform(10, 1000),
            'next_direction_strategy': random.choice(['always_up', 'always_down', 'alternate']),
            'optimistic': random.choice([True, False])
        }
    if mode == "fixed":
        params = {
            'win_percentage': 1,
            'lose_percentage': 2,
            'first_position_size_percentage': log_uniform(0.5, 100),
            'position_increment_percentage': log_uniform(10, 1000),
            'next_direction_strategy': random.choice(['always_up', 'always_down', 'alternate']),
            'optimistic': random.choice([True, False])
        }
    elif mode == "custom":
        params = {
            'win_percentage': 7,
            'lose_percentage': 7,
            'position_size_percentage_list': [10, 10, 22.5, 44, 100, 100, 100],
            #'position_size_percentage_list': [],
            'first_position_size_percentage': 1,
            'position_increment_percentage': 250,
            'next_direction_strategy': 'always_up',
            'optimistic': True
        }
        params = {
            'win_percentage': 8,
            'lose_percentage': 5,
            'position_size_percentage_list': [10, 10, 22.5, 44, 100, 100, 100],
            #'position_size_percentage_list': [],
            'first_position_size_percentage': 1,
            'position_increment_percentage': 250,
            'next_direction_strategy': 'always_up',
            'optimistic': True
        }
    else:
        assert False
    return ChBreakoutTrader(**params, params=params)


class ChBreakoutTrader:
    NO_TRADE = 0
    UP_TRADE = 1
    DOWN_TRADE = 2
    IN_REGION_1 = 3
    IN_REGION_2 = 4

    def __init__(self,
                 win_percentage,
                 lose_percentage,
                 position_size_percentage_list,
                 first_position_size_percentage,
                 position_increment_percentage,
                 next_direction_strategy,
                 optimistic,
                 params):

        self.params = params

        self.win_percentage = win_percentage
        self.lose_percentage = lose_percentage

        if position_size_percentage_list:
            assert 0 < min(position_size_percentage_list) and max(position_size_percentage_list) <= 100, \
                "position_size_percentage_list values must be in (0, 100]"
            self.position_size_percentage_list = position_size_percentage_list
        else:
            assert 0 < first_position_size_percentage <= 100, "first_position_size_percentage must be in (0, 100]"
            assert 0 < position_increment_percentage, "position_increment_percentage must be > 0"
            self.position_size_percentage_list = []
            for i in range(100):
                ps = first_position_size_percentage * (1 + position_increment_percentage / 100) ** i
                self.position_size_percentage_list.append(min(ps, 100))  # cap at 100%
        self.position_size_percentage_list_idx = 0

        assert next_direction_strategy in ['always_up', 'always_down', 'alternate']
        self.next_direction_strategy = next_direction_strategy
        if next_direction_strategy in ('always_up', 'alternate'):
            self.direction = 'up'
        elif next_direction_strategy in 'always_down':
            self.direction = 'down'
        else:
            raise ValueError(f"Unknown next_direction_strategy: {next_direction_strategy}")

        self.optimistic = optimistic

        self.state = self.NO_TRADE
        self.a = self.b = self.c = self.d = None

        self.session = {}  # current session
        self.sessions = []  # all sessions

        self.session_samples = {}
        self.position_size_list = []
        self.profit_list = []

        self.const_budget = INITIAL_BUDGET
        # self.real_budget = INITIAL_BUDGET

    def update(self, ohcl: Ohcl):
        if self.state == self.NO_TRADE:
            self.a = ohcl.close * (1 + self.win_percentage / 100)
            self.b = ohcl.close
            self.c = ohcl.close * (1 - self.lose_percentage / 100)
            self.d = ohcl.close * (1 - (self.lose_percentage + self.win_percentage) / 100)

            if self.next_direction_strategy in 'always_up':
                self.state = self.UP_TRADE
            elif self.next_direction_strategy in 'always_down':
                self.state = self.DOWN_TRADE
            elif self.next_direction_strategy in 'alternate':
                if self.direction == 'up':
                    self.state = self.UP_TRADE
                    self.direction = 'down'
                else:
                    self.state = self.DOWN_TRADE
                    self.direction = 'up'
            else:
                raise ValueError(f"Unknown next_direction_strategy: {self.next_direction_strategy}")

            self.const_budget = INITIAL_BUDGET
            self.position_size_percentage_list_idx = 0

            self.session = {
                "n_trades": 0,
                # "ohlcs": [ohcl],  # TODO: enable for plotting
                "levels": (self.a, self.b, self.c, self.d)
            }
            self.position_size_list = []
            self.profit_list = []
            return

        # self.session["ohlcs"].append(ohcl)  # TODO: enable for plotting

        assert self.state in (self.UP_TRADE, self.DOWN_TRADE)
        win = False
        lose = False
        if self.state == self.UP_TRADE:
            win_price = self.a
            entry_price = self.b
            lose_price = self.c
            result = ohcl.is_out_of_bounds(self.a, self.c)
            win_cond = Ohcl.OUT_HI
            lose_cond = Ohcl.OUT_LO
            next_trade = self.DOWN_TRADE
        else:  # self.DOWN_TRADE
            lose_price = self.b
            entry_price = self.c
            win_price = self.d
            result = ohcl.is_out_of_bounds(self.b, self.d)
            win_cond = Ohcl.OUT_LO
            lose_cond = Ohcl.OUT_HI
            next_trade = self.UP_TRADE

        if result == Ohcl.IN:
            # Nothing happens
            return

        # Trade ended
        self.session["n_trades"] += 1
        if result == win_cond:
            win = True
        if result == lose_cond:
            lose = True
        if result == Ohcl.OUT_BOTH:
            if self.optimistic:
                win = True
            else:
                lose = True

        assert not (win and lose)

        # Calculate the position size for this trade
        if self.position_size_percentage_list_idx >= len(self.position_size_percentage_list):
            position_size_percentage = 100
        else:
            position_size_percentage = self.position_size_percentage_list[self.position_size_percentage_list_idx]
        assert 0 < position_size_percentage <= 100, "position_size_percentage must be in (0, 100]"
        position_size = self.const_budget * position_size_percentage / 100
        self.position_size_list.append(position_size)
        # Subtract the amount used for this trade
        self.const_budget -= position_size
        # Increase the position size percentage for the next trade
        self.position_size_percentage_list_idx += 1

        if win:
            # Calculate the amount after the trade (profit or loss)
            position_size_after_trade = self.calculate_profit(
                entry_price=entry_price,
                exit_price=win_price,
                is_long=self.state == self.UP_TRADE,
                amount=position_size,
                commission=COMMISSION)
            # Record profit
            self.profit_list.append(position_size_after_trade - position_size)
            # Add the amount after the trade back to the budget
            self.const_budget += position_size_after_trade
            # Record session
            self.session["profit"] = self.const_budget - INITIAL_BUDGET
            self.sessions.append(self.session)
            if self.session["n_trades"] not in self.session_samples:
                self.session_samples[self.session["n_trades"]] = {
                    "position_sizes": [round(ps, 2) for ps in self.position_size_list],
                    "profits": [round(p, 2) for p in self.profit_list]
                }
            self.state = self.NO_TRADE

        if lose:
            # Calculate the amount after the trade (profit or loss)
            position_size_after_trade = self.calculate_profit(
                entry_price=entry_price,
                exit_price=lose_price,
                is_long=self.state == self.UP_TRADE,
                amount=position_size,
                commission=COMMISSION)
            # Record profit
            self.profit_list.append(position_size_after_trade - position_size)
            # Add the amount after the trade back to the budget
            self.const_budget += position_size_after_trade
            self.state = next_trade

        # if self.session["n_trades"] == 11:
        # if len(self.session["ohlcs"]) > 30:
        # if self.session["n_trades"] == 6:
        #     self.plot_session(self.session)

    def print_summary(self):
        print(f"\nChBreakoutTrader")
        for param in self.params:
            print(f"  {param}: {self.params[param]}")

        # Session samples
        print("Session samples:")
        # Read out incrementing key order from self.session_samples
        for k in sorted(self.session_samples.keys()):
            s = self.session_samples[k]
            print(f"  {k} trades:\n" 
                  f"    position_sizes: {s["position_sizes"]}\n"
                  f"    profits       : {s["profits"]}")

        # Profit distribution across sessions
        print("Profit distribution across sessions:")
        profit_per_no_trades_counter = {}
        for session in self.sessions:
            profit_per_no_trades_counter.setdefault(session["n_trades"], []).append(session.get("profit", 0))
        for n_trades in sorted(profit_per_no_trades_counter.keys()):
            profits = profit_per_no_trades_counter[n_trades]
            print(f"  {n_trades} trades: {len(profits)} sess, "
                  f"profit sum: {sum(p for p in profits):.2f} "
                  f"avg. profit: {sum(profits) / len(profits):.2f} "
                  f"({min(profits):.2f} - {max(profits):.2f})")

        # Trade count distribution across sessions
        # print("Trade count distribution across sessions:")
        # for n_trades, count in Counter((session["n_trades"] for session in self.sessions)).items():
        #     print(f"  {n_trades} trades: {count} sessions")

        # Sum profit
        sum_profit = sum(session.get("profit", 0) for session in self.sessions)
        print(f"Sum profit: {sum_profit}")
        if sum_profit > 0:
            print(f"WINNER!")

    @staticmethod
    def calculate_profit(entry_price, exit_price, is_long, amount, commission):
        amount -= amount * commission  # subtract entry commission
        shares = amount / entry_price  # number of shares bought/sold
        if is_long:
            profit = (exit_price - entry_price) * shares
        else:
            profit = (entry_price - exit_price) * shares

        amount += profit  # add profit to amount
        amount -= amount * commission  # subtract exit commission
        return amount

    @staticmethod
    def plot_session(session):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=list(range(len(session["ohlcs"]))),
                    open=[ohlc.open for ohlc in session["ohlcs"]],
                    high=[ohlc.high for ohlc in session["ohlcs"]],
                    low=[ohlc.low for ohlc in session["ohlcs"]],
                    close=[ohlc.close for ohlc in session["ohlcs"]],
                    name="OHLC"
                )
            ]
        )

        # Add four horizontal lines
        for y in (session["levels"]):
            fig.add_shape(
                type="line",
                xref="paper", x0=0, x1=1,  # span the full width of the plotting area
                yref="y", y0=y, y1=y,
                line=dict(color="royalblue", width=1.5, dash="dash")
            )
            # Optional annotation label on the right side
            fig.add_annotation(
                xref="paper", x=1.002, y=y, yref="y",
                xanchor="left", showarrow=False,
                text=f"{y:g}", font=dict(size=10, color="royalblue")
            )

        fig.update_layout(
            title="OHLC with Horizontal Levels",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            margin=dict(l=60, r=80, t=50, b=40)
        )

        fig.show()


def run_experiment(timeseries, trader):
    early_skip = True
    for row in tqdm(timeseries.df.itertuples(), total=len(timeseries.df)):
        # print(row.Index, row.Timestamp, row.Open, row.High, row.Low, row.Close, row.Volume)
        if row.Open > 10000:
            early_skip = False
        if early_skip:
            continue
        ohcl = Ohcl(row.Open, row.High, row.Low, row.Close)
        trader.update(ohcl)


def main():
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()
    log_file = open(log_filename, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    n_experiments = 1000

    ts = Timeseries("data/Bitcoin_Historical_Data/btcusd_1-min_data.csv")

    #random.seed(3)

    for _ in range(n_experiments):
        ch_breakout_trader = generate_random_ch_breakout_trader(mode="custom")
        run_experiment(ts, ch_breakout_trader)
        ch_breakout_trader.print_summary()

    # Sort results by score descending
    # results.sort(key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    main()
