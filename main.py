from collections import Counter
import math
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


class ChBreakoutTrader:
    NO_TRADE = 0
    UP_TRADE = 1
    DOWN_TRADE = 2
    IN_REGION_1 = 3
    IN_REGION_2 = 4

    def __init__(self, direction, win_percentage, lose_percentage, optimistic=True):
        assert direction in ['up', 'down']

        self.direction = direction
        self.win_percentage = win_percentage
        self.lose_percentage = lose_percentage
        self.optimistic = optimistic

        self.state = self.NO_TRADE
        self.a = self.b = self.c = self.d = None

        self.session = {}  # current session
        self.sessions = []  # all sessions

        self.const_budget = INITIAL_BUDGET
        # self.real_budget = INITIAL_BUDGET

    def update(self, ohcl: Ohcl):
        if self.state == self.NO_TRADE:
            self.a = ohcl.close * (1 + self.win_percentage / 100)
            self.b = ohcl.close
            self.c = ohcl.close * (1 - self.lose_percentage / 100)
            self.d = ohcl.close * (1 - (self.lose_percentage + self.win_percentage) / 100)
            self.state = self.UP_TRADE
            self.session = {
                "n_trades": 0,
                "ohlcs": [ohcl],
                "levels": (self.a, self.b, self.c, self.d)
            }
            self.const_budget = INITIAL_BUDGET
            return

        self.session["ohlcs"].append(ohcl)

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

        if win:
            self.const_budget = self.calculate_profit(
                entry_price=entry_price,
                exit_price=win_price,
                is_long=self.state == self.UP_TRADE,
                amount=self.const_budget,
                commission=COMMISSION)
            self.session["profit"] = self.const_budget - INITIAL_BUDGET
            self.sessions.append(self.session)
            self.state = self.NO_TRADE

        if lose:
            self.const_budget = self.calculate_profit(
                entry_price=entry_price,
                exit_price=lose_price,
                is_long=self.state == self.UP_TRADE,
                amount=self.const_budget,
                commission=COMMISSION)
            self.state = next_trade

        # if self.session["n_trades"] == 11:
        # if len(self.session["ohlcs"]) > 30:
        # if self.session["n_trades"] == 6:
        #     self.plot_session(self.session)

    def print_summary(self):
        print(f"ChBreakoutTrader(direction={self.direction}, win_percentage={self.win_percentage}, ")
        print(f"lose_percentage={self.lose_percentage}, optimistic={self.optimistic})")

        # Profit distribution across sessions
        print("Profit distribution across sessions:")
        profit_per_no_trades_counter = {}
        for session in self.sessions:
            profit_per_no_trades_counter.setdefault(session["n_trades"], []).append(session.get("profit", 0))
        for n_trades, profits in profit_per_no_trades_counter.items():
            print(f"  {n_trades} trades: {len(profits)} sess, "
                  f"profit sum: {sum(p for p in profits):.2f} "
                  f"average profit: {sum(profits) / len(profits):.2f} "
                  f"({min(profits):.2f} - {max(profits):.2f})")

        # Trade count distribution across sessions
        print("Trade count distribution across sessions:")
        for n_trades, count in Counter((session["n_trades"] for session in self.sessions)).items():
            print(f"  {n_trades} trades: {count} sessions")

        # Sum profit
        print(f"Sum profit: {sum(session.get("profit", 0) for session in self.sessions)}")

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


def log_uniform(min_val=0.1, max_val=1000):
    # Sample uniformly in log space
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    log_sample = random.uniform(log_min, log_max)
    return 10 ** log_sample


def generate_random_ch_breakout_trader():
    # Randomly sample parameters
    params = {
        'win_percentage': log_uniform(0.1, 10),
        'lose_percentage': log_uniform(0.1, 10),
        'optimistic': random.choice([True, False])
    }
    print(params)
    return ChBreakoutTrader("up", **params)


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
    n_experiments = 10

    ts = Timeseries("data/Bitcoin_Historical_Data/btcusd_1-min_data.csv")

    random.seed(3)

    for _ in range(n_experiments):
        ch_breakout_trader = generate_random_ch_breakout_trader()
        run_experiment(ts, ch_breakout_trader)
        ch_breakout_trader.print_summary()

    # Sort results by score descending
    # results.sort(key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    main()
