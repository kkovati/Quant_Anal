from collections import Counter
import math
import pandas as pd
import plotly.graph_objects as go
import random
from tqdm import tqdm


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


class OscillationCounter:
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

        self.session = {}
        self.n_trades_in_session = 0
        self.trades = []

    def update(self, ohcl: Ohcl):
        if self.state == self.NO_TRADE:
            self.a = ohcl.close * (1 + self.win_percentage / 100)
            self.b = ohcl.close
            self.c = ohcl.close * (1 - self.lose_percentage / 100)
            self.d = ohcl.close * (1 - (self.lose_percentage + self.win_percentage) / 100)
            self.state = self.UP_TRADE
            # print(f'NO_TRADE -> UP_TRADE {self.a} {self.b} {self.c} {self.d} ')
            self.session = {
                "n_trades": 0,
                "ohlcs": [ohcl],
                "levels": (self.a, self.b, self.c, self.d)
            }
            return

        self.session["ohlcs"].append(ohcl)

        if self.state == self.UP_TRADE:
            result = ohcl.is_out_of_bounds(self.a, self.c)
            if result == Ohcl.IN:
                # Nothing happens
                return
            # Trade ended
            self.session["n_trades"] += 1
            # print(f"Trade ended {ohcl.open} {ohcl.high} {ohcl.low} {ohcl.close}")
            # print(f"No. trades: {self.session["n_trades"]}")
            if result == Ohcl.OUT_HI:
                # Win
                # print("OUT_HI Win -> NO_TRADE")
                self.trades.append(self.session["n_trades"])
                self.state = self.NO_TRADE
            if result == Ohcl.OUT_LO:
                # Lose
                # print("OUT_LO Lose -> DOWN_TRADE")
                self.state = self.DOWN_TRADE
            if result == Ohcl.OUT_BOTH:
                if self.optimistic:
                    # Win
                    # print("OUT_BOTH Win -> NO_TRADE")
                    self.trades.append(self.session["n_trades"])
                    self.state = self.NO_TRADE
                else:
                    # Lose
                    # print("OUT_BOTH Lose -> DOWN_TRADE")
                    self.state = self.DOWN_TRADE

        if self.state == self.DOWN_TRADE:
            result = ohcl.is_out_of_bounds(self.b, self.d)
            if result == Ohcl.IN:
                # Nothing happens
                return
            # Trade ended
            self.session["n_trades"] += 1
            # print(f"Trade ended {ohcl.open} {ohcl.high} {ohcl.low} {ohcl.close}")
            # print(f"No. trades: {self.session["n_trades"]}")
            if result == Ohcl.OUT_LO:
                # Win
                # print("OUT_LO Win -> NO_TRADE")
                self.trades.append(self.session["n_trades"])
                self.state = self.NO_TRADE
            if result == Ohcl.OUT_HI:
                # Lose
                # print("OUT_HI Lose -> UP_TRADE")
                self.state = self.UP_TRADE
            if result == Ohcl.OUT_BOTH:
                if self.optimistic:
                    # Win
                    # print("OUT_BOTH Win -> NO_TRADE")
                    self.trades.append(self.session["n_trades"])
                    self.state = self.NO_TRADE
                else:
                    # Lose
                    # print("OUT_BOTH Lose -> UP_TRADE")
                    self.state = self.UP_TRADE

        # if self.session["n_trades"] == 11:
        # if len(self.session["ohlcs"]) > 30:
        # if self.session["n_trades"] == 6:
        #     self.plot_session(self.session)

    def calculate_profit(self, entry_price, exit_price, is_long, amount, commission):
        shares = amount / entry_price  # number of shares bought/sold
        if is_long:
            profit = (exit_price - entry_price) * shares
        else:
            profit = (entry_price - exit_price) * shares

        # TODO this commission calc is wrong
        total_profit = profit - 2 * commission  # subtract entry and exit commissions
        return amount + total_profit

    def plot_session(self, session):
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


def random_search(param_space, experiment_fn, iterations=10):
    param_space = {
        'lr': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'dropout': [0.1, 0.3, 0.5]
    }

    results = []

    for i in range(iterations):
        # Randomly sample parameters
        params = {k: random.choice(v) for k, v in param_space.items()}
        # Run experiment
        score = experiment_fn(params)
        # Log result
        results.append((params, score))
        print(f"Run {i+1}: Params={params}, Score={score}")

    # Sort results by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    ts = Timeseries("data/Bitcoin_Historical_Data/btcusd_1-min_data.csv")
    oc = OscillationCounter("up", 1, 5)

    early_skip = True
    for row in tqdm(ts.df.itertuples(), total=len(ts.df)):
        # print(row.Index, row.Timestamp, row.Open, row.High, row.Low, row.Close, row.Volume)
        if row.Open > 10000:
            early_skip = False
        if early_skip:
            continue
        ohcl = Ohcl(row.Open, row.High, row.Low, row.Close)
        oc.update(ohcl)
        # if row.Index > 10000:
        #     break

    # print(oc.trades)
    freq = Counter(oc.trades)
    print(freq)


if __name__ == '__main__':
    main()
