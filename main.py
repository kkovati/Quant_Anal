import pandas as pd
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

        self.n_trades_in_session = 0
        self.trades = []

    def update(self, ohcl: Ohcl):
        if self.state == self.NO_TRADE:
            self.a = ohcl.close * (1 + self.win_percentage / 100)
            self.b = ohcl.close
            self.c = ohcl.close * (1 - self.lose_percentage / 100)
            self.d = ohcl.close * (1 - (self.lose_percentage + self.win_percentage) / 100)
            self.n_trades_in_session = 0
            self.state = self.UP_TRADE

        if self.state == self.UP_TRADE:
            result = ohcl.is_out_of_bounds(self.a, self.c)
            if result == Ohcl.IN:
                # Nothing happens
                return
            # Trade ended
            self.n_trades_in_session += 1
            if result == Ohcl.OUT_HI:
                # Win
                self.trades.append(self.n_trades_in_session)
                self.state = self.NO_TRADE
            if result == Ohcl.OUT_LO:
                # Lose
                self.state = self.DOWN_TRADE
            if result == Ohcl.OUT_BOTH:
                if self.optimistic:
                    # Win
                    self.trades.append(self.n_trades_in_session)
                    self.state = self.NO_TRADE
                else:
                    # Lose
                    self.state = self.DOWN_TRADE

        if self.state == self.DOWN_TRADE:
            result = ohcl.is_out_of_bounds(self.b, self.d)
            if result == Ohcl.IN:
                # Nothing happens
                return
            # Trade ended
            self.n_trades_in_session += 1
            if result == Ohcl.OUT_LO:
                # Win
                self.trades.append(self.n_trades_in_session)
                self.state = self.NO_TRADE
            if result == Ohcl.OUT_HI:
                # Lose
                self.state = self.DOWN_TRADE
            if result == Ohcl.OUT_BOTH:
                if self.optimistic:
                    # Win
                    self.trades.append(self.n_trades_in_session)
                    self.state = self.NO_TRADE
                else:
                    # Lose
                    self.state = self.DOWN_TRADE


def main():
    ts = Timeseries("data/Bitcoin_Historical_Data/btcusd_1-min_data.csv")
    oc = OscillationCounter("up", 1, 3)

    for row in tqdm(ts.df.itertuples(), total=len(ts.df)):
        # print(row.Index, row.Timestamp, row.Open, row.High, row.Low, row.Close, row.Volume)
        ohcl = Ohcl(row.Open, row.High, row.Low, row.Close)
        oc.update(ohcl)
        if row.Index > 10000:
            break

    print(oc.trades)


if __name__ == '__main__':
    main()
