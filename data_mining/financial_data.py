import os
import pandas as pd
import yfinance as yf

# Cached files land in  data/cache/
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cache')

# Accept human-friendly interval names in addition to yfinance style strings
_INTERVAL_MAP = {
    'minute':  '1m',
    '1min':    '1m',
    '5min':    '5m',
    '15min':   '15m',
    '30min':   '30m',
    'hourly':  '1h',
    'hour':    '1h',
    'daily':   '1d',
    'day':     '1d',
    'weekly':  '1wk',
    'week':    '1wk',
    'monthly': '1mo',
    'month':   '1mo',
}


def get_financial_data(ticker: str, days: int, interval: str) -> pd.DataFrame:
    """Download (or load from cache) OHLCV data from Yahoo Finance.

    Args:
        ticker:   Yahoo Finance ticker symbol, e.g. 'BTC-USD' or 'AAPL'.
        days:     Look-back window in days, e.g. 60.
        interval: Bar size.  Accepts yfinance strings ('1h', '1d', …) or
                  plain English ('hourly', 'daily', …).

    Returns:
        pandas DataFrame with columns Open / High / Low / Close / Volume,
        indexed by datetime.
    """
    interval = _INTERVAL_MAP.get(interval.lower(), interval)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe_ticker = ticker.replace('-', '_').replace('/', '_')
    filename    = f"{safe_ticker}_{days}d_{interval}.csv"
    filepath    = os.path.join(_CACHE_DIR, filename)

    if os.path.exists(filepath):
        print(f"[cache] Loading from {filepath}")
        return _read_cached_csv(filepath)

    print(f"[download] {ticker}  |  period={days}d  |  interval={interval}")
    df = yf.download(ticker, period=f"{days}d", interval=interval,
                     progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the symbol and that the interval is supported "
                         "for the requested period.")

    # yfinance returns MultiIndex columns like (Price, Ticker) even for a
    # single symbol -- flatten to plain 'Open'/'High'/... so the CSV
    # round-trips cleanly and downstream code can rely on simple column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = 'Datetime'

    df.to_csv(filepath)
    print(f"[saved]  {filepath}  ({len(df)} rows)")
    return df


def _read_cached_csv(filepath: str) -> pd.DataFrame:
    """Read a cached OHLCV CSV.

    Transparently repairs cache files that were written by older versions
    of this module, back when ``yf.download`` returned MultiIndex columns
    and ``df.to_csv`` wrote them out as 3 header rows, e.g.::

        Price,Close,High,Low,Open,Volume
        Ticker,BTC-USD,BTC-USD,BTC-USD,BTC-USD,BTC-USD
        Datetime,,,,,
        2026-05-09 00:00:00+00:00,...

    Reading that shape with ``index_col=0, parse_dates=True`` pulls the
    literal strings ``'Ticker'``/``'Datetime'`` into the index and blows up
    once real date-arithmetic is attempted on it.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline()

    if first_line.startswith('Price,'):
        # Malformed legacy cache: 3 header rows, real data starts at row 4.
        columns = first_line.strip().split(',')[1:]
        df = pd.read_csv(filepath, skiprows=3, index_col=0, header=None)
        df.columns = columns
        df.index = pd.to_datetime(df.index, utc=True, format='ISO8601')
        df.index.name = 'Datetime'

        # Re-save in the clean format so future loads are simple/fast.
        df.to_csv(filepath)
        print(f"[repaired] {filepath}")
        return df

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


# ---------------------------------------------------------------------------

def main():
    df = get_financial_data(ticker='BTC-USD', days=60, interval='hourly')

    print(df.tail())
    print(f"\nShape : {df.shape}")
    print(f"Range : {df.index[0]}  →  {df.index[-1]}")


if __name__ == '__main__':
    main()
