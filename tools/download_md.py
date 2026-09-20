"""
Downloads daily OHLCV data from Yahoo Finance and appends it to the market data store.

The market data store is a folder of `<TICKER>.csv` files with the columns
`Date,Open,High,Low,Close,Volume` (see `Environment.load_ohlcv`). Prices are
split-adjusted but not dividend-adjusted, which is what yfinance returns with
`auto_adjust=False`.

Only rows strictly newer than the last date already in the file are appended, so the
script is safe to re-run.

Usage:
    uv run python tools/download_md.py QQQ --start 2025-05-12
    uv run python tools/download_md.py QQQ SPY --start 2025-05-12 --end 2025-09-15
    uv run python tools/download_md.py QQQ --start 2025-05-12 --dry-run
"""

import argparse
import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


def download(ticker, start, end):
    """
        Downloads daily OHLCV data for a single ticker.
    :param ticker: ticker to download
    :param start: first day to download (inclusive), as a date
    :param end: last day to download (inclusive), as a date
    :return: Dataframe with OHLCV columns indexed with day dates
    """
    # yfinance treats `end` as exclusive.
    df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     interval='1d', auto_adjust=False, progress=False, actions=False)
    if df.empty:
        return df

    # yfinance returns (field, ticker) MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=-1)

    df = df[COLUMNS]
    df.index = pd.to_datetime(df.index).date
    df.index.name = 'Date'
    return df


def append(df, path, dry_run=False):
    """
        Appends the rows of `df` that post-date the existing file's last date.
    :param df: freshly downloaded OHLCV data
    :param path: path of the `<TICKER>.csv` file to update
    :param dry_run: if True, report what would be written without writing it
    :return: the Dataframe of appended rows (possibly empty)
    """
    last_date = None
    if os.path.exists(path):
        existing = pd.read_csv(path, usecols=['Date'])
        if not existing.empty:
            last_date = pd.to_datetime(existing.Date).dt.date.max()
            df = df[df.index > last_date]

    if df.empty:
        print(f"{path}: up to date (last date {last_date})")
        return df

    df = df.round({'Open': 4, 'High': 4, 'Low': 4, 'Close': 4})
    df['Volume'] = df.Volume.astype('int64')

    if dry_run:
        print(f"{path}: would append {len(df)} rows ({df.index.min()} .. {df.index.max()})")
        print(df.to_string())
        return df

    header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        df.to_csv(f, header=header)
    print(f"{path}: appended {len(df)} rows ({df.index.min()} .. {df.index.max()})")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tickers', nargs='+', help='tickers to download, e.g. QQQ SPY')
    parser.add_argument('--start', required=True, help='first day to download, YYYY-MM-DD')
    parser.add_argument('--end', default=date.today().isoformat(),
                        help='last day to download, YYYY-MM-DD (default: today)')
    parser.add_argument('--md', default='md', help='market data folder (default: md)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the rows that would be appended without writing them')
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for ticker in args.tickers:
        df = download(ticker, start, end)
        if df.empty:
            print(f"{ticker}: no data returned for {start} .. {end}")
            continue
        append(df, os.path.join(args.md, f'{ticker}.csv'), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
