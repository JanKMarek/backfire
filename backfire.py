import os
import math
from datetime import datetime
import pandas as pd
import numpy as np

class Environment:
    """
      Represents the backteting environment
    """
    def __init__(self, md=".", out_dir=".", conf_dir="."):
        self.md = md
        self.out_dir = out_dir
        self.conf_dir = conf_dir

    def load_ohlcv(self, ticker, from_date, to_date=None):
        """
            Loads OHLCV data from market data store.
        :param ticker: ticker to load
        :param from_date:
        :param to_date: If None, then until the most recent available data
        :return: Dataframe with OHLCV columns and indexed with day dates
        """
        t = pd.read_csv(os.path.join(self.md, f'{ticker}.csv'), header=0)
        t = t[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        t.columns = ['Date', 'O', 'H', 'L', 'C', 'V']
        t['Date'] = pd.to_datetime(t.Date).dt.date
        t.set_index(keys=['Date'], drop=True, inplace=True)
        t = t.apply(pd.to_numeric)
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
        if isinstance(to_date, str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d").date()
        t = t[from_date:] if to_date is None else t[from_date:to_date]
        return t


class Signal:
    """
       Represents a signal (e.g., entry signal)
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, ohlcv):
        """
           Generates signal values.
        :param ohlcv: Dataframe with columns 'O', 'H', 'L', 'C', 'V' and indexed with day dates.
        :return: Dataframe with column 'es' and indexed with day dates
        """
        pass

class RandomEntrySignal(Signal):
    """
       Random entry. Useful for testing.
    """
    def __init__(self, name, prob=0.2):
        super().__init__(name)
        """
           Creates a random entry signal every 1/prob days
        """
        self.prob = prob

    def __call__(self, ohlcv):
        high = math.ceil(1 / self.prob)
        rv = ohlcv.apply(lambda row: np.random.randint(low=0, high=high) == 0, axis=1)
        rv.rename('es', inplace=True)
        return rv

class Strategy:
    """
        Represents a strategy.
    """
    def __init__(self, sl=0.07, tp=0.2, pos=100):
        self._sl = sl
        self._tp = tp
        self._pos = pos

    @property
    def sl(self):
        return self._sl

    @property
    def tp(self):
        return self._tp

    @property
    def pos(self):
        return self._pos

class Runner:
    """
        Backtests a signal and a strategy on a ticker.
    """
    def __init__(self, env, signal, strategy):
        self.env = env
        self.signal = signal
        self.strategy = strategy

    def _run(self, d):
        for colname in ['pos', 'sl', 'tp', 'memo']:
            d[colname] = np.nan

        for i in range(len(d)):
            ix = d.index[i]

            POS_FLAT = 0
            is_pos_flat = d.loc[d.index[i-1], 'pos'] == POS_FLAT if i > 0 else True
            if is_pos_flat and d.loc[ix, 'es']:
                d.loc[ix, 'pos'] = self.strategy.pos
                d.loc[ix, 'sl'] = (1 - self.strategy.sl) * d.loc[ix, 'H']
                d.loc[ix, 'tp'] = (1+self.strategy.tp)*d.loc[ix, 'H']
                d.loc[ix, 'memo'] = self.signal.name
            elif not is_pos_flat and d.loc[ix, 'C'] < d.loc[d.index[i-1], 'sl']:
                d.loc[ix, 'pos'] = POS_FLAT
                d.loc[ix, 'sl'] = np.nan
                d.loc[ix, 'tp'] = np.nan
                d.loc[ix, 'memo'] = 'sl'
            elif not is_pos_flat and d.loc[ix, 'C'] > d.loc[d.index[i-1], 'tp']:
                d.loc[ix, 'pos'] = POS_FLAT
                d.loc[ix, 'sl'] = np.nan
                d.loc[ix, 'tp'] = np.nan
                d.loc[ix, 'memo'] = 'tp'
            else:
                d.loc[ix, 'pos'] = d.loc[d.index[i-1], 'pos'] if i > 0 else POS_FLAT
                d.loc[ix, 'sl'] = d.loc[d.index[i-1], 'sl'] if i > 0 else np.nan
                d.loc[ix, 'tp'] = d.loc[d.index[i-1], 'tp'] if i > 0 else np.nan
            ix = d.index[-1]
            d.loc[ix, 'pos'] = POS_FLAT
            d.loc[ix, 'sl'] = np.nan
            d.loc[ix, 'tp'] = np.nan
            d.loc[ix, 'memo'] = 'tp'

        return d

    def _make_trades(self, ticker, ohlcv):
        t = []
        rv = ohlcv[~ohlcv['memo'].isnull()]
        for i in range(len(rv)):
            ix = rv.index[i]
            if rv.loc[ix, 'memo'] in ['tp', 'sl']:
                t.append({'ticker': ticker,
                          'entry_date': rv.index[i-1],
                          'entry_price': rv.loc[rv.index[i-1], 'H'],
                          'shares': rv.loc[rv.index[i-1], 'pos'],
                          'exit_date': ix,
                          'exit_price': rv.loc[ix, 'H'],
                          'memo': rv.loc[rv.index[i-1], 'memo'] + ' / ' + rv.loc[ix, 'memo']})
        return pd.DataFrame.from_records(t)

    def _make_price_and_signal(self, d):
        t = d[['C', 'es']]
        on_value = t.C.max()
        t[self.signal.name] = np.where(t.es, on_value, 0)
        return t[['C', self.signal.name]]

    def backtest(self, ticker, from_date, to_date=None):
        """
            Runs the backtest.
        :return:
            - Dataframe with trades
            - Dataframe with prices, signals and equity curve
        """
        ohlcv = self.env.load_ohlcv(ticker, from_date, to_date)
        signal_values = self.signal(ohlcv)
        d = pd.concat([ohlcv, signal_values], axis=1)
        d = self._run(d)
        price_and_signal = self._make_price_and_signal(d)
        trades = self._make_trades(ticker, d)
        return trades, price_and_signal

class Evaluator:

    def evaluate_trades(self, trades):
        trades['pnl'] = trades.shares * (trades.exit_price - trades.entry_price)
        trades['pnl_pcnt'] = trades.pnl / (trades.shares * trades.entry_price)
        trades['hp'] = trades.exit_date - trades.entry_date

        stats = {}
        stats['no_trades'] = len(trades)
        stats['no_winning_trades'] = len(trades[trades.pnl >= 0])
        stats['no_losing_trades'] = len(trades[trades.pnl < 0])
        stats['total_pnl'] = round(trades.pnl.sum(), 2)
        stats['avg_pnl_pcnt'] = round(trades.pnl_pcnt.mean(), 2)
        stats['min_pnl_pcnt'] = round(trades.pnl_pcnt.min(), 2)
        stats['max_pnl_pcnt'] = round(trades.pnl_pcnt.max(), 2)
        stats['std_pnl_pcnt'] = round(trades.pnl_pcnt.std(), 2)
        stats['avg_winning_pnl_pcnt'] = round(trades[trades.pnl >= 0].pnl_pcnt.mean(), 2)
        stats['avg_losing_pnl_pcnt'] = round(trades[trades.pnl < 0].pnl_pcnt.mean(), 2)
        stats['avg_hp'] = trades.hp.mean()
        stats['avg_winning_hp'] = trades[trades.pnl >= 0].hp.mean()
        stats['avg_losing_hp'] = trades[trades.pnl < 0].hp.mean()
        stats = pd.Series(data=stats)

        return stats, trades


