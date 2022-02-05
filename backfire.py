import os
import math
from datetime import datetime, date
import pandas as pd
import numpy as np
import pandas_datareader as pdr

class Environment:
    """
      Represents the backteting environment
    """
    def __init__(self, md=".", out_dir=".", conf_dir="."):
        self.md = md
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok = True)
        self.conf_dir = conf_dir

    def load_ohlcv(self, ticker, from_date, to_date=None):
        """
            Loads OHLCV data from market data store.
        :param ticker: ticker to load
        :param from_date:
        :param to_date: If None, then until the most recent available data
        :return: Dataframe with OHLCV columns and indexed with day dates
        """

        if isinstance(self.md, pd.DataFrame):
            return self.md

        if not os.path.exists(os.path.join(self.md, f'{ticker}.csv')):
            print(f"Downloading stock price data for {ticker}! ")
            df = pdr.DataReader(ticker, 'yahoo', start='2010-01-01', end=date.today().isoformat())
            df.to_csv(os.path.join(self.md, f'{ticker}.csv'))

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
       Represents a signal (e.g., entry signal).

           The signal is evaluated at the end of the day using the C(losing) price of the day.
           The signal value is then set True for that day.
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def ma(self, ohlcv, per):
        if per == 200:
            min_periods = 30
        elif per == 150:
            min_periods = 20
        elif per == 50:
            min_periods = 10
        elif per == 20:
            min_periods = 5
        elif per == 10:
            min_periods = 3
        else:
            min_periods = round(per/4)
        return ohlcv.C.shift(1).rolling(per, min_periods=min_periods).mean()

    def __call__(self, ohlcv):
        """
           Generates signal values.
        :param ohlcv: Dataframe with columns 'O', 'H', 'L', 'C', 'V' and indexed with day dates.
        :return: Dataframe with column 'es' (and any intermediate columns) and indexed with day dates
        """
        pass

class RandomEntrySignal(Signal):
    """
       Random entry. Useful for testing.
    """
    def __init__(self, prob=0.2):
        super().__init__(f"randomentry{prob:.1f}")
        """
           Creates a random entry signal every 1/prob days
        """
        self.prob = prob

    def __call__(self, ohlcv):
        high = math.ceil(1 / self.prob)
        rv = ohlcv.apply(lambda row: np.random.randint(low=0, high=high) == 0, axis=1)
        rv.rename('es', inplace=True)
        return rv

class AlwaysOnSignal(Signal):
    def __init__(self):
        super().__init__("BuyAndHold")

    def __call__(self, ohlcv):
        rv = ohlcv.apply(lambda row: True, axis=1)
        rv.rename('es', inplace=True)
        return rv

class FlatBaseBreakoutSignal(Signal):
    """
        Breakout from a flat base.
    """
    def __init__(self, length=4*5, height=.1):
        self.length = length
        self.height = height
        super().__init__(f"flatbase{length}_{height:.2f}")

    def __call__(self, ohlcv):

        # flat base
        ohlcv['shifted'] = ohlcv.C.shift(1)
        ohlcv['low'] = ohlcv.C.shift(1).rolling(self.length).min()
        ohlcv['high'] = ohlcv.C.shift(1).rolling(self.length).max()
        ohlcv['fb'] = ((ohlcv.high - ohlcv.low)) / ohlcv.low < self.height

        # breakout
        ohlcv['higher_vol'] = ohlcv.V > ohlcv.V.shift(1).rolling(self.length).mean()
        ohlcv['close_above'] = ohlcv.C > ohlcv.high
        ohlcv['breakout'] = ohlcv.higher_vol & ohlcv.close_above

        ohlcv['flat_base_breakout'] = ohlcv.fb & ohlcv.breakout

        rv = ohlcv.flat_base_breakout
        rv.rename('es', inplace=True)
        return rv

class BoucherSignal(Signal):
    def __init__(selfself):
        super().__init__(f"Boucher1")

    def __call__(self, ohlcv):

        # flat base
        ohlcv['shifted'] = ohlcv.C.shift(1)
        ohlcv['low'] = ohlcv.C.shift(1).rolling(self.length).min()
        ohlcv['high'] = ohlcv.C.shift(1).rolling(self.length).max()
        ohlcv['fb'] = ((ohlcv.high - ohlcv.low)) / ohlcv.low < self.height

        # thrust
        ohlcv['range'] = ohlcv.H - ohlcv.L
        ohlcv['avg_range'] = ohlcv['range'].shift(1).rolling(self.length).mean()
        ohlcv['range_greater'] = ohlcv['range'] > ohlcv.avg_range
        ohlcv['vol_higher'] = ohlcv.V > ohlcv.V.shift(1)
        ohlcv['close_upper_third'] = ohlcv.C - ohlcv.L > (ohlcv.H - ohlcv.L) * 0.66
        ohlcv['thrust'] = ohlcv.range_greater & ohlcv.vol_higher & ohlcv.close_upper_third

        # lap
        ohlcv['lap'] = ohlcv.C >= ohlcv.shift(1).C & ohlcv.C <= ohlcv.shift(1).H

        # gap
        ohlcv['gap'] = ohlcv.L >= ohlcv.shift(1).H

        #



        ohlcv['flat_base_breakout'] = ohlcv.fb & ohlcv.thrust_breakout

        rv = ohlcv.flat_base_breakout
        rv.rename('es', inplace=True)
        return rv

class Stage2Signal(Signal):
    def __init__(self, rise200d_months=1, rise200d_step=0.01):
        self.rise200d_months = rise200d_months
        self.rise200d_step = rise200d_step
        super().__init__(f"Stage2_rise200dlength={rise200d_months}_rise200dstep={rise200d_step}")

    def __call__(self, ohlcv):

        ohlcv['ma50'] = self.ma(ohlcv, 50)
        ohlcv['ma150'] = self.ma(ohlcv, 150)
        ohlcv['ma200'] = self.ma(ohlcv, 200)

        ohlcv['criterion_1'] = (ohlcv.C > ohlcv.ma150) & (ohlcv.C > ohlcv.ma200)
        ohlcv['criterion_2'] = ohlcv.ma150 > ohlcv.ma200
        ohlcv['criterion_3'] = ((ohlcv.ma200 - ohlcv.ma200.shift(self.rise200d_months*20))
                                / ohlcv.ma200.shift(self.rise200d_months*20) > self.rise200d_step)
        ohlcv['criterion_4'] = (ohlcv.ma50 > ohlcv.ma150) & (ohlcv.ma50 > ohlcv.ma200)
        ohlcv['criterion_5'] = ohlcv.C > ohlcv.ma50
        ohlcv['criterion_6'] = ohlcv.C > (ohlcv.C.rolling(250, min_periods=50).min() * 1.3)
        ohlcv['criterion_7'] = ohlcv.C > (ohlcv.C.rolling(250, min_periods=50).max() * 0.75)

        ohlcv['stage2'] = ohlcv.criterion_1 & ohlcv.criterion_2 & \
            ohlcv.criterion_3 & ohlcv.criterion_4 & ohlcv.criterion_5 & \
            ohlcv.criterion_6 & ohlcv.criterion_7

        ohlcv.to_csv(os.path.join('out_test', 'stage2.csv'))

        rv = ohlcv.stage2
        rv.rename('es', inplace=True)
        return rv

class Strategy:
    """
       A strategy.
    """
    def name(self):
        pass

    def backtest(self, ticker, from_date, to_date=None):
        pass

class SignalDrivenStrategy:
    """
        A signal driven strategy.

           The strategy simulates execution of trades in the morning. If the signal was triggered
        the previous day, the strategy will take position by buying at the opening price. If stoploss
        or takeprofit are triggered (i.e., previous day clos is below/above sl/tp_, the position
        is flattened by selling at the open as well. The position is updated for that day accordingly.
    """
    def __init__(self, env, signal, sl=0.07, tp=0.2, pos_management='fixed', initial_pos=100000):
        self._env = env
        self._signal = signal
        self._sl = sl
        self._tp = tp
        self._pos_management = pos_management
        self._pos = initial_pos

    @property
    def name(self):
        return f"SDS_signal={self._signal.name}_sl={self._sl}_tp={self._tp}_pos={self._pos_management}"

    @property
    def sl(self):
        return self._sl

    @property
    def tp(self):
        return self._tp

    @property
    def pos(self):
        return self._pos

    def _run(self, ohlcv, signal_values):
        """

        """
        d = pd.concat([ohlcv, signal_values], axis=1)
        for colname in ['pos', 'sl', 'tp', 'memo']:
            d[colname] = np.nan

        # first row:
        d.loc[d.index[0], 'pos'] = 0
        d.loc[d.index[0], 'sl'] = np.nan
        d.loc[d.index[0], 'tp'] = np.nan
        d.loc[d.index[0], 'memo'] = 'buy' if d.loc[d.index[0], 'es'] else np.nan

        # process all other rows except for the last one
        for i in range(1, len(d)-1):
            this_row = d.index[i]
            prev_row = d.index[i-1]

            # sell because of flag from prev day
            if d.loc[prev_row, 'pos'] != 0 and d.loc[prev_row, 'memo'] in ['sl', 'tp', 'so']:
                d.loc[this_row, 'pos'] = 0
                d.loc[this_row, 'sl'] = np.nan
                d.loc[this_row, 'tp'] = np.nan
                d.loc[this_row, 'memo'] = f"sold-{d.loc[prev_row, 'memo']}"
            # buy becaue of flag from prev day
            elif d.loc[prev_row, 'pos'] == 0 and d.loc[prev_row, 'memo'] == 'buy':
                d.loc[this_row, 'pos'] = self.pos / d.loc[this_row, 'O']
                d.loc[this_row, 'sl'] = round((1-self.sl) * d.loc[this_row, 'O'])
                d.loc[this_row, 'tp'] = round((1+self.tp) * d.loc[this_row, 'O'])
                d.loc[this_row, 'memo'] = f"bought-{self._signal.name}"
            # previous position is flat and signal triggered -> set action to buy next day
            elif d.loc[prev_row, 'pos'] == 0 and d.loc[this_row, 'es']:
                d.loc[this_row, 'pos'] = 0
                d.loc[this_row, 'sl'] = np.nan
                d.loc[this_row, 'tp'] = np.nan
                d.loc[this_row, 'memo'] = 'buy'
            # have position and stoploss triggered  - set action to sell next day
            elif d.loc[prev_row, 'pos'] != 0 and d.loc[this_row, 'C'] <= d.loc[prev_row, 'sl']:
                d.loc[this_row, 'pos'] = d.loc[prev_row, 'pos']
                d.loc[this_row, 'sl'] = d.loc[prev_row, 'sl']
                d.loc[this_row, 'tp'] = d.loc[prev_row, 'tp']
                d.loc[this_row, 'memo'] = 'sl'
            # have position and takeprofit triggered - set action to sell next day
            elif d.loc[prev_row, 'pos'] != 0 and d.loc[this_row, 'C'] >= d.loc[prev_row, 'tp']:
                d.loc[this_row, 'pos'] = d.loc[prev_row, 'pos']
                d.loc[this_row, 'sl'] = d.loc[prev_row, 'sl']
                d.loc[this_row, 'tp'] = d.loc[prev_row, 'tp']
                d.loc[this_row, 'memo'] = 'tp'
            # have position and signal went off - set action to sell next day
            elif d.loc[prev_row, 'pos'] != 0 and not d.loc[this_row, 'es']:
                d.loc[this_row, 'pos'] = d.loc[prev_row, 'pos']
                d.loc[this_row, 'sl'] = d.loc[prev_row, 'sl']
                d.loc[this_row, 'tp'] = d.loc[prev_row, 'tp']
                d.loc[this_row, 'memo'] = 'so'
            # else just carry over values
            else:
                d.loc[this_row, 'pos'] = d.loc[prev_row, 'pos']
                d.loc[this_row, 'sl'] = d.loc[prev_row, 'sl']
                d.loc[this_row, 'tp'] = d.loc[prev_row, 'tp']
                d.loc[this_row, 'memo'] = np.nan

        # last row
        prev_row = d.index[-2]
        this_row = d.index[-1]
        d.loc[this_row, 'pos'] = d.loc[prev_row, 'pos']
        d.loc[this_row, 'sl'] = d.loc[prev_row, 'sl']
        d.loc[this_row, 'tp'] = d.loc[prev_row, 'tp']
        d.loc[this_row, 'memo'] = 'sold-lastday' if d.loc[prev_row, 'pos'] != 0 else np.nan

        return d[['pos', 'sl', 'tp', 'memo']]

    def _make_trades(self, ticker, ohlcv, positions):
        t = []
        d = pd.concat([ohlcv, positions], axis=1)
#        rv = d[~d.memo.isnull()]
        rv = d[d.memo.str.startswith("bought") | d.memo.str.startswith("sold")]
        for i in range(len(rv)):
            ix = rv.index[i]
            if rv.loc[ix, 'memo'].startswith('sold'):
                t.append({'ticker': ticker,
                          'entry_date': rv.index[i-1],
                          'entry_price': rv.loc[rv.index[i-1], 'O'],
                          'shares': rv.loc[rv.index[i-1], 'pos'],
                          'exit_date': ix,
                          'exit_price': rv.loc[ix, 'O'],
                          'memo': rv.loc[rv.index[i-1], 'memo'] + ' / ' + rv.loc[ix, 'memo']})
        rv = pd.DataFrame.from_records(t)
        return rv

    def _make_price_and_signal(self, ohlcv, signal_values):
        """
           Merge ohlcv and signal_values into one dataframe for easy visualization.
        """
        rv = pd.concat([ohlcv, signal_values], axis=1)
        rv = rv[['O', 'H', 'L', 'C', 'V', 'es']]
        rv['es'] = np.where(rv.es, rv.C, 0)
        return rv

    def backtest(self, ticker, from_date, to_date=None):
        """
            Runs the backtest and generates the following output dataframes:
                price_and_signal: dataframe with columns 'O', 'H', 'L', 'C', 'V', 'es' and indexed with day dates
                positions: dataframe with columns 'pos', 'sl', 'tp', 'memo', 'equity' and indexed with day dates
                   'memo' contains: 'es'/'sl'/'tp'/'fday' on days either of these are triggered
                trades: dataframe with columns 'entry_date', 'shares', 'entry_price', 'entry_memo',
                    'exit_date', 'exit_price', 'exit_memo' and indexed with day dates
        :return:
            - Dataframe with trades
            - Dataframe with prices, signals and equity curve
        """
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
        if to_date is None:
            to_date = date.today()
        if isinstance(to_date, str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d").date()

        ohlcv = self._env.load_ohlcv(ticker, from_date, to_date)

        signal_values = self._signal(ohlcv)
        price_and_signal = self._make_price_and_signal(ohlcv, signal_values)

        positions = self._run(ohlcv, signal_values)

        trades = self._make_trades(ticker, ohlcv, positions)

        stats, trades = Evaluator().evaluate_trades(trades, self._pos, from_date, to_date)

        positions = pd.concat([price_and_signal, positions], axis=1)

        positions.to_csv(os.path.join(self._env.out_dir, f"pos_{ticker}_{self.name}.csv"),
                               index=True, header=True, float_format='%.2f')
        stats.to_csv(os.path.join(self._env.out_dir, f"stats_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')
        trades.to_csv(os.path.join(self._env.out_dir, f"trades_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')

        return price_and_signal, positions, trades

class BuyAndHoldStrategy(SignalDrivenStrategy):
    def __init__(self, env):
        super().__init__(env,AlwaysOnSignal(),1.0,1000)

class Evaluator:
    """
        Evaluates a set of trades. Trades are contained in a dataframe with columns:
               entry_date, shares, entry_price, exit_date, exit_price
    """

    def evaluate_trades(self, trades, initial_position, from_date, to_date):
        trades['pnl'] = trades.shares * (trades.exit_price - trades.entry_price)
        trades['pnl_pcnt'] = trades.pnl / (trades.shares * trades.entry_price)
        trades['hp'] = trades.exit_date - trades.entry_date

        stats = {}
        stats['no_trades'] = len(trades)
        stats['no_winning_trades'] = len(trades[trades.pnl >= 0])
        stats['no_losing_trades'] = len(trades[trades.pnl < 0])
        stats['win/loss ratio'] = round(stats["no_winning_trades"] / stats["no_trades"], 2)
        stats['avg_pnl_pcnt'] = round(trades.pnl_pcnt.mean(), 2)
        stats['min_pnl_pcnt'] = round(trades.pnl_pcnt.min(), 2)
        stats['max_pnl_pcnt'] = round(trades.pnl_pcnt.max(), 2)
        stats['std_pnl_pcnt'] = round(trades.pnl_pcnt.std(), 2)
        stats['avg_winning_pnl_pcnt'] = round(trades[trades.pnl >= 0].pnl_pcnt.mean(), 2)
        stats['avg_losing_pnl_pcnt'] = round(trades[trades.pnl < 0].pnl_pcnt.mean(), 2)
        stats['avg_hp'] = trades.hp.mean()
        stats['avg_winning_hp'] = trades[trades.pnl >= 0].hp.mean()
        stats['avg_losing_hp'] = trades[trades.pnl < 0].hp.mean()
        stats['total_pnl'] = round(trades.pnl.sum(), 2)

        rtn = 1 + (trades.pnl.sum() / initial_position)
        tim = to_date - from_date
        stats['cagr'] = math.pow(rtn, 365 / tim.days) - 1

        stats = pd.Series(name="stats", data=stats)
        return stats, trades


