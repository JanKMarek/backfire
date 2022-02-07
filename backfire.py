import os
import math
from datetime import datetime, date
import pandas as pd
import numpy as np
import pandas_datareader as pdr

class Environment:
    """
      Represents the backtesting environment
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

class AlwaysOffSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOffSignal")

    def __call__(self, ohlcv):
        rv = ohlcv.apply(lambda row: False, axis=1)
        return rv

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
        return rv

class AlwaysOnSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOnSignal")

    def __call__(self, ohlcv):
        rv = ohlcv.apply(lambda row: True, axis=1)
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
        return rv

class BoucherSignal(Signal):
    def __init__(self):
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
        return rv

class OpenProtectiveStop():
    """
       Risk management signal.
    """
    def __init__(self, stop_loss=0.07, take_profit_threshold=0.2, trailing_stop_period=10):
        self.stop_loss = stop_loss
        self.take_profit_threshold = take_profit_threshold
        self.trailing_stop_period = trailing_stop_period
        self.hit_take_profit_threshold = False

    def get_out(self, label):
        self.hit_take_profit_threshold = False
        return label

    @property
    def name(self):
        return f"OpenProtectiveStop_stoploss={self.stop_loss}_take_profit_threshold={self.take_profit_threshold}_trailing_stop_period={self.trailing_stop_period}"

    def __call__(self, current_price, buy_price, trailing_stop):
        if not self.hit_take_profit_threshold and current_price > buy_price * (1.0 + self.take_profit_threshold):
            self.hit_take_profit_threshold = True

        if current_price < buy_price * (1.0 - self.stop_loss):
            return self.get_out("sl")

        if self.trailing_stop_period is None or self.trailing_stop_period == 0:
            # no trailing stop observed
            if current_price > buy_price * (1.0 + self.take_profit_threshold):
                return self.get_out("tp")
            else:
                return None
        else:
            # observe trailing stop
            if self.hit_take_profit_threshold and (current_price > buy_price * (1.0 + self.take_profit_threshold)):
                if current_price < trailing_stop:
                    return self.get_out("tp")
                else:
                    return None
            else:
                return None

class NoRiskManagement(OpenProtectiveStop):
    def __init__(self):
        super().__init__(stop_loss=1.0, take_profit_threshold=1000)

    @property
    def name(self):
        return "NoRiskManagement"

class PositionManagement():
    def __init__(self, policy="fixed_amount", initial_position=100000):
        self.policy = policy
        self.initial_position = initial_position

    @property
    def name(self):
        return f"PositionManagement_policy={self.policy}_initial_position-{self.initial_position}"

    def pos(self, cash):
        if self.policy == "fixed_amount":
            return self.initial_position
        else:
            raise ValueError(f"Position management policy {self.policy} not implemented.")


class StrategyInterface:
    """
       A strategy.
    """
    def name(self):
        pass

    def backtest(self, ticker, from_date, to_date=None):
        pass

class SignalDrivenStrategy(StrategyInterface):
    """
        The signal driven strategy simulates a trader who evaluates the market situation every evening, decides
        to enter a position or exit the existing position and then executes the necessary entry/exit transactions
        next morning at the opening prices.

        The strategy is driven by three signals - entry, exit and risk management. The strategy enters a position
        whenever the entry signal is triggered, and exits the position whenever either the exit or the
        risk management signals are triggered.

        The tree signals are evaluated at the end of the day, and so can use any of the OHLCV fields:
        1. The entry signal is mandatory and generally uses only the stock price information.
        2. The exit signal can be specified as an explicit signal (e.g., market state deteriorates or stock
            displays weakness) or as the logical negation of the entry signal (so exit whenever entry signal
            turns off). If no exit signal is specified, the strategy relies entirely on the risk management
            signal for exits.
        3. The risk management signal is specified as the Open Protective Stop and uses the position information
            as well as the stock price information.

        In deciding the trading action for the next day, the combination of the entry/risk management signal takes
            precedence over the entry signal. Only one action can take place in one day - so the strategy
            either enters or exits a position next day. The actions include 'buy', 'sl', 'pc', 'tp' 'so'.

        The trading action for the day is executed the next morning at the opening prices. The 'es' and 'xs' fields
        reflect the signal at the end of the day while the 'pos' field represents the position throughout the day.
        Whenever the position is changed, the 'memo' field will contain either 'bought-' or 'sold-' and the
        txn-no will contain the transaction number (so gradual exit is supported).

        The backtest simulates a trader who calculates signals and trading actions in the evening and
            then executes the trades at the next day's open. The backtest function executes the following
            for each ohlcv row:
            1. Calculate position management related fields (pos, sl, tp, etc). Can only use the opening
               price from the current day and any previous days' values.
            2. Calculate signal values (es) - can use all OHLCV fields.
            3. Calculate the action for the next day (buy, sl, so, tp, etc). CAn use all OHLCV fields.
    """
    def __init__(self, env, entry_signal,
                 exit_signal=AlwaysOffSignal(),
                 risk_management=OpenProtectiveStop(),
                 position_management=PositionManagement(),
                 name=None):
        self._env = env
        self.entry_signal = entry_signal
        self.exit_signal = exit_signal
        self.risk_management = risk_management
        self.position_management=position_management
        if name is None:
            self._name = (f"SDS_" +
                          f"entry_signal={self.entry_signal.name}_" +
                          f"exit_signal={self.exit_signal.name}+" +
                          f"risk_management={self.risk_management.name}_" +
                          f"position_management={self.position_management.name}")
        else:
            self._name = name

    @property
    def name(self):
        return self._name

    def _run(self, pas):
        """
            Computes position information.
              Input: dataframe with OHLCV and entry/exit signals.
              Output: adds columns 'pos', 'cash', 'memo' with shares, cash and memos
        """
        # position management fields
        pas['pos'] = 0 # number of shares
        pas['cash'] = 0
        pas['action'] = np.nan
        pas['memo'] = np.nan
        pas['buy_price'] = np.nan
        if self.risk_management.trailing_stop_period is None or self.risk_management.trailing_stop_period == 0:
            pas["trailing_stop"] = np.nan
        else:
            pas['trailing_stop'] = pas.C.shift(1).rolling(self.risk_management.trailing_stop_period,min_periods=1).mean()
        # first row:
        pas.loc[pas.index[0], 'action'] = 'buy' if pas.loc[pas.index[0], 'es'] else np.nan
        pas.loc[pas.index[0], 'cash'] = self.position_management.initial_position

        # process rows 2 and further
        for i in range(1, len(pas)-1):
            this_row = pas.index[i]
            prev_row = pas.index[i-1]

            #
            # morning actions - update position based on prev day's position and prev day's action flag
            #  updates fields pos, cash, buy_price and memo

            # sell because of flag from prev day
            if pas.loc[prev_row, 'pos'] != 0 and pas.loc[prev_row, 'action'] in ['xs', 'sl', 'pc', 'tp']:
                pas.loc[this_row, 'pos'] = 0
                pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash'] + \
                                            pas.loc[prev_row, 'pos'] * pas.loc[this_row, 'O']
                pas.loc[this_row, 'buy_price'] = np.nan
                action = pas.loc[prev_row, 'action']
                pas.loc[this_row, 'memo'] =f"sold-{self.exit_signal.name if action == 'xs' else action}" 
            # buy because of flag from prev day
            elif pas.loc[prev_row, 'pos'] == 0 and pas.loc[prev_row, 'action'] == 'buy':
                pas.loc[this_row, 'pos'] = \
                    self.position_management.pos(pas.loc[prev_row, 'cash']) / pas.loc[this_row, 'O']
                pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash'] - \
                                            pas.loc[this_row, 'pos'] * pas.loc[this_row, 'O']
                pas.loc[this_row, 'buy_price'] = pas.loc[this_row, 'O']
                pas.loc[this_row, 'memo'] = f"bought-{self.entry_signal.name}"
            # no trading actions
            else:
                pas.loc[this_row, 'pos'] = pas.loc[prev_row, 'pos']
                pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash']
                pas.loc[this_row, 'buy_price'] = pas.loc[prev_row, 'buy_price']
                pas.loc[this_row, 'memo'] = np.nan

            #
            # evening actions - recalculate stoploss, set trading action flag for next day
            #     updates field action

            risk_management_signal = self.risk_management(pas.loc[this_row, 'C'],
                                                          pas.loc[this_row, 'buy_price'],
                                                          pas.loc[this_row, 'trailing_stop'])
            # position is flat and signal triggered -> set action to buy next day
            if pas.loc[this_row, 'pos'] == 0 and pas.loc[this_row, 'es']:
                pas.loc[this_row, 'action'] = 'buy'
            # have position and exit signal triggered
            elif pas.loc[this_row, 'pos'] != 0 and pas.loc[this_row, 'xs']:
                pas.loc[this_row, 'action'] = 'xs'
            # have position and risk management signal triggered
            elif pas.loc[this_row, 'pos'] != 0 and risk_management_signal is not None:
                pas.loc[this_row, 'action'] = risk_management_signal
            # else no action flag for tomorrow
            else:
                pas.loc[this_row, 'action'] = np.nan

        # last row
        prev_row = pas.index[-2]
        this_row = pas.index[-1]
        if pas.loc[prev_row, 'pos'] != 0:
            pas.loc[this_row, 'pos'] = 0
            pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash'] + \
                                        pas.loc[prev_row, 'pos'] * pas.loc[this_row, 'H']
            pas.loc[this_row, 'memo'] = 'sold-lastday'
        else:
            pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash']

        return pas

    def _make_trades(self, ticker, positions):
        t = []
        d = positions
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

    def _make_price_and_signals(self, ohlcv, entry_signal, exit_signal):
        """
           Merge ohlcv and signal_values into one dataframe for easy visualization.
        """
        rv = pd.concat([ohlcv, entry_signal, exit_signal], axis=1)
        rv = rv[['O', 'H', 'L', 'C', 'V', 'es', 'xs']]
        rv['es'] = np.where(rv.es, rv.C, 0)
        rv['xs'] = np.where(rv['xs'], rv.C, 0)
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

        entry_signal = self.entry_signal(ohlcv)
        entry_signal.rename('es', inplace=True)
        exit_signal = self.exit_signal(ohlcv)
        exit_signal.rename('xs', inplace=True)
        price_and_signals = self._make_price_and_signals(ohlcv, entry_signal, exit_signal)

        positions = self._run(price_and_signals)

        trades = self._make_trades(ticker, positions)

        stats, trades = Evaluator().evaluate_trades(trades, self.position_management.initial_position,
                                                    from_date, to_date)

        positions.to_csv(os.path.join(self._env.out_dir, f"pos_{ticker}_{self.name}.csv"),
                               index=True, header=True, float_format='%.2f')
        stats.to_csv(os.path.join(self._env.out_dir, f"stats_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')
        trades.to_csv(os.path.join(self._env.out_dir, f"trades_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')

        return price_and_signals, positions, trades

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
        stats['avg_winning_pnl_pcnt'] = round(trades[trades.pnl >= 0].pnl_pcnt.mean(), 2)
        stats['avg_losing_pnl_pcnt'] = round(trades[trades.pnl < 0].pnl_pcnt.mean(), 2)
        stats['r'] = stats.avg_winning_pnl_pcnt / stats.avg_losing_pnl_pcnt
        stats['min_pnl_pcnt'] = round(trades.pnl_pcnt.min(), 2)
        stats['max_pnl_pcnt'] = round(trades.pnl_pcnt.max(), 2)
        stats['std_pnl_pcnt'] = round(trades.pnl_pcnt.std(), 2)
        stats['avg_hp'] = trades.hp.mean()
        stats['avg_winning_hp'] = trades[trades.pnl >= 0].hp.mean()
        stats['avg_losing_hp'] = trades[trades.pnl < 0].hp.mean()
        stats['total_pnl'] = round(trades.pnl.sum(), 2)

        rtn = 1 + (trades.pnl.sum() / initial_position)
        tim = to_date - from_date
        stats["time_span"] = tim
        stats['cagr'] = math.pow(rtn, 365 / tim.days) - 1

        stats = pd.Series(name="stats", data=stats)
        return stats, trades


