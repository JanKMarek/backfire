import os
import math
from datetime import datetime, date
import pandas as pd
import numpy as np
import pandas_datareader as pdr

class Environment:
    """
      Represents the backtesting environment:
        - md - directory with market data
        - out_dir - output directory
        - conf_dir - configuration directory
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

    """
    def __init__(self, name, env=None):
        self._name = name
        self.env = env

    @property
    def name(self):
        return self._name

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

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
        :return: Dataframe with columns 'es' and 'id' (and possibly intermediate columns) and indexed with day dates
        """
        pass

class AlwaysOnSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOnSignal")

    def __call__(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = True
        rv['id'] = 1
        return rv

class AlwaysOffSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOffSignal")

    def __call__(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        rv['id'] = 1
        return rv


class BasicRiskManagement():
    """
       Implements three common risk/profit management techniques:
         i/ Stop loss. Returns label "sl" if stop loss reached.
         ii/ Take Profit. Returns label "tp" if take profit is reached and not in
                  trailing stop mode.
         iii/ Trailing Stop. Returns label "ts" if price has fallen below the trailing stop
                  moving average AFTER take profit has been reached.


        Common configurations:
          RM(stop_loss=0.07, take_profit=None, trailing_stop_period=None):
              7% stop loss, no take profit or trailing stop. Exit on exit signal only.
          RM(stop_loss=0.07, take_profit=0.2, trailing_stop_period=10):
              7% stop loss, trail with 10d MA after 20% take profit reached
    """
    def __init__(self,
                 stop_loss=None, # 0.07 means 7%
                 take_profit=None, # 0.2 is 20%
                 trailing_stop_period=None): # in days
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop_period = trailing_stop_period

        if take_profit is None and trailing_stop_period is not None:
            raise ValueError("Need take profit to use trailing_stop_period.")

        self.hit_take_profit = False
        self.last_blocked_signal = None

    @property
    def name(self):
        return f"OpenProtectiveStop_stoploss={self.stop_loss}_take_profit={self.take_profit}_trailing_stop_period={self.trailing_stop_period}"

    def is_blocked(self, signal_id):
        return True if signal_id == self.last_blocked_signal else False

    def clear_take_profit_hit(self):
        self.hit_take_profit = False

    def __call__(self, current_price, buy_price, trailing_stop, entry_signal_id):
        """
           Returns None if no action necessary, or action label (sl, tp, ts)
        """
        if self.stop_loss is not None and (current_price < buy_price * (1.0 - self.stop_loss)):
            return self.get_out("sl", entry_signal_id)

        if self.trailing_stop_period is None: # not in trailing stop mode
            if self.take_profit is not None and current_price > buy_price * (1.0 + self.take_profit):
                return self.get_out("tp", entry_signal_id)
        else: # in trailing stop mode
            if not self.hit_take_profit and current_price > buy_price * (1.0 + self.take_profit):
                self.hit_take_profit = True
            if self.hit_take_profit and current_price < trailing_stop:
                return self.get_out("ts", entry_signal_id)

        return None

    def get_out(self, label, signal_id):
        self.clear_take_profit_hit()
        self.last_blocked_signal = signal_id
        return label

class NoRiskManagement(BasicRiskManagement):
    def __init__(self):
        super().__init__(stop_loss=1.0, take_profit=1000, trailing_stop_period=None)

    @property
    def name(self):
        return "NoRiskManagement"

class PositionManagement():
    def __init__(self, policy="fixed_amount", initial_position=100000, fraction=1.0):
        self.policy = policy
        self.initial_position = initial_position
        self.fraction = fraction

    @property
    def name(self):
        return f"PositionManagement_policy={self.policy}_initial_position-{self.initial_position}"

    def pos(self, cash):
        if self.policy == "fixed_amount":
            return self.initial_position
        elif self.policy == "fixed_fraction":
            return cash * self.fraction
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
                 risk_management=NoRiskManagement(),
                 position_management=PositionManagement(),
                 name=None):
        self._env = env
        self.entry_signal = entry_signal
        self.exit_signal=AlwaysOffSignal() if exit_signal is None else exit_signal
        self.risk_management = NoRiskManagement() if risk_management is None else risk_management
        self.position_management = PositionManagement() if position_management is None else position_management
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
              Input: dataframe with OHLCV and entry/exit signals. These are values at the end of the day.
              Output: adds columns 'pos', 'cash', 'memo' with shares, cash and memos.
                  action - action to execute the next morning: es, xs and risk mgmt signals sl, pc, tp
                  pos - position (# of shares) at the end of the day, after the morning actions were executed
                  cash - dtto
                  memo - action plus signal name
                  buy_price - price at which the current position was bought

        """
        # position management fields
        pas['pos'] = 0 # number of shares
        pas['cash'] = 0
        pas['action'] = np.nan
        pas['memo'] = np.nan
        pas['buy_price'] = np.nan
        pas['trailing_stop'] = np.nan
        if self.risk_management.trailing_stop_period is not None:
            pas['trailing_stop'] = pas.C.shift(1).rolling(self.risk_management.trailing_stop_period,min_periods=1).mean()

        # first row:
        pas.loc[pas.index[0], 'action'] = 'buy' if pas.loc[pas.index[0], 'es'] else np.nan
        pas.loc[pas.index[0], 'cash'] = self.position_management.initial_position

        # process rows 2 and further
        for i in range(1, len(pas)-1):
            this_row = pas.index[i]
            prev_row = pas.index[i-1]

            bkpt_date = '2000-02-15'
            if this_row == datetime.strptime(bkpt_date, '%Y-%m-%d').date():
                print('breakpoint')

            #
            # morning actions - update position based on prev day's position and prev day's action flag
            #  updates fields pos, cash, buy_price and memo

            # sell because of flag from prev day
            if pas.loc[prev_row, 'pos'] != 0 and pas.loc[prev_row, 'action'] in ['xs', 'sl', 'pc', 'tp', 'ts']:
                pas.loc[this_row, 'pos'] = 0
                pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash'] + \
                                            pas.loc[prev_row, 'pos'] * pas.loc[this_row, 'O']
                pas.loc[this_row, 'buy_price'] = np.nan
                action = pas.loc[prev_row, 'action']
                pas.loc[this_row, 'memo'] =f"sold-{self.exit_signal.name if action == 'xs' else action}"
            # buy because of flag from prev day
            elif (pas.loc[prev_row, 'pos'] == 0 and
                  pas.loc[prev_row, 'action'] == 'buy' and
                  (not self.risk_management.is_blocked(pas.loc[prev_row, 'es_id']))):
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

            risk_management_signal = self.risk_management(
                current_price=pas.loc[this_row, 'C'],
                buy_price=pas.loc[this_row, 'buy_price'],
                trailing_stop=pas.loc[this_row, 'trailing_stop'],
                entry_signal_id=pas.loc[this_row, 'es_id'])
            # position is flat and signal triggered -> set action to buy next day
            if pas.loc[this_row, 'pos'] == 0 and pas.loc[this_row, 'es']:
                pas.loc[this_row, 'action'] = 'buy'
            # have position and exit signal triggered
            elif pas.loc[this_row, 'pos'] != 0 and pas.loc[this_row, 'xs']:
                pas.loc[this_row, 'action'] = 'xs'
                self.risk_management.clear_take_profit_hit()
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
        entry_signal.rename(columns={'id': 'es_id'}) # NB don't need to worry about exit signal
        rv = pd.concat([ohlcv, entry_signal, exit_signal], axis=1)
        rv = rv[['O', 'H', 'L', 'C', 'V', 'es', 'es_id', 'xs']]
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

        es = self.entry_signal(ohlcv)
        es.rename(columns={'es': 'es', 'id': 'es_id'}, inplace=True)
        xs = self.exit_signal(ohlcv)
        xs.rename(columns={'es': 'xs', 'id': 'xs_id'}, inplace=True)
        price_and_signals = self._make_price_and_signals(ohlcv, es, xs)

        positions = self._run(price_and_signals)

        trades = self._make_trades(ticker, positions)

        stats, trades = Evaluator().evaluate_trades(trades,
                                                    positions,
                                                    self.position_management.initial_position,
                                                    from_date, to_date)

        positions.to_csv(os.path.join(self._env.out_dir, f"pos_{ticker}_{self.name}.csv"),
                               index=True, header=True, float_format='%.2f')
        stats.to_csv(os.path.join(self._env.out_dir, f"stats_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')
        trades.to_csv(os.path.join(self._env.out_dir, f"trades_{ticker}_{self.name}.csv"),
                     header=True, index=True, float_format='%.2f')

        return price_and_signals, positions, trades, stats

class Evaluator:
    """
        Evaluates a set of trades. Trades are contained in a dataframe with columns:
               entry_date, shares, entry_price, exit_date, exit_price
    """

    def evaluate_trades(self, trades, positions, initial_position, from_date, to_date):
        trades['pnl'] = trades.shares * (trades.exit_price - trades.entry_price)
        trades['pnl_pcnt'] = trades.pnl / (trades.shares * trades.entry_price)
        trades['hp'] = trades.exit_date - trades.entry_date
        trades['hp'] = trades.hp.apply(lambda x: x.days)

        stats = {}
        stats['no_trades'] = len(trades)
        stats['no_winning_trades'] = len(trades[trades.pnl >= 0])
        stats['no_losing_trades'] = len(trades[trades.pnl < 0])
        stats['EV'] = round(trades.pnl_pcnt.mean(), 2)
        stats['win/loss ratio'] = round(stats["no_winning_trades"] / stats["no_trades"], 2)
        stats['avg_winning_pnl_pcnt'] = round(trades[trades.pnl >= 0].pnl_pcnt.mean(), 2)
        stats['avg_losing_pnl_pcnt'] = round(trades[trades.pnl < 0].pnl_pcnt.mean(), 2)
        stats['r'] = round(stats['avg_winning_pnl_pcnt'] / abs(stats['avg_losing_pnl_pcnt']), 2)
        stats['min_pnl_pcnt'] = round(trades.pnl_pcnt.min(), 2)
        stats['max_pnl_pcnt'] = round(trades.pnl_pcnt.max(), 2)
#        stats['std_pnl_pcnt'] = round(trades.pnl_pcnt.std(), 2)
        stats['avg_hp'] = round(trades.hp.mean(), 2)
        stats['avg_winning_hp'] = round(trades[trades.pnl >= 0].hp.mean(), 2)
        stats['avg_losing_hp'] = round(trades[trades.pnl < 0].hp.mean(), 2)
        stats['total_pnl'] = round(trades.pnl.sum(), 2)
        stats['positive_pnl'] = round(trades[trades.pnl >= 0].pnl.sum(), 2)
        stats['negative_pnl'] = round(trades[trades.pnl < 0].pnl.sum(), 2)

        rtn = 1 + (trades.pnl.sum() / initial_position)
        stats['rtn'] = round(rtn, 2)
        tim = to_date - from_date
        stats["time_span"] = tim
        stats['cagr'] = round(math.pow(rtn, 365 / tim.days) - 1, 2)

        # Max Drawdown - Realized
        equity = trades.pnl.cumsum().to_frame()
        equity.columns = ['cum_pnl']
        equity['equity'] = initial_position + equity.cum_pnl
        equity['CumMax'] = equity.equity.cummax()
        equity['dd'] = equity.CumMax - equity.equity
        equity['dd_pcnt'] = equity.dd / equity.CumMax
        stats['max_dd_pcnt_realized'] = round(equity.dd_pcnt.max(), 2)

        # Max Drawdown - Unrealized
        positions['balance'] = positions.cash + positions.pos * positions.C
        positions['unrealized_CumMax'] = positions.balance.cummax()
        positions['unrealized_dd'] = positions.unrealized_CumMax - positions.balance
        positions['unrealized_dd_pcnt'] = positions.unrealized_dd / positions.unrealized_CumMax
        stats['max_dd_pcnt_unrealized'] = round(positions.unrealized_dd_pcnt.max(), 2)

        stats = pd.Series(name="stats", data=stats)
        return stats, trades

# if __name__ == "__main__":
#
#     short_MA = 50 # days
#     long_MA = 200 # days
#     stop_loss = None
#     take_profit = 0.6
#     trailing_stop_period = 200
#
#     ticker = 'QQQ' # '^IXIC_1990'
#     name = f"Index_{str(short_MA)}dMAvs{str(long_MA)}dMA"
#     from_date = '2000-01-01'
#     out_dir = f"../out/{name}"
#     md = "../md"
#
#     env = Environment(md=md, out_dir=out_dir)
#
#
#     entry_signal = ShortMAAboveLongMA(short_MA=short_MA, long_MA=long_MA)
#     exit_signal = ShortMABelowLongMA(short_MA=short_MA, long_MA=long_MA)
#     #risk_management = NoRiskManagement()
#     risk_management = BasicRiskManagement(
#         stop_loss=stop_loss,
#         take_profit=take_profit,
#         trailing_stop_period=trailing_stop_period)
#     position_management = PositionManagement(initial_position=100000, policy="fixed_fraction", fraction=1.0)
#
#     s = SignalDrivenStrategy(
#         env=env,
#         entry_signal=entry_signal,
#         exit_signal=exit_signal,
#         risk_management=risk_management,
#         position_management=position_management,
#         name=name)
#     s.backtest(ticker=ticker, from_date=from_date)



