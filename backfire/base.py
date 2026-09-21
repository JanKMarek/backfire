import os
import math
from datetime import datetime, date
import pandas as pd
import numpy as np
import pandas_datareader as pdr

# Trading days in a year, used to annualize metrics computed over the daily equity curve.
_TRADING_DAYS_PER_YEAR = 252


class Environment:
    """
      Represents the backtesting environment:
        - md - directory with market data
        - out_dir - output directory
        - conf_dir - configuration directory
    """
    def __init__(self, md="./md", out_dir="./out", conf_dir="."):
        self.md = md
        self.out_dir = out_dir
        if out_dir:  # an empty out_dir means the run keeps no persistent output
            os.makedirs(out_dir, exist_ok = True)
        self.conf_dir = conf_dir

    def save(self, df, filename, **to_csv_kwargs):
        """
            Writes a dataframe into the output directory, or nowhere if the environment
            was created without one.
        :return: the path written, or None if there is no output directory
        """
        if not self.out_dir:
            return None
        path = os.path.join(self.out_dir, filename)
        df.to_csv(path, **to_csv_kwargs)
        return path

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
        rv = self._call_impl(ohlcv)

        # create signal ids
        is_start_of_true_block = rv.es & (rv.es != rv.es.shift(1).fillna(False))
        block_ids_raw = is_start_of_true_block.cumsum()
        rv['id'] = block_ids_raw.where(rv.es)

        return rv

    def _call_impl(self, ohlcv):
        """
           Returns a dataframe with date index containing dates, and at least column 'es' containing True/False
        :param ohlcv:
        :return:
        """
        # return dataframe indexed on date with at least one column 'es' containing True/False
        pass

    def bind(self, env):
        """
           Gives the signal the environment of the strategy it is part of, so that it can load
           market data other than the underlying's (e.g. an index). A signal created with its
           own environment keeps it. Combinators bind the signals they combine as well.
        """
        if self.env is None:
            self.env = env

    def evaluate_position(self, row):
        """
           The position dependent part of the signal, e.g. a profit target measured against
           the buy price of the open trade. Unlike _call_impl it cannot be computed up front
           from the prices alone, so the strategy evaluates it in the evening of every day it
           holds a position, once that day's position and buy price are known. The signal is
           on for the day if either part is on.
        :param row: the day's row of the positions table - the OHLCV fields plus 'pos',
                    'buy_price' and the signal values
        :return: True if the signal is on for the day. Most signals depend on the market
                 only, so the default is False.
        """
        return False

    def reset(self):
        """
           Forgets any state the signal keeps about the open trade (e.g. whether a trailing
           stop has been armed). The strategy calls it whenever the position is closed, for
           whatever reason. Most signals keep no such state, so the default does nothing.
        """
        pass

class BasicRiskManagement():
    """
       Protects the portfolio capital from losing trades for which the exit signal did not
       activate: the initial stop loss. Returns label "sl" when the price has fallen
       stop_loss below the buy price of the open position.

       Selling on a profit target or on a retracement are exit rules rather than risk
       management rules - see signals.TakeProfitSignal and signals.TrailingTakeProfitSignal.

        Common configuration:
          RM(stop_loss=0.07): 7% stop loss, otherwise exit on exit signal only.
    """
    def __init__(self,
                 stop_loss=None): # 0.07 means 7%
        if stop_loss is not None and not isinstance(stop_loss, (int, float)):
            raise ValueError(
                f"BasicRiskManagement.stop_loss must be a number or YAML null, "
                f"got {stop_loss!r}. Use 'null' (or '~') for 'no value' in YAML, not "
                f"'$null', which YAML parses as the string '$null'.")

        self.stop_loss = stop_loss

    @property
    def name(self):
        return f"BasicRiskManagement_stop_loss={self.stop_loss}"

    def __call__(self, this_row):
        """
           Returns None if no action necessary, or action label (sl)
        """
        current_price = this_row['C']
        buy_price = this_row['buy_price']

        if self.stop_loss is not None and (current_price < buy_price * (1.0 - self.stop_loss)):
            return "sl"

        return None

class NoRiskManagement(BasicRiskManagement):
    def __init__(self):
        super().__init__()

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
            displays weakness), as the logical negation of the entry signal (so exit whenever entry signal
            turns off) or as a rule on the open trade such as a profit target (TakeProfitSignal) or a
            trailing take profit (TrailingTakeProfitSignal), both evaluated day by day against the buy price - see
            Signal.evaluate_position. If no exit signal is specified, the strategy relies entirely on the
            risk management signal for exits.
        3. The risk management signal is the initial stop loss and uses the position information
            as well as the stock price information.

        In deciding the trading action for the next day, the combination of the exit/risk management signal takes
            precedence over the entry signal. Only one action can take place in one day - so the strategy
            either enters or exits a position next day. The actions include 'buy', 'sell' and 'sl'.
            Once a position is closed, for whatever reason, the entry signal that was on at the time is
            blocked: the strategy re-enters only on a new entry signal.

        The trading action for the day is executed the next morning at the opening prices. The 'es' and 'xs' fields
        reflect the signal at the end of the day while the 'pos' field represents the position throughout the day.
        Whenever the position is changed, the 'memo' field will contain either 'bought-' or 'sold-' and the
        txn-no will contain the transaction number (so gradual exit is supported).

        The backtest simulates a trader who calculates signals and trading actions in the evening and
            then executes the trades at the next day's open. The backtest function executes the following
            for each ohlcv row:
            1. Calculate position management related fields (pos, buy_price, etc). Can only use the opening
               price from the current day and any previous days' values.
            2. Calculate signal values (es, xs) - can use all OHLCV fields.
            3. Calculate the action for the next day (buy, sell, sl). Can use all OHLCV fields.
    """
    def __init__(self,
                 env,
                 entry_signal,
                 exit_signal=None,
                 risk_management=NoRiskManagement(),
                 position_management=PositionManagement(),
                 name=None,
                 save_signals=True):
        from .signals import AlwaysOffSignal
        self._env = env
        self.entry_signal = entry_signal
        self.exit_signal=AlwaysOffSignal() if exit_signal is None else exit_signal
        self.entry_signal.bind(env)
        self.exit_signal.bind(env)
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
        self.save_signals = save_signals

    @property
    def name(self):
        return self._name

    def _run(self, pas):
        """
            Computes position information.
              Input: dataframe with OHLCV and entry/exit signals. These are values at the end of the day.
              Output: adds columns 'pos', 'cash', 'memo' with shares, cash and memos.
                  action - action to execute the next morning: buy, sell and risk mgmt signal sl
                  pos - position (# of shares) at the end of the day, after the morning actions were executed
                  cash - dtto
                  memo - action plus signal name
                  buy_price - price at which the current position was bought

        """

        # 'action' and 'memo' hold strings, 'cash' holds fractional amounts. Seed them with the
        # dtype they end up with: pandas no longer silently upcasts a column on .loc assignment.
        pas['pos'] = 0  # in shares
        pas['cash'] = 0.0
        pas['action'] = pd.Series(np.nan, index=pas.index, dtype=object)
        pas['delta_shares'] = 0
        pas['memo'] = pd.Series(np.nan, index=pas.index, dtype=object)
        pas['buy_price'] = np.nan

        pas.loc[pas.index[0], 'cash'] = self.position_management.initial_position

        # the entry signal (id) that was on when the position was last closed; it is not
        # acted on again, the strategy waits for a new entry signal
        blocked_es_id = None

        # TODO: skip? simplifies the logic
        # first row:
        #pas.loc[pas.index[0], 'action'] = 'buy' if pas.loc[pas.index[0], 'es'] else np.nan

        # process from row 2 onwards (so that we have a previous row)
        for i in range(1, len(pas)-1):
            this_row = pas.index[i]
            prev_row = pas.index[i-1]

            bkpt_date = '2000-02-15'
            if this_row == datetime.strptime(bkpt_date, '%Y-%m-%d').date():
                pass # print('breakpoint')

            # Execution Engine:
            #   Executes morning trades based on fields action, shares and memo; updates fields position, cash,
            #       buy_price and memo. Fields action, shares and memo are set the previous night by risk management
            #       and position management.
            #
            #       delta_shares, memo = self.position_management.morning_execution(pas.loc[prev_row'], pas.loc[this_row])
            pas.loc[this_row, 'pos'] = pas.loc[prev_row, 'pos'] + pas.loc[prev_row, 'delta_shares']
            pas.loc[this_row, 'cash'] = pas.loc[prev_row, 'cash'] + (-1) * pas.loc[prev_row, 'delta_shares'] * pas.loc[this_row, 'O']
            if pas.loc[prev_row, 'delta_shares'] != 0:
                if pas.loc[prev_row, 'action'] in ['buy']:
                    t = 'bought'
                elif pas.loc[prev_row, 'action'] in ['sell', 'sl']:
                    t = 'sold'
                elif pas.loc[prev_row, 'action'] in ['add', 'reduce']:
                    t = pas.loc[prev_row, 'action'] # NB this will be ignored by make_trades
                else:
                    raise ValueError(f"This should never happen: action: {action}, delta_shares: {delta_shares}")
                pas.loc[this_row, 'memo'] = f"{t}:{pas.loc[prev_row, 'delta_shares']} shares;{pas.loc[prev_row, 'memo']}"
            if pas.loc[prev_row, 'delta_shares'] > 0 and pd.isna(pas.loc[prev_row, 'buy_price']):
                # record initial buy price, will be used by both risk and position management
                pas.loc[this_row, 'buy_price'] = pas.loc[this_row, 'O']
            elif pas.loc[this_row, 'delta_shares'] == 0 and not pd.isna(pas.loc[prev_row, 'action']):
                # we closed pos'n, clear buy_price
                pas.loc[this_row, 'buy_price'] = np.nan
            else:
                # carry over the buy_price
                pas.loc[this_row, 'buy_price'] = pas.loc[prev_row, 'buy_price']

            # Evening Actions:
            #     risk mgmt and pos mgmt set fields action, delta_shares, memo: risk mgmt takes precedence
            #     risk mgmt sets action sl
            #     pos mgmt sets actions buy, sell, add, reduce

            # risk management: sl
            risk_management_action = self.risk_management(pas.loc[this_row])
            if risk_management_action == 'sl':
                pas.loc[this_row, 'action'] = risk_management_action
                pas.loc[this_row, 'delta_shares'] = -pas.loc[this_row, 'pos']
                pas.loc[this_row, 'memo'] = risk_management_action
                blocked_es_id = pas.loc[this_row, 'es_id']
                self.exit_signal.reset()
                continue

            # the position dependent part of the exit signal (e.g. a profit target or a
            # trailing stop) can only be evaluated now that the day's position and buy price
            # are known; it is recorded in 'xs' like the market part, as the closing price on
            # the days it is on
            if (pas.loc[this_row, 'pos'] != 0
                    and self.exit_signal.evaluate_position(pas.loc[this_row])):
                pas.loc[this_row, 'xs'] = pas.loc[this_row, 'C']

            # position management: buy, add, sell, reduce
            # position is flat and signal triggered -> set action to buy next day
            if pas.loc[this_row, 'pos'] == 0 and pas.loc[this_row, 'es']:
                if pas.loc[this_row, 'es_id'] == blocked_es_id:
                    continue
                if pas.loc[this_row, 'xs']:
                    continue
                pas.loc[this_row, 'action'] = 'buy'
                pas.loc[this_row, 'delta_shares'] = math.floor(pas.loc[this_row, 'cash'] / pas.loc[this_row, 'C'])
                pas.loc[this_row, 'memo'] = self.entry_signal.name
            # have position and exit signal triggered
            elif pas.loc[this_row, 'pos'] != 0 and pas.loc[this_row, 'xs']:
                pas.loc[this_row, 'action'] = 'sell'
                pas.loc[this_row, 'delta_shares'] = -pas.loc[this_row, 'pos']
                pas.loc[this_row, 'memo'] = self.exit_signal.name
                blocked_es_id = pas.loc[this_row, 'es_id']
                self.exit_signal.reset()
            # else no action flag for tomorrow
            else:
                pass

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
        rv = d[d.memo.str.startswith("bought", na=False) | d.memo.str.startswith("sold", na=False)]
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
        # pass the columns explicitly so that a strategy which never traded still returns
        # a well formed (empty) trade list
        rv = pd.DataFrame.from_records(t, columns=['ticker', 'entry_date', 'entry_price',
                                                   'shares', 'exit_date', 'exit_price', 'memo'])
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
                positions: dataframe with columns 'pos', 'sl', 'memo', 'equity' and indexed with day dates
                   'memo' contains: 'es'/'sl'/'fday' on days either of these are triggered
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
        if self.save_signals:
            self._env.save(es, f"{self.entry_signal.name}.csv")
        es.rename(columns={'es': 'es', 'id': 'es_id'}, inplace=True)
        xs = self.exit_signal(ohlcv)
        if self.save_signals:
            self._env.save(xs, f"{self.exit_signal.name}.csv")
        xs.rename(columns={'es': 'xs', 'id': 'xs_id'}, inplace=True)
        price_and_signals = self._make_price_and_signals(ohlcv, es, xs)

        positions = self._run(price_and_signals)

        trades = self._make_trades(ticker, positions)

        stats, trades = Evaluator().evaluate_trades(trades,
                                                    positions,
                                                    self.position_management.initial_position,
                                                    from_date, to_date)

        self._env.save(positions, f"pos_{ticker}_{self.name}.csv",
                       index=True, header=True, float_format='%.2f')
        self._env.save(stats, f"stats_{ticker}_{self.name}.csv",
                       header=True, index=True, float_format='%.2f')
        self._env.save(trades, f"trades_{ticker}_{self.name}.csv",
                       header=True, index=True, float_format='%.2f')

        return price_and_signals, positions, trades, stats

class Evaluator:
    """
        Evaluates a set of trades. Trades are contained in a dataframe with columns:
               entry_date, shares, entry_price, exit_date, exit_price

        Trade level metrics are derived from the trades, the risk adjusted ones (drawdowns,
        time in drawdown, Sharpe, Calmar) from the daily mark to market balance held in
        'positions', so that they account for the open position and not only for closed trades.
    """

    def evaluate_trades(self, trades, positions, initial_position, from_date, to_date):
        if trades.empty:
            # a strategy may legitimately not trade at all over a short trading period;
            # seed the derived columns so that the metrics below evaluate to NA rather than fail
            for column in ['pnl', 'pnl_pcnt', 'hp']:
                trades[column] = pd.Series(dtype=float)
        else:
            trades['pnl'] = trades.shares * (trades.exit_price - trades.entry_price)
            trades['pnl_pcnt'] = trades.pnl / (trades.shares * trades.entry_price)
            trades['hp'] = trades.exit_date - trades.entry_date
            trades['hp'] = trades.hp.apply(lambda x: x.days)

        stats = {}
        stats['no_trades'] = len(trades)
        stats['no_winning_trades'] = len(trades[trades.pnl >= 0])
        stats['no_losing_trades'] = len(trades[trades.pnl < 0])
        stats['EV'] = round(trades.pnl_pcnt.mean(), 2)
        stats['win/loss ratio'] = (round(stats["no_winning_trades"] / stats["no_trades"], 2)
                                   if stats["no_trades"] else np.nan)
        stats['avg_winning_pnl_pcnt'] = round(trades[trades.pnl >= 0].pnl_pcnt.mean(), 2)
        stats['avg_losing_pnl_pcnt'] = round(trades[trades.pnl < 0].pnl_pcnt.mean(), 2)
        # R is undefined when the strategy had no losing trade to size the winners against
        stats['r'] = (round(stats['avg_winning_pnl_pcnt'] / abs(stats['avg_losing_pnl_pcnt']), 2)
                      if stats['avg_losing_pnl_pcnt'] else np.nan)
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
        # keep the unrounded CAGR, the Calmar ratio below divides by a small number
        cagr = math.pow(rtn, 365 / tim.days) - 1
        stats['cagr'] = round(cagr, 2)

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
        max_dd = positions.unrealized_dd_pcnt.max()
        stats['max_dd_pcnt_unrealized'] = round(max_dd, 2)

        # Max time in drawdown: the longest stretch, in calendar days, that the mark to market
        # balance spends below a previous high water mark - from the day the mark was set to
        # the last day before the balance recovers it, or to the last trading day if it never
        # does. A strategy that keeps making new highs spends no time in drawdown.
        days = pd.Series(pd.to_datetime(positions.index), index=positions.index)
        at_high_water_mark = positions.balance >= positions.unrealized_CumMax
        stats['max_time_in_dd'] = (days - days.where(at_high_water_mark).ffill()).max().days

        # Sharpe ratio over the daily mark to market returns, annualized and measured against
        # a zero risk free rate. Undefined for a balance that never moves (e.g. no trades).
        daily_rtn = positions.balance.pct_change().dropna()
        volatility = daily_rtn.std()
        stats['sharpe'] = (round(math.sqrt(_TRADING_DAYS_PER_YEAR) * daily_rtn.mean() / volatility, 2)
                           if volatility > 0 else np.nan)

        # Calmar ratio: CAGR per unit of the deepest drawdown endured to earn it. Undefined
        # for a strategy that never drew down.
        stats['calmar'] = round(cagr / max_dd, 2) if max_dd > 0 else np.nan

        stats = pd.Series(name="stats", data=stats)
        return stats, trades




