import os
import math
import numpy as np
import pandas as pd

from .base import Signal

class AlwaysOnSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOnSignal")

    def _call_impl(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = True
        return rv

class AlwaysOffSignal(Signal):
    def __init__(self):
        super().__init__("AlwaysOffSignal")

    def _call_impl(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        return rv

class ReverseSignal(Signal):
    """
        Logical negation of a market signal. Only the market part of the signal is reversed;
        a position dependent signal such as TakeProfitSignal cannot be reversed.
    """
    def __init__(self, signal):
        super().__init__(f"ReverseSignal_{signal.name}")
        self.signal = signal

    def _call_impl(self, ohlcv):
        rv = self.signal(ohlcv)
        rv['es'] = ~rv.es
        return rv

    def bind(self, env):
        super().bind(env)
        self.signal.bind(env)

class OrSignal(Signal):
    """
        Logical OR of any number of signals: on whenever at least one of the signals is on,
        off otherwise. A signal value that is missing (NaN, e.g. during a warm up period)
        counts as off. The position dependent part of the signals (see
        Signal.evaluate_position) is combined the same way, so a market exit signal, a
        TakeProfitSignal and a TrailingTakeProfitSignal can be OR-ed into one exit signal.

        The returned dataframe keeps every constituent's value in a column named after the
        constituent, next to the combined 'es'.
    """
    def __init__(self, *signals):
        if not signals:
            raise ValueError("OrSignal needs at least one signal.")
        for s in signals:
            if not isinstance(s, Signal):
                raise ValueError(f"OrSignal arguments must be signals, got {s!r}.")
        super().__init__("OrSignal_" + "_or_".join(s.name for s in signals))
        self.signals = signals

    def _call_impl(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        for s in self.signals:
            value = s(ohlcv)['es'].reindex(rv.index).fillna(False).astype(bool)
            rv[s.name] = value
            rv['es'] = rv['es'] | value
        return rv

    def evaluate_position(self, row):
        # evaluate every signal: a stateful one (e.g. a trailing stop) must see every day
        return any([s.evaluate_position(row) for s in self.signals])

    def reset(self):
        for s in self.signals:
            s.reset()

    def bind(self, env):
        super().bind(env)
        for s in self.signals:
            s.bind(env)

class TakeProfitSignal(Signal):
    """
        Sell into strength: on when the open trade has gained more than the threshold over
        its buy price (0.25 is 25%). Used as an exit signal, the strategy closes the position
        and, as after any exit, does not re-enter until a new entry signal is triggered.

        The signal depends on the open position rather than on the market alone, so it is
        off on every day up front and is evaluated day by day during the backtest (see
        Signal.evaluate_position).
    """
    def __init__(self, threshold): # 0.25 is 25%
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"TakeProfitSignal.threshold must be a positive number, "
                             f"e.g. 0.25 for 25%, got {threshold!r}.")
        super().__init__(f"TakeProfit_{threshold}")
        self.threshold = threshold

    def _call_impl(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        return rv

    def evaluate_position(self, row):
        if row['pos'] == 0 or pd.isna(row['buy_price']):
            return False
        return bool(row['C'] > row['buy_price'] * (1.0 + self.threshold))

class TrailingTakeProfitSignal(Signal):
    """
        Sell into weakness: once the open trade has gained more than the threshold over its
        buy price (0.2 is 20%) the trailing take profit is armed, and from then on the signal
        is on whenever the price closes below the 'period' day moving average of the previous
        closes. Used as an exit signal, the strategy closes the position and, as after any
        exit, does not re-enter until a new entry signal is triggered.

        The signal depends on the open position, so it is off on every day up front and is
        evaluated day by day during the backtest (see Signal.evaluate_position). The moving
        average is computed up front and kept in the 'trailing_take_profit' column of the
        signal values; whether the stop is armed is remembered per trade and forgotten when
        the position is closed (see Signal.reset).
    """
    def __init__(self, threshold, period): # 0.2 is 20%, period in days
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"TrailingTakeProfitSignal.threshold must be a positive number, "
                             f"e.g. 0.2 for 20%, got {threshold!r}.")
        if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
            raise ValueError(f"TrailingTakeProfitSignal.period must be a positive number of days, "
                             f"got {period!r}.")
        super().__init__(f"TrailingTakeProfit_{threshold}_{period}dMA")
        self.threshold = threshold
        self.period = period
        self.armed = False
        self.trailing_take_profit = None

    def _call_impl(self, ohlcv):
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        rv['trailing_take_profit'] = ohlcv.C.shift(1).rolling(self.period, min_periods=1).mean()
        self.trailing_take_profit = rv['trailing_take_profit']
        return rv

    def evaluate_position(self, row):
        if self.trailing_take_profit is None:
            raise RuntimeError(f"{self.name}: the signal values must be computed from the "
                               f"prices before the position dependent part is evaluated.")
        if row['pos'] == 0 or pd.isna(row['buy_price']):
            return False
        if not self.armed and row['C'] > row['buy_price'] * (1.0 + self.threshold):
            self.armed = True
        return self.armed and bool(row['C'] < self.trailing_take_profit[row.name])

    def reset(self):
        self.armed = False

class RetracementSignal(Signal):
    """
        Trailing stop: on when the price closes more than the retracement (0.1 is 10%) below
        the most recent high, the highest intraday high since the position was opened. Used
        as an exit signal, the strategy closes the position and, as after any exit, does not
        re-enter until a new entry signal is triggered.

        The signal depends on the open position, so it is off on every day up front and is
        evaluated day by day during the backtest (see Signal.evaluate_position). The high is
        remembered per trade and forgotten when the position is closed (see Signal.reset).
    """
    def __init__(self, retracement): # 0.1 is 10%
        if (isinstance(retracement, bool) or not isinstance(retracement, (int, float))
                or not 0 < retracement < 1):
            raise ValueError(f"RetracementSignal.retracement must be a number between 0 and 1, "
                             f"e.g. 0.1 for 10%, got {retracement!r}.")
        super().__init__(f"Retracement_{retracement}")
        self.retracement = retracement
        self.high = None

    def _call_impl(self, ohlcv):
        self.reset()
        rv = pd.DataFrame(index=ohlcv.index)
        rv['es'] = False
        return rv

    def evaluate_position(self, row):
        if row['pos'] == 0 or pd.isna(row['buy_price']):
            return False
        self.high = row['H'] if self.high is None else max(self.high, row['H'])
        return bool(row['C'] < self.high * (1.0 - self.retracement))

    def reset(self):
        self.high = None

class ShortMAAboveLongMA(Signal):
    def __init__(self, short_MA, long_MA):
        super().__init__(f"{str(short_MA)}dMAAbove{str(long_MA)}MA")
        self.short_MA = short_MA
        self.long_MA = long_MA

    def _call_impl(self, in_ohlcv):
        ohlcv = in_ohlcv.copy()
        ohlcv['ma' + str(self.long_MA)] = self.ma(ohlcv, self.long_MA)
        ohlcv['ma' + str(self.short_MA)] = self.ma(ohlcv, self.short_MA)
        ohlcv['es'] = ohlcv['ma' + str(self.short_MA)] >= ohlcv['ma' + str(self.long_MA)]
        return ohlcv['es'].to_frame()

class ShortMABelowLongMA(Signal):
    def __init__(self, short_MA, long_MA):
        super().__init__(f"{str(short_MA)}dMABelow{str(long_MA)}MA")
        self.short_MA = short_MA
        self.long_MA = long_MA

    def _call_impl(self, in_ohlcv):
        ohlcv = in_ohlcv.copy()
        ohlcv['ma' + str(self.long_MA)] = self.ma(ohlcv, self.long_MA)
        ohlcv['ma' + str(self.short_MA)] = self.ma(ohlcv, self.short_MA)
        ohlcv['es'] = ohlcv['ma' + str(self.short_MA)] < ohlcv['ma' + str(self.long_MA)]
        return ohlcv['es'].to_frame()

class MWPowerTrend(Signal):
    def __init__(self, short_MA=21, long_MA=50, short_above_long=5, price_above_short=10):
        super().__init__(f"PowerTrend_{str(short_MA)}dEMAVS{str(long_MA)}dSMA")
        self.short_MA = short_MA
        self.long_MA = long_MA
        self.short_above_long = short_above_long
        self.price_above_short = price_above_short

    def _call_impl(self, in_ohlcv):
        in_ohlcv = in_ohlcv.copy()

        in_ohlcv['EMA21'] = in_ohlcv['C'].ewm(span=self.short_MA, adjust=False).mean()
        in_ohlcv['SMA50'] = in_ohlcv['C'].rolling(window=self.long_MA, min_periods=1).mean()

        in_ohlcv['EMA_above_SMA'] = in_ohlcv['EMA21'] > in_ohlcv['SMA50']
        in_ohlcv['EMA_above_SMA_5d'] = \
            in_ohlcv['EMA_above_SMA'].rolling(window=self.short_above_long).sum() == self.short_above_long

        in_ohlcv['Price_above_EMA21'] = in_ohlcv['C'] > in_ohlcv['EMA21']
        in_ohlcv['Price_above_EMA21_10d'] = \
            in_ohlcv['Price_above_EMA21'].rolling(window=self.price_above_short).sum() == self.price_above_short

        in_ohlcv['PowerTrend_Trigger'] = (in_ohlcv['EMA_above_SMA_5d']) & (in_ohlcv['Price_above_EMA21_10d'])
        in_ohlcv['PowerTrend'] = False
        power_trend_active = False
        for i in range(len(in_ohlcv)):
            if power_trend_active:
                # Check if 21d EMA crossed below 50d SMA
                if not in_ohlcv['EMA_above_SMA'].iloc[i]:
                    power_trend_active = False
                else:
                    in_ohlcv.loc[in_ohlcv.index[i], 'PowerTrend'] = True
            elif in_ohlcv['PowerTrend_Trigger'].iloc[i]:
                power_trend_active = True
                in_ohlcv.loc[in_ohlcv.index[i], 'PowerTrend'] = True

        in_ohlcv.rename(columns={'PowerTrend': 'es'}, inplace=True)
        return in_ohlcv['es'].to_frame()

class ConsecutiveHigherHighsLows(Signal):
    def __init__(self, consecutive_days=3):
        super().__init__(f"{consecutive_days}ConsecutiveHigherHighsLows")
        self.consecutive_days = consecutive_days

    def _call_impl(self, in_ohlcv):
        ohlcv = in_ohlcv.copy()

        ohlcv['higher_high'] = ohlcv['H'] > ohlcv['H'].shift(1)
        ohlcv['higher_low'] = ohlcv['L'] > ohlcv['L'].shift(1)
        ohlcv['consecutive_condition'] = ohlcv['higher_high'] & ohlcv['higher_low']
        ohlcv['es'] = ohlcv['consecutive_condition'].rolling(window=self.consecutive_days).sum() == self.consecutive_days

        # Fill NaN values resulting from rolling operation with False
        # The first (consecutive_days - 1) rows will be NaN
        ohlcv['es'] = ohlcv['es'].fillna(False)

        return ohlcv['es'].to_frame()



class _FTDMachine:
    """
        The Follow Through Day state machine of FTDSignal, one day at a time. See
        docs/SIGNALS.md, which this implements rule by rule; the states are the constants on
        FTDSignal and the comments below name the spec paragraph each transition comes from.

        Two peaks are tracked, with different lifetimes:
          peak      the highest high since the most recent Follow Through Day that has not
                    failed, or since the start of the data. It is never reset by a Follow
                    Through Day - a failed one does not start a new peak window - so it is
                    still the original correction's peak when an FTD fails.
          ftd_peak  the highest high since the confirmed Follow Through Day. The new correction
                    that re-arms the signal is measured against this one, not against peak,
                    which by construction sits well above the rally the FTD confirmed.
    """
    def __init__(self, signal, idx):
        self.s = signal
        self.H = idx.H.to_numpy()
        self.L = idx.L.to_numpy()
        self.C = idx.C.to_numpy()
        self.V = idx.V.to_numpy()
        self.dates = idx.index
        # a day makes a new low when its low is the lowest of the day0_window days ending on
        # it; the window is NaN until it is full, so no day 0 is found during the warmup
        self.window_low = idx.L.rolling(signal.day0_window,
                                        min_periods=signal.day0_window).min().to_numpy()
        self.state = FTDSignal.WATCHING
        self.peak, self.peak_i = self.H[0], 0
        self.ftd_peak = self.ftd_peak_i = self.ftd_rally_low = self.ftd_i = None
        self.ftd_rally_day = None
        self.day0_i = self.rally_low = self.rally_day = None

    # --- the tests of docs/SIGNALS.md ------------------------------------------------------
    def _is_day0(self, i, peak, peak_i):
        """ A new day0_window low deep enough below a peak old enough (SIGNALS.md 'Day 0'). """
        return (not math.isnan(self.window_low[i]) and self.L[i] <= self.window_low[i]
                and self.L[i] <= peak * (1.0 - self.s.min_decline)
                and i - peak_i >= self.s.min_peak_age_days)

    def _is_ftd(self, i):
        """ Day ftd_min_days..ftd_max_days, the gain and volume above the previous day's. """
        return (self.s.ftd_min_days <= self.rally_day <= self.s.ftd_max_days
                and self.C[i] >= self.C[i - 1] * (1.0 + self.s.ftd_min_gain)
                and self.V[i] > self.V[i - 1])

    # --- bookkeeping -----------------------------------------------------------------------
    def _open_day0(self, i):
        self.day0_i, self.rally_low, self.rally_day = i, self.L[i], None
        self.state = FTDSignal.DAY0

    def _end_attempt(self):
        self.day0_i = self.rally_low = self.rally_day = None

    def _end_uptrend(self):
        self.ftd_peak = self.ftd_peak_i = self.ftd_rally_low = None

    # --- one day ---------------------------------------------------------------------------
    def step(self, i):
        """ Processes the day; returns True if it is a Follow Through Day. """
        if self.state == FTDSignal.UPTREND and self.H[i] > self.ftd_peak:
            self.ftd_peak, self.ftd_peak_i = self.H[i], i
        if self.H[i] > self.peak:
            self.peak, self.peak_i = self.H[i], i

        if self.state == FTDSignal.RALLY:
            self.rally_day += 1
            if self.C[i] < self.rally_low:
                # undercut: a close below the rally low ends the attempt. The undercutting day
                # may itself be the next day 0, so it falls through to the day 0 search below.
                self._end_attempt()
                self.state = FTDSignal.WATCHING
            elif self._is_ftd(i):
                # the Follow Through Day confirms the attempt; it wins over the timeout on the
                # last day of the window
                self.ftd_i, self.ftd_rally_low = i, self.rally_low
                self.ftd_rally_day = self.rally_day
                self.ftd_peak, self.ftd_peak_i = self.H[i], i
                self._end_attempt()
                self.state = FTDSignal.UPTREND
                return True
            elif self.rally_day >= self.s.ftd_max_days:
                # timeout: no Follow Through Day by the end of the window. Like the undercut,
                # the day falls through to the day 0 search.
                self._end_attempt()
                self.state = FTDSignal.WATCHING

        elif self.state == FTDSignal.UPTREND:
            if self.C[i] < self.ftd_rally_low:
                # failed Follow Through Day: the original correction is still in force, so the
                # peak is kept and the day falls through to the day 0 search
                self._end_uptrend()
                self.state = FTDSignal.WATCHING
            elif self._is_day0(i, self.ftd_peak, self.ftd_peak_i):
                # a new correction, measured from the highest high since the Follow Through Day
                self.peak, self.peak_i = self.ftd_peak, self.ftd_peak_i
                self._end_uptrend()
                self._open_day0(i)
                return False

        elif self.state == FTDSignal.DAY0:
            if self.L[i] < self.rally_low:
                # until day 1 arrives a lower low becomes the new day 0, even if it closes up -
                # so this day cannot also be day 1
                self.day0_i, self.rally_low = i, self.L[i]
                return False
            if self.C[i] > self.C[i - 1]:
                # day 1, the First Day of the Rally; the rally low is fixed from here on
                self.rally_day = 1
                self.state = FTDSignal.RALLY
            return False

        if self.state == FTDSignal.WATCHING and self._is_day0(i, self.peak, self.peak_i):
            self._open_day0(i)
        return False

    def record(self, i):
        """ The day's row of diagnostics - see FTDSignal._call_impl. """
        uptrend = self.state == FTDSignal.UPTREND
        peak, peak_i = ((self.ftd_peak, self.ftd_peak_i) if uptrend
                        else (self.peak, self.peak_i))
        # on the Follow Through Day itself the attempt is already closed, but its rally low and
        # day number are what the day is about, so they are the ones reported
        rally_low = self.ftd_rally_low if uptrend else self.rally_low
        rally_day = self.ftd_rally_day if uptrend and self.ftd_i == i else self.rally_day
        return {'index_close': self.C[i],
                'state': self.state,
                'peak': peak,
                'peak_date': self.dates[peak_i],
                'decline_pct': self.L[i] / peak - 1.0,
                'day0_date': None if self.day0_i is None else self.dates[self.day0_i],
                'rally_low': rally_low,
                'rally_day': rally_day,
                'ftd_date': None if self.ftd_i is None else self.dates[self.ftd_i]}


class FTDSignal(Signal):
    """
        Follow Through Day: the index turnaround that says a correction is over and a new
        uptrend has begun. The signal is on for one day only - the Follow Through Day itself,
        at most once per rally attempt. The strategy enters on it and leaves again on its exit
        signal; this signal does not decide when the uptrend ends.

        Every day is processed by a state machine that starts WATCHING; docs/SIGNALS.md is the
        specification and _FTDMachine implements it rule by rule.

          WATCHING: no rally attempt is under way. A day whose low is the lowest of the last
            day0_window days, at least min_decline below the peak, with the peak at least
            min_peak_age_days earlier, is day 0 - the rally low - and starts a DAY0.
          DAY0: waiting for day 1. A day that makes a lower low becomes the new day 0, even if
            it closes up; otherwise the first day that closes higher than the previous day is
            day 1, the First Day of the Rally, and starts a RALLY. The rally low is fixed then.
          RALLY: the day after day 1 is day 2 and so on, counted in trading days. The first
            rule that matches wins:
              - the index closes below the rally low: WATCHING (the attempt was undercut). An
                intraday dip below the rally low with a close at or above it does not count.
              - day ftd_min_days..ftd_max_days, a gain of at least ftd_min_gain over the
                previous close and volume above the previous day's: this is the Follow Through
                Day, the signal is on for the day and the state becomes UPTREND.
              - day ftd_max_days: WATCHING (the attempt timed out without a Follow Through Day)
          UPTREND: the signal stays off until the market is in a correction again, which
            happens either when the index closes below the rally low of the confirmed rally
            (the Follow Through Day failed and the original correction, peak included, is still
            in force) or when it declines far enough from the highest high since the Follow
            Through Day to satisfy the correction test again.

        The day that ends a rally attempt or an uptrend is offered to the day 0 test of the
        same day, so an undercutting day can be the next day 0 at once. SIGNALS.md says so for
        the undercut; this reads the timeout and the failed Follow Through Day the same way.

        The peak is the highest high since the start of the data, so the first Follow Through
        Day cannot come before day min_peak_age_days + ftd_min_days or so, and the results
        depend on where the loaded data begins.

        The state machine runs on the prices of 'index', loaded from the environment the
        strategy runs in (see Signal.bind) over the underlying's trading period; index=None
        runs it on the underlying's own prices. The returned dataframe keeps the index close,
        the state and the peak, day 0, rally low and rally day it is working with next to 'es'.
    """
    WATCHING = "WATCHING"
    DAY0 = "DAY0"
    RALLY = "RALLY"
    UPTREND = "UPTREND"

    def __init__(self, index="IXIC", min_decline=0.08, min_peak_age_days=20, day0_window=5,
                 ftd_min_gain=0.0125, ftd_min_days=4, ftd_max_days=25):
        if index is not None and not isinstance(index, str):
            raise ValueError(f"FTDSignal.index must be a ticker or YAML null, got {index!r}.")
        for param, value in [("min_peak_age_days", min_peak_age_days),
                             ("day0_window", day0_window), ("ftd_min_days", ftd_min_days),
                             ("ftd_max_days", ftd_max_days)]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"FTDSignal.{param} must be a positive number of days, "
                                 f"got {value!r}.")
        if ftd_min_days > ftd_max_days:
            raise ValueError(f"FTDSignal.ftd_min_days ({ftd_min_days}) must not exceed "
                             f"ftd_max_days ({ftd_max_days}).")
        if (isinstance(min_decline, bool) or not isinstance(min_decline, (int, float))
                or not 0 < min_decline < 1):
            raise ValueError(f"FTDSignal.min_decline must be a number between 0 and 1, "
                             f"e.g. 0.08 for 8%, got {min_decline!r}.")
        if isinstance(ftd_min_gain, bool) or not isinstance(ftd_min_gain, (int, float)) or ftd_min_gain < 0:
            raise ValueError(f"FTDSignal.ftd_min_gain must be a non negative number, "
                             f"e.g. 0.0125 for 1.25%, got {ftd_min_gain!r}.")
        prefix = "FTD" if index is None else f"FTD_{index}"
        super().__init__(f"{prefix}_{min_decline}_{min_peak_age_days}_{day0_window}_"
                         f"{ftd_min_gain}_{ftd_min_days}_{ftd_max_days}")
        self.index = index
        self.min_decline = min_decline
        self.min_peak_age_days = min_peak_age_days
        self.day0_window = day0_window
        self.ftd_min_gain = ftd_min_gain
        self.ftd_min_days = ftd_min_days
        self.ftd_max_days = ftd_max_days

    def _index_ohlcv(self, ohlcv):
        if self.index is None:
            return ohlcv
        if self.env is None:
            raise RuntimeError(f"{self.name}: the signal needs an environment to load the "
                               f"{self.index} prices from (see Signal.bind).")
        first, last = ohlcv.index[0], ohlcv.index[-1]
        rv = self.env.load_ohlcv(self.index, first, last)
        rv = rv[(rv.index >= first) & (rv.index <= last)]
        if rv.empty:
            raise ValueError(f"{self.name}: no {self.index} prices between {first} and {last}.")
        return rv

    def _align(self, rv, calendar):
        """
            Puts the index's daily values on the underlying's trading calendar. A day the
            underlying traded and the index did not carries the previous day's values over -
            only such a day, so that a value the machine does not have on a day it did run
            (no rally attempt is under way, say) stays empty. 'es' never carries over: it is a
            one day pulse and has to fire exactly once per Follow Through Day. A Follow
            Through Day the underlying did not trade fires on its next trading day, so that
            the entry signal is not lost.
        """
        ftd_days = list(rv.index[rv.es])
        rv = rv.reindex(calendar)
        did_not_trade = rv.state.isna()     # the machine leaves no row without a state
        rv = rv.mask(did_not_trade, rv.ffill(), axis=0)
        rv['state'] = rv.state.fillna(self.WATCHING)
        rv['es'] = False
        for day in ftd_days:
            i = calendar.searchsorted(day, side='left')
            if i < len(calendar):
                rv.iloc[i, rv.columns.get_loc('es')] = True
        rv['es'] = rv.es.astype(bool)
        return rv

    def _call_impl(self, ohlcv):
        idx = self._index_ohlcv(ohlcv)
        machine, es, rows = _FTDMachine(self, idx), [], []
        for i in range(len(idx)):
            es.append(machine.step(i))
            rows.append(machine.record(i))
        rv = pd.DataFrame(rows, index=idx.index)
        rv.insert(0, 'es', es)
        return self._align(rv, ohlcv.index)

class BreakBelowMA(Signal):
    def __init__(self, period=50):
        super().__init__(f"BreakBelow{period}dMA")
        self.period = period

    def _call_impl(self, ohlcv):
        ohlcv = ohlcv.copy()
        ohlcv['ma' + str(self.period)] = self.ma(ohlcv, self.period)
        ohlcv['es'] = ohlcv['ma' + str(self.period)] >= ohlcv.C
        return ohlcv['es'].to_frame()


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




class WONMarketIndicator():
    def __init__(self, sell_day_return=-0.002, sell_day_vol_return=0.0, sell_day_window_length=5 * 5,
                 sell_day_count_threshold=5, sell_day_cancel_return=0.05,
                 ftd_return=0.013, ftd_vol_return=0, ftd_min_dist=4, ftd_max_dist=12,
                 adv_day_count=20):
        self.SELL_DAY_RETURN = sell_day_return
        self.SELL_DAY_VOL_RETURN = sell_day_vol_return
        self.SELL_DAY_WINDOW_LENGTH = sell_day_window_length
        self.SELL_DAY_COUNT_THRESHOLD = sell_day_count_threshold
        self.SELL_DAY_CANCEL_RETURN = sell_day_cancel_return
        self.FTD_RETURN = ftd_return
        self.FTD_VOL_RETURN = ftd_vol_return
        self.FTD_MIN_DIST = ftd_min_dist
        self.FTD_MAX_DIST = ftd_max_dist
        self.ADV_DAY_COUNT = adv_day_count

    def __call__(self, ohlcv):
        t = ohlcv
        t['ret'] = (t.Close / t.Close.shift()) - 1
        t['volume_ret'] = (t.Volume / t.Volume.shift()) - 1
        t['adv'] = t.Volume.rolling(self.ADV_DAY_COUNT).mean()
        t['vol_vs_adv'] = t.Volume / t.adv

        # distribution days
        t['is_sell_day'] = (t.ret < self.SELL_DAY_RETURN) & (t.volume_ret > self.SELL_DAY_VOL_RETURN)
        t['sell_day_count'] = 0

        # market state transitions
        t['MARKET_STATE'] = np.NaN  # 1: uptrend, -1: correction
        t['rally_day_count'] = 0
        t['rally_day_close'] = np.NaN
        t['FTD'] = False
        t['memo'] = ""
        for ndx, row in t.iterrows():
            MARKET_STATE = t.columns.get_loc('MARKET_STATE')

            if ndx == 7306:
                print("breakpoint")

            if ndx < self.SELL_DAY_WINDOW_LENGTH:
                t.iloc[ndx, MARKET_STATE] = 1
                continue

            if t.iloc[ndx - 1, MARKET_STATE] == 1:
                # update sell day count
                sell_day_count = 0
                for i in range(self.SELL_DAY_WINDOW_LENGTH):
                    w = ndx - i
                    r = t.iloc[ndx, t.columns.get_loc('Close')]/t.iloc[ndx-i, t.columns.get_loc('Close')] - 1
                    isd = t.iloc[ndx-i, t.columns.get_loc('is_sell_day')]
                    if isd and r <= self.SELL_DAY_CANCEL_RETURN:
                        sell_day_count = sell_day_count + 1
                t.iloc[ndx, t.columns.get_loc('sell_day_count')] = sell_day_count
                if sell_day_count >= self.SELL_DAY_COUNT_THRESHOLD:
                    t.iloc[ndx, MARKET_STATE] = -1  # to market in correction
                else:
                    t.iloc[ndx, MARKET_STATE] = t.iloc[ndx - 1, MARKET_STATE]  # stay in uptrend
                continue

            if (t.iloc[ndx - 1, MARKET_STATE] == -1):  # in correction
                if t.iloc[ndx-1, t.columns.get_loc('rally_day_count')] == 0:
                    if t.iloc[ndx, t.columns.get_loc('ret')] > 0:
                        t.iloc[ndx, t.columns.get_loc('rally_day_count')] = 1
                        t.iloc[ndx, t.columns.get_loc('rally_day_close')] = t.iloc[ndx, t.columns.get_loc('Close')]
                    t.iloc[ndx, MARKET_STATE] = t.iloc[ndx - 1, MARKET_STATE]
                else:  # we are in a rally
                    if t.iloc[ndx, t.columns.get_loc('Close')] < t.iloc[ndx - 1, t.columns.get_loc('rally_day_close')]:
                        t.iloc[ndx, t.columns.get_loc('rally_day_count')] = 0
                        t.iloc[ndx, MARKET_STATE] = t.iloc[ndx - 1, MARKET_STATE]
                        t.iloc[ndx, t.columns.get_loc('memo')] = 'Below FDR close'
                    elif t.iloc[ndx - 1, t.columns.get_loc('rally_day_count')] >= self.FTD_MAX_DIST:
                        t.iloc[ndx, t.columns.get_loc('rally_day_count')] = 0
                        t.iloc[ndx, MARKET_STATE] = t.iloc[ndx - 1, MARKET_STATE]
                        t.iloc[ndx, t.columns.get_loc('memo')] = "Max rally day count"
                    elif ((t.iloc[ndx, t.columns.get_loc('ret')] > self.FTD_RETURN) and
                          (t.iloc[ndx, t.columns.get_loc('volume_ret')] > self.FTD_VOL_RETURN) and
                          (t.iloc[ndx, t.columns.get_loc('Volume')] > t.iloc[ndx, t.columns.get_loc('adv')]) and
                          (t.iloc[ndx-1, t.columns.get_loc('rally_day_count')] >= self.FTD_MIN_DIST-1)):
                        t.iloc[ndx, t.columns.get_loc('FTD')] = True
                        t.iloc[ndx, MARKET_STATE] = 1  # to uptrend
                    else:
                        t.iloc[ndx, t.columns.get_loc('rally_day_count')] = t.iloc[ndx - 1, t.columns.get_loc(
                            'rally_day_count')] + 1
                        t.iloc[ndx, t.columns.get_loc('rally_day_close')] = t.iloc[
                            ndx - 1, t.columns.get_loc('rally_day_close')]
                        t.iloc[ndx, MARKET_STATE] = t.iloc[ndx - 1, MARKET_STATE]
                continue

            print(f"Should never get here: {ndx}!")

        return t


# mi = WONMarketIndicator()
# rv = mi(ohlcv)
#
# print("done")
