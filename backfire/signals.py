import os
import math
import numpy as np
import pandas as pd

from .base import Signal

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

class FTDSignal(Signal):
    def __init__(self, ftd_min_gain=0.017, rally_attempt_min_days=4):
        super().__init__(f"FTD_{ftd_min_gain}_{rally_attempt_min_days}")
        self.ftd_min_gain = ftd_min_gain
        self.rally_attempt_min_days = rally_attempt_min_days

    def is_follow_through_day(self, ohlcv_data, ftd_min_gain=0.017, rally_attempt_min_days=4):
        """
        Identifies William O'Neil's Follow-Through Day (FTD) from OHLCV data.

        Args:
            ohlcv_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
                                       The DataFrame should be sorted by date in ascending order.
                                       The index should be a DatetimeIndex.
            ftd_min_gain (float): Minimum percentage gain for a day to be considered an FTD
                                  (e.g., 0.017 for 1.7%). O'Neil often cited ranges like 1.5% to 2% or more.
            rally_attempt_min_days (int): Minimum day of the rally attempt on which an FTD can occur (typically 4).

        Returns:
            pd.Series: A boolean Series with the same index as ohlcv_data,
                       True if the day is an FTD, False otherwise.
        """
        if not isinstance(ohlcv_data, pd.DataFrame):
            raise ValueError("ohlcv_data must be a pandas DataFrame.")
        if not all(col in ohlcv_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if ohlcv_data.empty:
            return pd.Series(dtype=bool)

        # Ensure data is sorted by date
        ohlcv_data = ohlcv_data.sort_index()

        ftd_signals = pd.Series(False, index=ohlcv_data.index)
        rally_attempt_day_count = 0
        rally_low = None  # Stores the low of Day 1 of the current rally attempt
        last_significant_low_price = None  # Stores the price of the most recent significant low
        days_since_significant_low = 0

        # Calculate daily percentage change for the 'Close' price
        ohlcv_data['PriceChange'] = ohlcv_data['Close'].pct_change()
        # Shift volume to correctly compare current day's volume with previous day's volume
        ohlcv_data['PreviousVolume'] = ohlcv_data['Volume'].shift(1)

        # Iterate through the data starting from the second day (to have a previous day for comparison)
        for i in range(1, len(ohlcv_data)):

            if i == 1:
                ftd_signals = ftd_signals.to_frame()
                ftd_signals['close'] = 0
                ftd_signals['price_change'] = 0
                ftd_signals['volume'] = 0
                ftd_signals['previous_volume'] = 0
                ftd_signals['rally_attempt_day_count'] = 0
                ftd_signals['rally_low'] = 0
                ftd_signals['last_significant_low_price'] = 0
                ftd_signals['days_since_significant_low'] = 0
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('close')] = ohlcv_data.iloc[i-1, ohlcv_data.columns.get_loc('Close')]
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('price_change')] = ohlcv_data.iloc[i-1, ohlcv_data.columns.get_loc('PriceChange')]
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('volume')] = ohlcv_data.iloc[i-1, ohlcv_data.columns.get_loc('Volume')]
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('previous_volume')] = ohlcv_data.iloc[i-1, ohlcv_data.columns.get_loc('PreviousVolume')]
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('rally_attempt_day_count')] = rally_attempt_day_count
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('rally_low')] = rally_low
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('last_significant_low_price')] = last_significant_low_price
            ftd_signals.iloc[i-1, ftd_signals.columns.get_loc('days_since_significant_low')] = days_since_significant_low

            current_day = ohlcv_data.iloc[i]
            previous_day = ohlcv_data.iloc[i - 1]

            # --- 1. Identify a Potential Market Low before starting a rally attempt count ---
            # This is a simplified way to check for a prior decline.
            # A robust system would have a more explicit definition of a "market correction".
            # We look for a new N-day low to consider the start of a "bottoming process".
            # Let's define a "significant low" as a new 10-day low for this example.
            lookback_for_new_low = 10
            if i >= lookback_for_new_low:
                if current_day['Low'] <= ohlcv_data['Low'].iloc[max(0, i - lookback_for_new_low):i].min():
                    last_significant_low_price = current_day['Low']
                    days_since_significant_low = 0  # Reset counter
                    rally_attempt_day_count = 0  # Reset any ongoing rally attempt if we hit a new low
                    rally_low = None
                    # print(f"{ohlcv_data.index[i].date()}: New significant low detected at {last_significant_low_price:.2f}. Resetting rally attempt.")

            if last_significant_low_price is not None:
                days_since_significant_low += 1

            # --- 2. Start or Continue a Rally Attempt ---
            if rally_attempt_day_count == 0:
                # Day 1 of Rally Attempt: Market closes up after a significant low has been registered.
                # We need to ensure we are past a potential bottoming point.
                if last_significant_low_price is not None and current_day['Close'] > previous_day['Close']:
                    # To be Day 1, the close must be above the low of the day that made the significant low.
                    # And the current low must hold above that significant low.
                    if current_day['Low'] >= last_significant_low_price:
                        rally_attempt_day_count = 1
                        rally_low = current_day['Low']  # This is the low of Day 1 of the rally attempt.
                        # print(f"{ohlcv_data.index[i].date()}: Day 1 of Rally Attempt. Rally Low: {rally_low:.2f}. Days since sig. low: {days_since_significant_low}")
                continue  # Move to the next day if we just started Day 1 or are not in an attempt

            if rally_attempt_day_count > 0:
                # If current day's low undercuts the low of Day 1 of this rally attempt, reset.
                if current_day['Low'] < rally_low:
                    # print(f"{ohlcv_data.index[i].date()}: Rally attempt failed. Current Low {current_day['Low']:.2f} undercut Rally Low {rally_low:.2f}")
                    rally_attempt_day_count = 0
                    rally_low = None
                    last_significant_low_price = None  # Reset this too, to look for a new bottoming process
                    days_since_significant_low = 0

                    # Check if this day itself can be a new Day 1 (if it closes up after undercutting)
                    # This requires re-evaluating if a new significant low was formed.
                    # For simplicity, we'll just reset and wait for a new sequence.
                    # A more complex logic could try to immediately re-evaluate for a new Day 1.
                    if current_day['Close'] > previous_day['Close']:
                        # Potentially a new Day 1, but we need to ensure it's off a proper low.
                        # The main loop will re-evaluate `last_significant_low_price` on the next iteration.
                        pass
                    continue

                rally_attempt_day_count += 1
                # print(f"{ohlcv_data.index[i].date()}: Day {rally_attempt_day_count} of Rally Attempt. Rally Low: {rally_low:.2f}")

                # --- 3. Identify Follow-Through Day ---
                if rally_attempt_day_count >= rally_attempt_min_days:
                    is_significant_gain = current_day['PriceChange'] >= ftd_min_gain
                    is_volume_higher = current_day['Volume'] > current_day[
                        'PreviousVolume']  # Compare with actual previous day's volume

                    # Optional: Check if volume is also above its N-day average
                    # lookback_vol_avg = 50
                    # if i >= lookback_vol_avg:
                    #     avg_volume = ohlcv_data['Volume'].iloc[max(0, i - lookback_vol_avg):i].mean()
                    #     is_volume_above_average = current_day['Volume'] > avg_volume
                    # else:
                    #     is_volume_above_average = False # Not enough data for average

                    if is_significant_gain and is_volume_higher:  # and is_volume_above_average (if using)
                        ftd_signals.iloc[i] = True
                        # print(f"🎉 {ohlcv_data.index[i].date()}: FOLLOW-THROUGH DAY! "
                        #       f"Day {rally_attempt_day_count} of attempt. "
                        #       f"Gain: {current_day['PriceChange']:.2%}, "
                        #       f"Vol: {current_day['Volume']:.0f} > Prev Vol: {current_day['PreviousVolume']:.0f}")

                        # After an FTD, O'Neil suggests the market is in a "confirmed uptrend".
                        # Reset rally attempt tracking for this specific sequence.
                        # The market could have multiple FTDs or failed FTDs over time.
                        rally_attempt_day_count = 0
                        rally_low = None
                        last_significant_low_price = None  # Reset to look for new major cycle lows later
                        days_since_significant_low = 0

        # Clean up helper columns if they were added to the original DataFrame (if not a copy)
        # If a copy is passed to the function, this is not strictly necessary for the caller.
        # However, it's good practice if the function modifies the input df.
        # For this implementation, we assume a copy is made by the caller if original df needs to be preserved.
        # del ohlcv_data['PriceChange']
        # del ohlcv_data['PreviousVolume']

        return ftd_signals

    def _call_impl(self, ohlcv):
        w = ohlcv.copy()
        w.rename(columns={'O': 'Open',
                          'H': 'High',
                          'L': 'Low',
                          'C': 'Close',
                          'V': 'Volume'}, inplace=True)
        w.index = pd.to_datetime(w.index)
        rv = self.is_follow_through_day(
            ohlcv_data=w,
            ftd_min_gain=self.ftd_min_gain,
            rally_attempt_min_days=self.rally_attempt_min_days)
        rv.rename(columns={0: 'es'}, inplace=True)
        # rv.rename('es', inplace=True)
        # rv = rv.to_frame()
        rv.index = rv.index.date
        return rv

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