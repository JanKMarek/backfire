import os

from .base import Signal

# ticker = '^IXIC'
# label = "SDS_standard"
# from_date = "2000-01-01"
# out_dir = r"./out_market_indicator_visualize"
#
# import os
# from datetime import datetime
# import pandas as pd
# import numpy as np
# pd.options.display.float_format = '{:,.2f}'.format
#
# ohlcv = pd.read_csv(os.path.join(r'../md/daily', '^IXIC_1990.csv'))
# #ohlcv['Date'] = pd.to_datetime(ohlcv.Date)
# #ohlcv.set_index('Date', inplace=True, drop=True)
# ohlcv = ohlcv[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
# ohlcv

class ShortMAAboveLongMA(Signal):
    def __init__(self, short_MA, long_MA):
        super().__init__(f"{str(short_MA)}dMAAbove{str(long_MA)}MA")
        self.short_MA = short_MA
        self.long_MA = long_MA

    def __call__(self, in_ohlcv):
        ohlcv = in_ohlcv.copy()
        ohlcv['ma' + str(self.long_MA)] = self.ma(ohlcv, self.long_MA)
        ohlcv['ma' + str(self.short_MA)] = self.ma(ohlcv, self.short_MA)
        ohlcv['above'] = ohlcv['ma' + str(self.short_MA)] >= ohlcv['ma' + str(self.long_MA)]

        # create signal ids
        is_start_of_true_block = ohlcv.above & (ohlcv.above != ohlcv.above.shift(1).fillna(False))
        block_ids_raw = is_start_of_true_block.cumsum()
        ohlcv['id'] = block_ids_raw.where(ohlcv.above)
        ohlcv['es'] = ohlcv.above

        #ohlcv.to_csv(os.path.join(self.env.out_dir, f"{str(short_MA)}ma_above_{str(long_MA)}ma.csv"))
        return ohlcv[['es', 'id']]


class ShortMABelowLongMA(Signal):
    def __init__(self, short_MA, long_MA):
        super().__init__(f"{str(short_MA)}dMABelow{str(long_MA)}MA")
        self.short_MA = short_MA
        self.long_MA = long_MA

    def __call__(self, in_ohlcv):
        ohlcv = in_ohlcv.copy()
        ohlcv['ma' + str(self.long_MA)] = self.ma(ohlcv, self.long_MA)
        ohlcv['ma' + str(self.short_MA)] = self.ma(ohlcv, self.short_MA)
        ohlcv['below'] = ohlcv['ma' + str(self.short_MA)] < ohlcv['ma' + str(self.long_MA)]
        # create signal ids
        is_start_of_true_block = ohlcv.below & (ohlcv.below != ohlcv.below.shift(1).fillna(False))
        block_ids_raw = is_start_of_true_block.cumsum()
        ohlcv['id'] = block_ids_raw.where(ohlcv.below)
        ohlcv['es'] = ohlcv.below

        #ohlcv.to_csv(os.path.join(self.env.out_dir, f"{str(short_MA)}ma_below_{str(long_MA)}ma.csv"))
        return ohlcv[['es', 'id']]



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