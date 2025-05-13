
ticker = '^IXIC'
label = "SDS_standard"
from_date = "2000-01-01"
out_dir = r"./out_market_indicator_visualize"

import os
from datetime import datetime
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.2f}'.format

ohlcv = pd.read_csv(os.path.join(r'../md/daily', '^IXIC_1990.csv'))
#ohlcv['Date'] = pd.to_datetime(ohlcv.Date)
#ohlcv.set_index('Date', inplace=True, drop=True)
ohlcv = ohlcv[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
ohlcv


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


mi = WONMarketIndicator()
rv = mi(ohlcv)

print("done")