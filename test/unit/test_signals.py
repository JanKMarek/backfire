from datetime import date

import pandas as pd
import pytest

from backfire.base import Environment, Signal, SignalDrivenStrategy
from backfire.signals import (
    AlwaysOffSignal,
    AlwaysOnSignal,
    FTDSignal,
    OrSignal,
    RetracementSignal,
    ReverseSignal,
    TakeProfitSignal,
    TrailingTakeProfitSignal,
)


class FixedSignal(Signal):
    """
        A signal with hand written daily values, for testing combinators.
    """
    def __init__(self, name, values):
        super().__init__(name)
        self.values = values

    def _call_impl(self, ohlcv):
        return pd.DataFrame({'es': self.values}, index=ohlcv.index)


@pytest.fixture
def ohlcv():
    days = pd.bdate_range("2020-01-01", periods=4).date
    return pd.DataFrame({'O': 1.0, 'H': 2.0, 'L': 0.5, 'C': 1.0, 'V': 100.0}, index=days)


def test_or_is_on_when_at_least_one_signal_is_on(ohlcv):
    a = FixedSignal("a", [True, False, False, True])
    b = FixedSignal("b", [False, True, False, True])

    rv = OrSignal(a, b)(ohlcv)

    assert rv.es.tolist() == [True, True, False, True]
    assert rv.es.dtype == bool
    # the constituents are kept for inspection next to the combined value
    assert rv['a'].tolist() == [True, False, False, True]
    assert rv['b'].tolist() == [False, True, False, True]


def test_or_takes_any_number_of_signals(ohlcv):
    one = OrSignal(FixedSignal("a", [False, True, False, False]))(ohlcv)
    three = OrSignal(AlwaysOffSignal(), AlwaysOffSignal(), AlwaysOnSignal())(ohlcv)

    assert one.es.tolist() == [False, True, False, False]
    assert three.es.all()


def test_or_treats_a_missing_signal_value_as_off(ohlcv):
    a = FixedSignal("a", [None, None, True, None])
    b = FixedSignal("b", [None, False, False, True])

    rv = OrSignal(a, b)(ohlcv)

    assert rv.es.tolist() == [False, False, True, True]


def test_or_signal_ids_number_the_combined_on_blocks(ohlcv):
    a = FixedSignal("a", [True, False, False, True])
    b = FixedSignal("b", [True, False, True, True])

    rv = OrSignal(a, b)(ohlcv)

    assert rv.es.tolist() == [True, False, True, True]
    assert rv.id.tolist()[0] == 1
    assert pd.isna(rv.id.tolist()[1])
    assert rv.id.tolist()[2:] == [2, 2]


def test_or_is_named_after_its_signals():
    assert OrSignal(AlwaysOnSignal(), AlwaysOffSignal()).name == \
        "OrSignal_AlwaysOnSignal_or_AlwaysOffSignal"


def test_or_needs_at_least_one_signal():
    with pytest.raises(ValueError, match="at least one"):
        OrSignal()


def test_or_rejects_arguments_that_are_not_signals():
    with pytest.raises(ValueError, match="must be signals"):
        OrSignal(AlwaysOnSignal(), "ShortMAAboveLongMA")


def position_row(close, buy_price=100.0, pos=1000, day=None):
    return pd.Series({'C': close, 'buy_price': buy_price, 'pos': pos}, name=day)


def test_a_market_signal_has_no_position_dependent_part():
    assert AlwaysOnSignal().evaluate_position(position_row(200.0)) is False


def test_take_profit_is_off_up_front_and_on_once_the_trade_gain_exceeds_the_threshold(ohlcv):
    tp = TakeProfitSignal(threshold=0.25)

    assert not tp(ohlcv).es.any()
    assert tp.evaluate_position(position_row(125.0)) is False   # at the threshold, not above
    assert tp.evaluate_position(position_row(125.01)) is True
    assert tp.evaluate_position(position_row(90.0)) is False


def test_take_profit_is_off_without_an_open_position():
    tp = TakeProfitSignal(threshold=0.25)

    assert tp.evaluate_position(position_row(200.0, pos=0)) is False
    assert tp.evaluate_position(position_row(200.0, buy_price=float('nan'))) is False


def test_take_profit_is_named_after_its_threshold():
    assert TakeProfitSignal(threshold=0.25).name == "TakeProfit_0.25"


@pytest.mark.parametrize("threshold", [None, 0, -0.1, "0.25", True])
def test_take_profit_needs_a_positive_number(threshold):
    with pytest.raises(ValueError, match="positive number"):
        TakeProfitSignal(threshold=threshold)


def test_or_forwards_the_position_dependent_part_of_its_signals():
    combined = OrSignal(AlwaysOffSignal(), TakeProfitSignal(threshold=0.1))

    assert combined.evaluate_position(position_row(111.0)) is True
    assert combined.evaluate_position(position_row(105.0)) is False


@pytest.fixture
def trending_ohlcv():
    """
        Ten days of closes 100, 102, ... 118, so the 5d average of the previous closes on
        the last day is 112.
    """
    days = pd.bdate_range("2020-01-01", periods=10).date
    close = [100.0 + 2 * i for i in range(10)]
    return pd.DataFrame({'O': close, 'H': close, 'L': close, 'C': close, 'V': 100.0}, index=days)


def test_trailing_take_profit_is_off_up_front_and_keeps_its_moving_average(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)

    rv = ts(trending_ohlcv)

    assert not rv.es.any()
    assert rv.trailing_take_profit.iloc[-1] == 112.0
    assert pd.isna(rv.trailing_take_profit.iloc[0])        # no previous close on the first day
    assert rv.trailing_take_profit.iloc[1] == 100.0        # partial window while warming up


def test_trailing_take_profit_arms_on_the_threshold_gain_and_fires_below_the_average(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    ts(trending_ohlcv)
    last = trending_ohlcv.index[-1]                 # average of the previous closes: 112

    assert ts.evaluate_position(position_row(107.0, day=last)) is False   # not armed yet
    assert ts.evaluate_position(position_row(113.0, day=last)) is False   # arms, above avg
    assert ts.armed is True
    assert ts.evaluate_position(position_row(111.0, day=last)) is True


def test_trailing_take_profit_forgets_the_armed_state_when_reset(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    ts(trending_ohlcv)
    last = trending_ohlcv.index[-1]
    ts.evaluate_position(position_row(113.0, day=last))

    ts.reset()

    assert ts.armed is False
    assert ts.evaluate_position(position_row(107.0, day=last)) is False


def test_trailing_take_profit_is_off_without_an_open_position(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    ts(trending_ohlcv)
    last = trending_ohlcv.index[-1]

    assert ts.evaluate_position(position_row(111.0, pos=0, day=last)) is False
    assert ts.armed is False


def test_trailing_take_profit_must_see_the_prices_before_it_is_evaluated():
    with pytest.raises(RuntimeError, match="computed from the prices"):
        TrailingTakeProfitSignal(threshold=0.1, period=5).evaluate_position(position_row(111.0))


def test_trailing_take_profit_is_named_after_its_threshold_and_period():
    assert TrailingTakeProfitSignal(threshold=0.2, period=10).name == "TrailingTakeProfit_0.2_10dMA"


@pytest.mark.parametrize("threshold, period", [(0, 10), (-0.1, 10), ("0.2", 10), (True, 10),
                                               (0.2, 0), (0.2, 2.5), (0.2, None), (0.2, "10")])
def test_trailing_take_profit_needs_a_positive_threshold_and_a_positive_number_of_days(threshold, period):
    with pytest.raises(ValueError, match="positive number"):
        TrailingTakeProfitSignal(threshold=threshold, period=period)


def test_or_evaluates_every_signal_so_a_trailing_take_profit_sees_every_day(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    combined = OrSignal(TakeProfitSignal(threshold=0.05), ts)
    combined(trending_ohlcv)
    last = trending_ohlcv.index[-1]

    # the take profit is on, and the trailing take profit must still get to arm itself
    assert combined.evaluate_position(position_row(111.0, day=last)) is True
    assert ts.armed is True


def bar_row(high, close, buy_price=100.0, pos=1000):
    return pd.Series({'H': high, 'C': close, 'buy_price': buy_price, 'pos': pos})


def test_retracement_is_off_up_front_and_on_once_the_close_retraces_from_the_high(ohlcv):
    rs = RetracementSignal(retracement=0.1)

    assert not rs(ohlcv).es.any()
    assert rs.evaluate_position(bar_row(high=110.0, close=105.0)) is False
    assert rs.evaluate_position(bar_row(high=120.0, close=115.0)) is False   # new high
    assert rs.evaluate_position(bar_row(high=109.0, close=108.0)) is False   # at 90% of 120
    assert rs.evaluate_position(bar_row(high=109.0, close=107.99)) is True


def test_retracement_fires_on_a_close_far_below_the_same_days_high():
    rs = RetracementSignal(retracement=0.1)

    assert rs.evaluate_position(bar_row(high=120.0, close=107.0)) is True


def test_retracement_forgets_the_high_when_reset():
    rs = RetracementSignal(retracement=0.1)
    rs.evaluate_position(bar_row(high=150.0, close=150.0))

    rs.reset()

    assert rs.evaluate_position(bar_row(high=110.0, close=105.0)) is False


def test_retracement_forgets_the_high_when_computed_from_new_prices(ohlcv):
    rs = RetracementSignal(retracement=0.1)
    rs.evaluate_position(bar_row(high=150.0, close=150.0))

    rs(ohlcv)

    assert rs.high is None


def test_retracement_is_off_without_an_open_position():
    rs = RetracementSignal(retracement=0.1)

    assert rs.evaluate_position(bar_row(high=150.0, close=100.0, pos=0)) is False
    assert rs.evaluate_position(bar_row(high=150.0, close=100.0, buy_price=float('nan'))) is False
    assert rs.high is None


def test_retracement_is_named_after_its_retracement():
    assert RetracementSignal(retracement=0.1).name == "Retracement_0.1"


@pytest.mark.parametrize("retracement", [None, 0, -0.1, 1, 1.5, "0.1", True])
def test_retracement_needs_a_number_between_0_and_1(retracement):
    with pytest.raises(ValueError, match="between 0 and 1"):
        RetracementSignal(retracement=retracement)


def bars(closes, volumes=None, lows=None, highs=None, start="2020-01-01"):
    """
        Daily bars with the given closes; the low is 1 below the close, the high 1 above it
        and the volume 100 unless given.
    """
    days = pd.bdate_range(start, periods=len(closes)).date
    lows = [c - 1 for c in closes] if lows is None else lows
    highs = [c + 1 for c in closes] if highs is None else highs
    volumes = [100.0] * len(closes) if volumes is None else volumes
    return pd.DataFrame({'O': closes, 'H': highs, 'L': lows, 'C': closes, 'V': volumes},
                        index=days)


# The test parameters below are the real ones shrunk so that a test case is a handful of bars:
# a 3 day window, a peak at least 3 days old, a 2% gain and a day 4 to day 6 window for the
# Follow Through Day. In every sequence starting with a close of 100 the peak is therefore the
# high of day 0, 101, and a day 0 needs a low of 92.92 or less made on day 3 or later.
W, D0, R, U = FTDSignal.WATCHING, FTDSignal.DAY0, FTDSignal.RALLY, FTDSignal.UPTREND

# day 3 is day 0 (low 92), day 4 is day 1, days 5 and 6 are days 2 and 3 and day 7 is day 4
# of the attempt with a 2.1% gain - the Follow Through Day when its volume is higher
DECLINE_AND_RALLY = [100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 94.2, 96.2]
FTD_VOLUMES = [100.0] * 7 + [200.0]


def ftd(**kwargs):
    params = dict(index=None, min_decline=0.08, min_peak_age_days=3, day0_window=3,
                  ftd_min_gain=0.02, ftd_min_days=4, ftd_max_days=6)
    params.update(kwargs)
    return FTDSignal(**params)


# --- day 0 and the correction test -----------------------------------------------------

def test_no_day0_before_the_peak_is_old_enough():
    # deep enough from the first day on, but the peak has to be min_peak_age_days old
    rv = ftd(day0_window=1)(bars([100.0, 90.0, 91.0, 90.5]))

    assert rv.state.tolist() == [W, W, W, D0]
    assert rv.day0_date.tolist()[3] == rv.index[3]


def test_no_day0_when_the_decline_from_the_peak_is_too_shallow():
    rv = ftd()(bars([100.0, 98.0, 96.0, 95.0, 94.5]))

    assert rv.state.tolist() == [W, W, W, W, W]
    assert rv.decline_pct.tolist()[4] == pytest.approx(93.5 / 101.0 - 1)    # -7.4%, not -8%


def test_a_new_low_that_fails_the_correction_test_leaves_no_trace():
    # the same shallow decline, then a low that does clear the 8%: it is day 0 at once
    rv = ftd()(bars([100.0, 98.0, 96.0, 95.0, 94.5, 92.0]))

    assert rv.state.tolist() == [W, W, W, W, W, D0]
    assert rv.rally_low.tolist()[5] == 91.0


def test_day0_is_the_qualifying_new_low():
    rv = ftd()(bars(DECLINE_AND_RALLY[:4]))

    assert rv.state.tolist() == [W, W, W, D0]
    assert rv.day0_date.tolist()[3] == rv.index[3]
    assert rv.rally_low.tolist()[3] == 92.0
    assert not rv.es.any()


def test_the_peak_is_the_highest_high_since_the_start_of_the_data():
    # the peak is the high of day 2 (111), so day 4 is deep enough but too close to it
    rv = ftd()(bars([100.0, 105.0, 110.0, 108.0, 100.0, 99.0]))

    assert rv.state.tolist() == [W, W, W, W, W, D0]
    assert rv.peak.tolist()[5] == 111.0
    assert rv.peak_date.tolist()[5] == rv.index[2]


def test_no_day0_during_the_day0_window_warmup():
    rv = ftd(day0_window=5, min_peak_age_days=1)(bars([100.0, 90.0, 89.0, 88.0]))

    assert rv.state.tolist() == [W, W, W, W]


# --- day 0 sliding and day 1 -----------------------------------------------------------

def test_day0_slides_to_a_later_lower_low():
    rv = ftd()(bars([100.0, 98.0, 96.0, 93.0, 92.0]))

    assert rv.state.tolist() == [W, W, W, D0, D0]
    assert rv.day0_date.tolist()[4] == rv.index[4]
    assert rv.rally_low.tolist()[4] == 91.0


def test_a_lower_low_that_closes_up_is_a_new_day0_not_day1():
    # day 4 closes higher than day 3 but makes a lower low, so it is the new day 0; day 5 is
    # the first day after it that closes up and it is day 1
    closes = [100.0, 98.0, 96.0, 93.0, 93.5, 94.0]
    lows = [99.0, 97.0, 95.0, 92.0, 91.0, 92.5]
    rv = ftd()(bars(closes, lows=lows))

    assert rv.state.tolist() == [W, W, W, D0, D0, R]
    assert rv.day0_date.tolist()[4] == rv.index[4]
    assert rv.rally_low.tolist()[4] == 91.0
    assert pd.isna(rv.rally_day.tolist()[4])    # day 4 is day 0, not day 1 of the attempt
    assert rv.rally_day.tolist()[5] == 1


def test_a_new_window_low_above_the_day0_low_does_not_move_day0():
    # day 6 makes the lowest low of the last 3 days, but it is above the day 0 low of 92
    closes = [110.0, 98.0, 96.0, 93.0, 92.8, 92.6, 92.4]
    lows = [109.0, 97.0, 95.0, 92.0, 96.0, 95.0, 94.0]
    rv = ftd()(bars(closes, lows=lows))

    assert rv.state.tolist() == [W, W, W, D0, D0, D0, D0]
    assert rv.day0_date.tolist()[6] == rv.index[3]
    assert rv.rally_low.tolist()[6] == 92.0


def test_day1_is_the_first_up_close_after_day0():
    rv = ftd()(bars(DECLINE_AND_RALLY[:5]))

    assert rv.state.tolist() == [W, W, W, D0, R]
    assert rv.rally_day.tolist()[4] == 1
    assert rv.rally_low.tolist()[4] == 92.0


def test_a_flat_close_is_not_day1():
    rv = ftd()(bars([100.0, 98.0, 96.0, 93.0, 93.0]))

    assert rv.state.tolist() == [W, W, W, D0, D0]


# --- the rally attempt -----------------------------------------------------------------

def test_the_rally_low_is_frozen_at_day1_and_an_intraday_dip_below_it_does_not_end_the_attempt():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 93.5]
    lows = [99.0, 97.0, 95.0, 92.0, 93.0, 93.5, 91.5]     # day 6 dips to 91.5, closes at 93.5
    rv = ftd()(bars(closes, lows=lows))

    assert rv.state.tolist() == [W, W, W, D0, R, R, R]
    assert rv.rally_low.tolist()[4:] == [92.0, 92.0, 92.0]
    assert rv.rally_day.tolist()[4:] == [1, 2, 3]


def test_a_close_below_the_rally_low_ends_the_attempt_and_the_day_is_the_next_day0():
    # day 6 closes at 91.5, below the rally low of 92; its own low is the next day 0
    rv = ftd()(bars([100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 91.5]))

    assert rv.state.tolist() == [W, W, W, D0, R, R, D0]
    assert rv.day0_date.tolist()[6] == rv.index[6]
    assert rv.rally_low.tolist()[6] == 90.5
    assert not rv.es.any()


def test_days_2_and_3_of_the_attempt_can_never_be_a_follow_through_day():
    # day 5 gains 2.7% and day 6 another 2.1%, both on higher volume, but they are days 2 and 3
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 96.5, 98.5]
    rv = ftd()(bars(closes, volumes=[100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 300.0]))

    assert rv.state.tolist() == [W, W, W, D0, R, R, R]
    assert not rv.es.any()


# day 3 is day 0, day 4 is day 1 and day 6 closes at 100, so day 7 - day 4 of the attempt -
# is a Follow Through Day exactly when it closes at or above 100 * (1 + ftd_min_gain)
RALLY_TO_100 = [100.0, 98.0, 96.0, 93.0, 94.0, 96.0, 100.0]


def test_the_follow_through_day_needs_the_minimum_gain():
    threshold = 100.0 * (1 + 0.02)

    on_it = ftd()(bars(RALLY_TO_100 + [threshold], volumes=FTD_VOLUMES))
    below_it = ftd()(bars(RALLY_TO_100 + [threshold - 0.01], volumes=FTD_VOLUMES))

    assert on_it.es.tolist() == [False] * 7 + [True]
    assert below_it.es.tolist() == [False] * 8
    assert below_it.state.tolist()[7] == R


def test_the_follow_through_day_needs_higher_volume_than_the_previous_day():
    rv = ftd()(bars(DECLINE_AND_RALLY))     # a flat 100 throughout

    assert rv.state.tolist() == [W, W, W, D0, R, R, R, R]
    assert not rv.es.any()


def test_a_follow_through_day_on_the_last_day_of_the_window_beats_the_timeout():
    # day 1 is day 4, so day 6 of the attempt is day 9 - the last day ftd_max_days allows
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.2, 94.4, 94.6, 94.8, 96.8]
    rv = ftd()(bars(closes, volumes=[100.0] * 9 + [200.0]))

    assert rv.state.tolist()[9] == U
    assert rv.es.tolist() == [False] * 9 + [True]
    assert rv.rally_day.tolist()[9] == 6


def test_the_attempt_times_out_at_the_end_of_the_window():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.2, 94.4, 94.6, 94.8, 94.9]
    rv = ftd()(bars(closes, volumes=[100.0] * 9 + [200.0]))

    assert rv.state.tolist() == [W, W, W, D0, R, R, R, R, R, W]
    assert not rv.es.any()


def test_after_a_timeout_the_next_qualifying_low_is_a_new_day0_measured_from_the_same_peak():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.2, 94.4, 94.6, 94.8, 94.9, 92.5]
    rv = ftd()(bars(closes, volumes=[100.0] * 9 + [200.0, 100.0]))

    assert rv.state.tolist()[9:] == [W, D0]
    assert rv.rally_low.tolist()[10] == 91.5
    assert rv.peak_date.tolist()[10] == rv.index[0]


# --- the pulse -------------------------------------------------------------------------

def test_the_signal_is_true_only_on_the_follow_through_day():
    closes = DECLINE_AND_RALLY + [96.5, 97.0, 97.5]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [100.0] * 3))

    assert rv.es.tolist() == [False] * 7 + [True, False, False, False]
    assert rv.state.tolist()[7:] == [U, U, U, U]
    assert rv.rally_day.tolist()[7] == 4
    assert rv.id.tolist()[7] == 1


# a Follow Through Day on day 7, a close below its rally low of 92 on day 8 that fails it and
# is the next day 0, day 1 on day 9 and a second Follow Through Day on day 12
TWO_FTDS = DECLINE_AND_RALLY + [91.5, 92.5, 92.6, 92.7, 94.6]
TWO_FTDS_VOLUMES = FTD_VOLUMES + [100.0] * 4 + [200.0]


def test_each_follow_through_day_gets_its_own_id():
    rv = ftd()(bars(TWO_FTDS, volumes=TWO_FTDS_VOLUMES))

    assert rv.es.tolist() == [False] * 7 + [True] + [False] * 4 + [True]
    assert rv.id.tolist()[7] == 1
    assert rv.id.tolist()[12] == 2


# --- re-arming after a Follow Through Day ----------------------------------------------

def test_the_signal_stays_off_while_the_uptrend_holds():
    closes = DECLINE_AND_RALLY + [96.5, 96.6, 96.7, 96.8, 96.9]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [200.0] * 5))

    assert rv.es.tolist() == [False] * 7 + [True] + [False] * 5
    assert set(rv.state.tolist()[7:]) == {U}


def test_a_close_below_the_confirmed_rally_low_fails_the_follow_through_day():
    rv = ftd()(bars(DECLINE_AND_RALLY + [91.5], volumes=FTD_VOLUMES + [100.0]))

    assert rv.state.tolist()[7:] == [U, D0]
    assert rv.day0_date.tolist()[8] == rv.index[8]


def test_a_failed_follow_through_day_reuses_the_original_correction_peak():
    # day 8 is 10.4% below the original peak of 101 but only 6.9% below the high since the
    # Follow Through Day, so it is a day 0 only because the failure restores the old peak
    rv = ftd()(bars(DECLINE_AND_RALLY + [91.5], volumes=FTD_VOLUMES + [100.0]))

    assert rv.state.tolist()[8] == D0
    assert rv.peak.tolist()[8] == 101.0
    assert rv.peak_date.tolist()[8] == rv.index[0]


def test_a_new_correction_after_a_follow_through_day_is_measured_from_the_high_since_it():
    # a run up to a new high of 111 on day 10, then an 8% decline from it by day 13, with no
    # close below the confirmed rally low of 92
    closes = DECLINE_AND_RALLY + [100.0, 105.0, 110.0, 108.0, 105.0, 102.0]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [100.0] * 6))

    assert rv.state.tolist()[12:] == [U, D0]
    assert rv.peak.tolist()[13] == 111.0
    assert rv.peak_date.tolist()[13] == rv.index[10]


def test_a_shallow_pullback_after_a_follow_through_day_does_not_re_arm():
    # day 11 is 9.4% below the original peak of 101 - a day 0 if the original peak still
    # counted - but only 5.9% below the high of 97.2 made on the Follow Through Day
    closes = DECLINE_AND_RALLY + [96.0, 95.0, 94.0, 92.5]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [100.0] * 4))

    assert set(rv.state.tolist()[7:]) == {U}
    assert rv.es.tolist() == [False] * 7 + [True] + [False] * 4


def test_a_decline_from_a_too_young_post_follow_through_day_high_does_not_re_arm():
    # a new high of 111 on day 8, then a deep drop: only day 11 is min_peak_age_days after it
    closes = DECLINE_AND_RALLY + [110.0, 100.0, 99.0, 98.0]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [100.0] * 4))

    assert rv.state.tolist()[8:] == [U, U, U, D0]
    assert rv.peak_date.tolist()[11] == rv.index[8]


# --- the machine explaining itself -----------------------------------------------------

def marks(rv, column='event'):
    """ One of the mark columns as a list, with an empty string on the days without one. """
    return rv[column].fillna('').tolist()


def test_a_qualifying_new_low_is_recorded_as_day0():
    rv = ftd()(bars(DECLINE_AND_RALLY[:4]))

    assert marks(rv) == ['', '', '', 'DAY0']


def test_a_lower_low_before_day1_is_recorded_as_day0_lowered():
    rv = ftd()(bars([100.0, 98.0, 96.0, 93.0, 92.0]))

    assert marks(rv) == ['', '', '', 'DAY0', 'DAY0_LOWERED']


def test_the_first_up_close_after_day0_is_recorded_as_day1():
    rv = ftd()(bars(DECLINE_AND_RALLY[:5]))

    assert marks(rv) == ['', '', '', 'DAY0', 'DAY1']


def test_an_undercutting_day_that_is_the_next_day0_records_both():
    rv = ftd()(bars([100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 91.5]))

    assert marks(rv)[6] == 'UNDERCUT+DAY0'


def test_the_end_of_the_window_is_recorded_as_a_timeout():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.2, 94.4, 94.6, 94.8, 94.9]
    rv = ftd()(bars(closes, volumes=[100.0] * 9 + [200.0]))

    assert marks(rv)[9] == 'TIMEOUT'


def test_the_follow_through_day_is_recorded_as_ftd():
    rv = ftd()(bars(DECLINE_AND_RALLY, volumes=FTD_VOLUMES))

    assert marks(rv)[7] == 'FTD'


def test_a_close_below_the_confirmed_rally_low_is_recorded_as_a_failed_follow_through_day():
    rv = ftd()(bars(DECLINE_AND_RALLY + [91.5], volumes=FTD_VOLUMES + [100.0]))

    assert marks(rv)[8] == 'FTD_FAILED+DAY0'


def test_re_arming_on_a_decline_from_the_high_since_is_recorded_as_a_new_correction():
    closes = DECLINE_AND_RALLY + [100.0, 105.0, 110.0, 108.0, 105.0, 102.0]
    rv = ftd()(bars(closes, volumes=FTD_VOLUMES + [100.0] * 6))

    assert marks(rv)[13] == 'NEW_CORRECTION+DAY0'


def test_a_new_low_too_shallow_to_be_day0_records_the_decline_block():
    rv = ftd()(bars([100.0, 98.0, 96.0, 95.0, 94.5]))

    assert marks(rv, 'day0_block')[4] == 'DECLINE'
    assert marks(rv) == [''] * 5


def test_a_new_low_below_a_peak_too_young_records_the_peak_age_block():
    rv = ftd(day0_window=1)(bars([100.0, 90.0, 91.0, 90.5]))

    # days 1 and 2 are deep enough below the peak but too close to it
    assert marks(rv, 'day0_block')[1:] == ['PEAK_AGE', 'PEAK_AGE', '']
    assert marks(rv) == ['', '', '', 'DAY0']


def test_a_new_low_that_fails_both_clauses_records_both():
    # day 1 is neither 8% below the peak nor far enough from it
    rv = ftd(day0_window=1)(bars([100.0, 95.0, 90.0, 89.0]))

    assert marks(rv, 'day0_block')[1] == 'DECLINE+PEAK_AGE'


def test_a_gain_on_day_2_or_3_of_the_attempt_records_the_too_early_block():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 96.5, 98.5]
    rv = ftd()(bars(closes, volumes=[100.0] * 5 + [200.0, 300.0]))

    assert marks(rv, 'ftd_block') == [''] * 5 + ['TOO_EARLY', 'TOO_EARLY']
    assert not rv.es.any()


def test_a_gain_on_flat_volume_records_the_volume_block():
    rv = ftd()(bars(DECLINE_AND_RALLY))     # a flat 100 throughout

    assert marks(rv, 'ftd_block')[7] == 'VOLUME'
    assert not rv.es.any()


def test_a_rally_day_without_the_gain_is_not_blocked_it_is_simply_not_a_candidate():
    rv = ftd()(bars(DECLINE_AND_RALLY[:7], volumes=[100.0] * 5 + [200.0, 300.0]))

    assert marks(rv, 'ftd_block') == [''] * 7


def test_the_signal_is_on_exactly_on_the_days_whose_event_contains_ftd():
    rv = ftd()(bars(TWO_FTDS, volumes=TWO_FTDS_VOLUMES))

    fired = [i for i, event in enumerate(marks(rv)) if 'FTD' in event.split('+')]
    assert fired == [7, 12]
    assert [i for i, on in enumerate(rv.es) if on] == fired
    # FTD_FAILED is a transition of its own and must not be read as a pulse
    assert marks(rv)[8] == 'FTD_FAILED+DAY0'


# --- running on a separate index -------------------------------------------------------

@pytest.fixture
def index_md(tmp_path):
    """
        Market data directory holding IXIC.csv - the decline and rally with a Follow Through Day
        on the last day.
    """
    index = bars(DECLINE_AND_RALLY, volumes=FTD_VOLUMES)
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    index.rename(columns={'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'V': 'Volume'}) \
        .rename_axis('Date').to_csv(md_dir / "IXIC.csv")
    return str(md_dir)


def test_ftd_runs_on_the_index_prices_loaded_from_the_environment(index_md):
    underlying = bars([50.0] * 8)      # flat: on its own prices the signal would never fire
    signal = ftd(index="IXIC")
    signal.bind(Environment(md=index_md, out_dir=""))

    rv = signal(underlying)

    assert rv.index.tolist() == underlying.index.tolist()
    assert rv.index_close.tolist() == DECLINE_AND_RALLY
    assert rv.es.tolist() == [False] * 7 + [True]


def test_the_pulse_is_not_repeated_on_a_day_the_index_did_not_trade(index_md):
    days = list(pd.bdate_range("2020-01-01", periods=8).date)
    extra = date(2020, 1, 11)          # a Saturday, after the Follow Through Day
    underlying = pd.DataFrame({'O': 50.0, 'H': 51.0, 'L': 49.0, 'C': 50.0, 'V': 100.0},
                              index=days + [extra])
    signal = ftd(index="IXIC")
    signal.bind(Environment(md=index_md, out_dir=""))

    rv = signal(underlying)

    assert rv.es.tolist() == [False] * 7 + [True, False]
    assert rv.state.tolist()[-1] == U       # the state does carry over
    # the event does not carry over either
    assert marks(rv) == ['', '', '', 'DAY0', 'DAY1', '', '', 'FTD', '']


def test_a_follow_through_day_the_underlying_missed_fires_on_its_next_trading_day(index_md):
    days = list(pd.bdate_range("2020-01-01", periods=7).date)    # the index trades one more
    later = date(2020, 1, 13)          # the Monday after the Follow Through Day
    underlying = pd.DataFrame({'O': 50.0, 'H': 51.0, 'L': 49.0, 'C': 50.0, 'V': 100.0},
                              index=days + [later])
    signal = ftd(index="IXIC")
    signal.bind(Environment(md=index_md, out_dir=""))

    rv = signal(underlying)

    assert rv.es.tolist() == [False] * 7 + [True]
    # the event moves with the pulse, onto the next day the underlying traded
    assert marks(rv) == ['', '', '', 'DAY0', 'DAY1', '', '', 'FTD']


def test_ftd_on_an_index_needs_an_environment():
    with pytest.raises(RuntimeError, match="needs an environment"):
        ftd(index="IXIC")(bars(DECLINE_AND_RALLY))


# --- construction ----------------------------------------------------------------------

def test_ftd_defaults_and_name():
    signal = FTDSignal()

    assert (signal.index, signal.min_decline, signal.min_peak_age_days, signal.day0_window,
            signal.ftd_min_gain, signal.ftd_min_days, signal.ftd_max_days) \
        == ("^IXIC", 0.08, 20, 5, 0.0125, 4, 25)
    assert signal.name == "FTD_^IXIC_0.08_20_5_0.0125_4_25"
    assert FTDSignal(index=None).name == "FTD_0.08_20_5_0.0125_4_25"


@pytest.mark.parametrize("kwargs, match", [
    ({'index': 5}, "ticker"),
    ({'min_peak_age_days': 0}, "positive number of days"),
    ({'day0_window': 2.5}, "positive number of days"),
    ({'ftd_min_days': True}, "positive number of days"),
    ({'ftd_max_days': 0}, "positive number of days"),
    ({'ftd_min_days': 10, 'ftd_max_days': 5}, "must not exceed"),
    ({'min_decline': 0}, "between 0 and 1"),
    ({'min_decline': 1.5}, "between 0 and 1"),
    ({'min_decline': "0.08"}, "between 0 and 1"),
    ({'ftd_min_gain': -0.01}, "non negative"),
    ({'ftd_min_gain': "0.02"}, "non negative"),
])
def test_ftd_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FTDSignal(**kwargs)


def test_a_strategy_binds_its_environment_to_nested_signals():
    env = Environment(md="md", out_dir="")
    entry = ftd(index="IXIC")
    nested = ftd(index="IXIC")
    SignalDrivenStrategy(env=env, entry_signal=entry,
                         exit_signal=OrSignal(AlwaysOffSignal(), ReverseSignal(nested)))

    assert entry.env is env
    assert nested.env is env


def test_bind_keeps_an_environment_the_signal_was_given():
    own = Environment(md="md", out_dir="")
    signal = AlwaysOnSignal()
    signal.env = own

    signal.bind(Environment(md="other", out_dir=""))

    assert signal.env is own


def test_or_forwards_reset_to_its_signals(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    combined = OrSignal(AlwaysOffSignal(), ts)
    combined(trending_ohlcv)
    ts.evaluate_position(position_row(111.0, day=trending_ohlcv.index[-1]))

    combined.reset()

    assert ts.armed is False
