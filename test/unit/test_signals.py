import pandas as pd
import pytest

from backfire.base import Signal
from backfire.signals import (
    AlwaysOffSignal,
    AlwaysOnSignal,
    OrSignal,
    RetracementSignal,
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


def test_or_forwards_reset_to_its_signals(trending_ohlcv):
    ts = TrailingTakeProfitSignal(threshold=0.1, period=5)
    combined = OrSignal(AlwaysOffSignal(), ts)
    combined(trending_ohlcv)
    ts.evaluate_position(position_row(111.0, day=trending_ohlcv.index[-1]))

    combined.reset()

    assert ts.armed is False
