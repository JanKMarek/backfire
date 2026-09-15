import pandas as pd
import pytest

from backfire.base import Signal
from backfire.signals import AlwaysOffSignal, AlwaysOnSignal, OrSignal


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
