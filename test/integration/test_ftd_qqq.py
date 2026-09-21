"""
    Runs FTDSignal with its default parameters over the QQQ market data in md/ and checks the
    Follow Through Days it finds against the reference dates in docs/SIGNALS.md.

    Those dates are IBD style discretionary calls on the Nasdaq Composite, so they are sanity
    checks rather than exact expectations: SIGNALS.md says the mechanical rules may confirm a
    few days earlier or later and that volume on an ETF proxy such as QQQ does not always
    agree with index volume. Hence the tolerance below, and the four reference episodes the
    rules cannot reach at all, which are xfail with the reason spelled out.
"""
from datetime import date
from pathlib import Path

import pytest

from backfire.base import Environment
from backfire.signals import FTDSignal

MD = Path(__file__).resolve().parents[2] / "md"
FIRST, LAST = date(1999, 3, 10), date(2026, 9, 15)
TOLERANCE_TRADING_DAYS = 10


@pytest.fixture(scope="module")
def qqq():
    """ The signal values over the whole of QQQ, with a date to bar number map. """
    env = Environment(md=str(MD), out_dir="")
    ohlcv = env.load_ohlcv("QQQ", FIRST, LAST)
    signal = FTDSignal(index=None)              # the defaults, on QQQ's own prices
    signal.bind(env)
    rv = signal(ohlcv)
    return rv, {day: i for i, day in enumerate(rv.index)}


def unreachable(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# docs/SIGNALS.md, 'Reference follow-through days'
REFERENCE_FTDS = [
    ("2001-04-10", "Dot-com crash 1"),
    ("2001-10-03", "Post 9/11"),
    ("2002-10-15", "Dot-com crash 2"),
    pytest.param("2003-03-17", "Dot-com crash 3", marks=unreachable(
        "the rules confirm this bottom on 2003-02-18 and stay in UPTREND from there: QQQ "
        "never closes below that rally low and keeps making higher highs, so the correction "
        "test never holds again")),
    pytest.param("2009-03-12", "GFC", marks=unreachable(
        "still UPTREND from the 2008-12-02 Follow Through Day: QQQ never closes below its "
        "rally low of 22.09 (the 2009-03-09 low closed at 22.73) and the high since it, made "
        "on 2009-02-10, is only 18 trading days before that low - short of min_peak_age_days")),
    pytest.param("2010-09-01", "Sep 2010", marks=unreachable(
        "still UPTREND from the 2010-07-13 Follow Through Day: the late August decline never "
        "reaches min_decline below the high of 42.03 made on 2010-08-09")),
    ("2019-01-04", "Xmas 2018"),
    ("2020-04-02", "Covid crash"),
    pytest.param("2023-01-06", "2023 AI/tech rally", marks=unreachable(
        "still UPTREND from the 2022-10-21 Follow Through Day: the 2022-12-28 low is 11.7% "
        "below the high of 296.88 made on 2022-12-13, but that high is only 10 trading days "
        "old - short of min_peak_age_days")),
    ("2023-11-02", "Late 2023"),
    ("2025-04-22", "2025"),
    ("2026-04-08", "2026"),
]


@pytest.mark.parametrize("reference, episode", REFERENCE_FTDS)
def test_a_follow_through_day_is_found_near_each_reference_date(qqq, reference, episode):
    rv, bar_of = qqq
    detected = [bar_of[day] for day in rv.index[rv.es]]
    target = min(rv.index.searchsorted(date.fromisoformat(reference), side='left'),
                 len(rv.index) - 1)

    gap = min((bar - target for bar in detected), key=abs)

    assert abs(gap) <= TOLERANCE_TRADING_DAYS, \
        f"{episode}: the nearest Follow Through Day, {rv.index[target + gap]}, is {gap} " \
        f"trading days from {reference}"


def test_the_signal_fires_a_plausible_number_of_times(qqq):
    rv, _ = qqq

    # 62 over the 27 years of QQQ data at the time of writing; the guard is loose on purpose
    # and is here to catch a machine that gets stuck on or off, not to pin the count down
    assert 40 <= rv.es.sum() <= 90


def test_every_pulse_is_a_follow_through_day_by_the_rules(qqq):
    rv, bar_of = qqq
    signal = FTDSignal(index=None)

    for day in rv.index[rv.es]:
        i = bar_of[day]
        previous = rv.index[i - 1]
        gain = rv.index_close[day] / rv.index_close[previous] - 1
        assert gain >= signal.ftd_min_gain, f"{day}: the gain is only {gain:.2%}"
        assert signal.ftd_min_days <= rv.rally_day[day] <= signal.ftd_max_days, \
            f"{day}: it is day {rv.rally_day[day]} of the rally attempt"


def test_the_signal_values_line_up_with_the_underlying(qqq):
    rv, _ = qqq

    assert rv.es.dtype == bool
    assert not rv.es.isna().any()
    assert rv.es.sum() == rv.id.dropna().nunique()      # one id per pulse
    assert set(rv.state.unique()) <= {FTDSignal.WATCHING, FTDSignal.DAY0, FTDSignal.RALLY,
                                      FTDSignal.UPTREND}
