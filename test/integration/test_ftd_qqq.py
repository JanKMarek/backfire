"""
    Runs FTDSignal with its default parameters over the QQQ market data in md/ and checks the
    Follow Through Days it finds against the reference dates in docs/ftd_reference.yaml.

    Those dates are IBD style discretionary calls on the Nasdaq Composite, so they are sanity
    checks rather than exact expectations: SIGNALS.md says the mechanical rules may confirm a
    few days earlier or later and that volume on an ETF proxy such as QQQ does not always
    agree with index volume. Hence the tolerance below, and the four reference episodes the
    rules cannot reach at all, which are xfail with the reason spelled out in the YAML and
    whose 'unreachable' block names the clause the signal records for rejecting them.
"""
from datetime import date
from pathlib import Path

import pytest

from backfire.base import Environment
from backfire.signal_analysis import load_references
from backfire.signals import FTDSignal

MD = Path(__file__).resolve().parents[2] / "md"
FIRST, LAST = date(1999, 3, 10), date(2026, 9, 15)
TOLERANCE_TRADING_DAYS = 10

REFERENCES = [ref for ref in load_references() if ref['kind'] == 'reference']
CANDIDATES = [ref for ref in load_references() if ref['kind'] == 'candidate']
UNREACHABLE = [ref for ref in REFERENCES if ref.get('unreachable')]


@pytest.fixture(scope="module")
def qqq():
    """ The signal values over the whole of QQQ, with a date to bar number map. """
    env = Environment(md=str(MD), out_dir="")
    ohlcv = env.load_ohlcv("QQQ", FIRST, LAST)
    signal = FTDSignal(index=None)              # the defaults, on QQQ's own prices
    signal.bind(env)
    rv = signal(ohlcv)
    return rv, {day: i for i, day in enumerate(rv.index)}


def reference_case(ref):
    """ One reference date as a test case, xfail when the rules cannot reach it. """
    marks = ()
    if ref.get('unreachable'):
        marks = pytest.mark.xfail(strict=True, reason=ref['unreachable']['reason'])
    return pytest.param(ref, id=f"{ref['date']} {ref['episode']}", marks=marks)


def nearest_pulse(rv, bar_of, day):
    """ The Follow Through Day closest to the day, and its distance in trading days. """
    detected = [bar_of[pulse] for pulse in rv.index[rv.es]]
    target = min(rv.index.searchsorted(day, side='left'), len(rv.index) - 1)
    gap = min((bar - target for bar in detected), key=abs)
    return rv.index[target + gap], gap


def test_the_reference_list_is_the_one_signals_md_documents():
    # docs/SIGNALS.md tabulates 12 reference dates and 29 candidates; the YAML is the copy
    # the code reads, so the two have to stay the same size
    assert (len(REFERENCES), len(CANDIDATES)) == (12, 29)
    assert all(ref['date'] is not None for ref in REFERENCES + CANDIDATES)


@pytest.mark.parametrize("ref", [reference_case(ref) for ref in REFERENCES])
def test_a_follow_through_day_is_found_near_each_reference_date(qqq, ref):
    rv, bar_of = qqq

    detected, gap = nearest_pulse(rv, bar_of, ref['date'])

    assert abs(gap) <= TOLERANCE_TRADING_DAYS, \
        f"{ref['episode']}: the nearest Follow Through Day, {detected}, is {gap} " \
        f"trading days from {ref['date']}"


@pytest.mark.parametrize("ref", UNREACHABLE,
                         ids=[f"{ref['date']} {ref['episode']}" for ref in UNREACHABLE])
def test_the_signal_records_why_an_unreachable_reference_is_out_of_reach(qqq, ref):
    rv, bar_of = qqq
    unreachable = ref['unreachable']

    # the correction precondition is what stops each of these, on the day the reference call
    # was made off; the signal writes the clause that failed into day0_block
    blocked = rv.day0_block[unreachable['low']]
    assert unreachable['block'] in str(blocked).split('+'), \
        f"{ref['episode']}: on {unreachable['low']} the signal recorded {blocked!r}, " \
        f"not {unreachable['block']}"

    # and it is still in the uptrend the earlier Follow Through Day confirmed
    assert rv.state[unreachable['low']] == FTDSignal.UPTREND
    assert nearest_pulse(rv, bar_of, ref['date'])[0] == unreachable['detected_instead']


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
        assert 'FTD' in str(rv.event[day]).split('+'), \
            f"{day}: the signal fired but recorded the event {rv.event[day]!r}"


def test_the_signal_values_line_up_with_the_underlying(qqq):
    rv, _ = qqq

    assert rv.es.dtype == bool
    assert not rv.es.isna().any()
    assert rv.es.sum() == rv.id.dropna().nunique()      # one id per pulse
    assert set(rv.state.unique()) <= {FTDSignal.WATCHING, FTDSignal.DAY0, FTDSignal.RALLY,
                                      FTDSignal.UPTREND}
