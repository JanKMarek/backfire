from datetime import date

import pandas as pd
import pytest
import yaml

from backfire.signal_analysis import (
    attempt_summary,
    baseline_dates,
    extract_episodes,
    failure_rate,
    forward_outcomes,
    forward_paths,
    forward_table,
    load_references,
    match_references,
    pulse_dates,
    sensitivity,
)
from backfire.signals import FTDSignal


def bars(closes, volumes=None, lows=None, highs=None, opens=None, start="2020-01-01"):
    """
        Daily bars with the given closes; the low is 1 below the close, the high 1 above it,
        the open half a point below it and the volume 100 unless given.
    """
    days = pd.bdate_range(start, periods=len(closes)).date
    return pd.DataFrame(
        {'O': [c - 0.5 for c in closes] if opens is None else opens,
         'H': [c + 1 for c in closes] if highs is None else highs,
         'L': [c - 1 for c in closes] if lows is None else lows,
         'C': closes,
         'V': [100.0] * len(closes) if volumes is None else volumes},
        index=days)


def ftd(**kwargs):
    """ The signal shrunk so that a case is a handful of bars - see test_signals.py. """
    params = dict(index=None, min_decline=0.08, min_peak_age_days=3, day0_window=3,
                  ftd_min_gain=0.02, ftd_min_days=4, ftd_max_days=6)
    params.update(kwargs)
    return FTDSignal(**params)


# day 3 is day 0 (low 92), day 4 is day 1 and day 7 is day 4 of the attempt with a 2.1% gain -
# the Follow Through Day when its volume is higher
DECLINE_AND_RALLY = [100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 94.2, 96.2]
FTD_VOLUMES = [100.0] * 7 + [200.0]


def run(closes, volumes=None, **kwargs):
    """ The prices and the signal values over them, the pair every function here takes. """
    ohlcv = bars(closes, volumes=volumes)
    return ohlcv, ftd(**kwargs)(ohlcv)


# --- the rally attempts ----------------------------------------------------------------

def test_a_confirmed_attempt_is_one_row_with_its_day0_day1_and_follow_through_day():
    ohlcv, sv = run(DECLINE_AND_RALLY, FTD_VOLUMES)

    episodes = extract_episodes(ohlcv, sv)

    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert (row.day0_date, row.day1_date, row.ftd_date) == \
        (ohlcv.index[3], ohlcv.index[4], ohlcv.index[7])
    assert (row.peak, row.peak_date, row.peak_age) == (101.0, ohlcv.index[0], 3)
    assert row.rally_low == 92.0
    assert row.decline_pct == pytest.approx(92.0 / 101.0 - 1)
    assert (row.outcome, row.end_date, row.rally_day) == ('FTD', ohlcv.index[7], 4)
    assert row.ftd_gain == pytest.approx(96.2 / 94.2 - 1)
    assert row.ftd_volume_ratio == pytest.approx(2.0)
    assert row.uptrend_end == 'OPEN'        # the data ends while the uptrend still holds


def test_an_undercut_attempt_ends_on_the_undercutting_day_and_the_next_one_opens_on_it():
    ohlcv, sv = run([100.0, 98.0, 96.0, 93.0, 94.0, 94.5, 91.5])

    episodes = extract_episodes(ohlcv, sv)

    assert episodes.outcome.tolist() == ['UNDERCUT', 'OPEN']
    assert episodes.end_date.tolist() == [ohlcv.index[6], None]
    assert episodes.day0_date.tolist() == [ohlcv.index[3], ohlcv.index[6]]
    assert episodes.day1_date.tolist()[1] is None       # day 1 never arrived


def test_an_attempt_that_runs_out_of_window_ends_in_a_timeout():
    closes = [100.0, 98.0, 96.0, 93.0, 94.0, 94.2, 94.4, 94.6, 94.8, 94.9]
    ohlcv, sv = run(closes, [100.0] * 9 + [200.0])

    episodes = extract_episodes(ohlcv, sv)

    assert episodes.outcome.tolist() == ['TIMEOUT']
    assert episodes.end_date.tolist() == [ohlcv.index[9]]


def test_a_day0_that_slides_to_a_lower_low_moves_the_day0_of_the_same_attempt():
    ohlcv, sv = run([100.0, 98.0, 96.0, 93.0, 92.0])

    episodes = extract_episodes(ohlcv, sv)

    assert len(episodes) == 1
    assert episodes.iloc[0].day0_date == ohlcv.index[4]
    assert episodes.iloc[0].rally_low == 91.0


def test_a_follow_through_day_the_index_closed_back_below_is_recorded_as_failed():
    # a Follow Through Day on day 7, undone on day 8, and a second one on day 12
    closes = DECLINE_AND_RALLY + [91.5, 92.5, 92.6, 92.7, 94.6]
    ohlcv, sv = run(closes, FTD_VOLUMES + [100.0] * 4 + [200.0])

    episodes = extract_episodes(ohlcv, sv)

    assert episodes.outcome.tolist() == ['FTD', 'FTD']
    assert episodes.iloc[0].uptrend_end == 'FTD_FAILED'
    assert episodes.iloc[0].days_to_failure == 1
    assert episodes.iloc[0].uptrend_end_date == ohlcv.index[8]
    assert episodes.iloc[1].ftd_date == ohlcv.index[12]


def test_an_uptrend_that_ends_in_a_new_correction_is_recorded_as_such():
    closes = DECLINE_AND_RALLY + [100.0, 105.0, 110.0, 108.0, 105.0, 102.0]
    ohlcv, sv = run(closes, FTD_VOLUMES + [100.0] * 6)

    episodes = extract_episodes(ohlcv, sv)

    assert episodes.iloc[0].uptrend_end == 'NEW_CORRECTION'
    assert pd.isna(episodes.iloc[0].days_to_failure)
    # the new correction opens the next attempt on the same day
    assert episodes.iloc[1].day0_date == episodes.iloc[0].uptrend_end_date


def test_the_attempt_summary_counts_the_outcomes_and_the_failures():
    closes = DECLINE_AND_RALLY + [91.5, 92.5, 92.6, 92.7, 94.6]
    ohlcv, sv = run(closes, FTD_VOLUMES + [100.0] * 4 + [200.0])
    episodes = extract_episodes(ohlcv, sv)

    summary = attempt_summary(episodes)

    assert summary['FTD'] == 2
    assert summary['FTD failed later'] == 1
    assert summary['attempts'] == 2
    assert failure_rate(episodes) == pytest.approx(0.5)


def test_extract_episodes_needs_a_signal_that_records_its_events():
    ohlcv = bars(DECLINE_AND_RALLY)
    sv = pd.DataFrame({'es': False}, index=ohlcv.index)

    with pytest.raises(ValueError, match="no 'event' column"):
        extract_episodes(ohlcv, sv)


# --- the rising edges ------------------------------------------------------------------

def test_pulse_dates_are_the_days_a_signal_turns_on():
    ohlcv = bars([100.0] * 6)
    sv = pd.DataFrame({'es': [False, True, True, False, True, False]}, index=ohlcv.index)

    assert pulse_dates(sv) == [ohlcv.index[1], ohlcv.index[4]]


# --- what the market did next ----------------------------------------------------------

@pytest.fixture
def rising():
    """ Closes 100, 101, ... 109, with the open half a point below the close. """
    return bars([100.0 + i for i in range(10)])


def test_forward_returns_are_measured_from_the_next_days_open(rising):
    rv = forward_outcomes(rising, [rising.index[0]], horizons=(2, 5))

    row = rv.iloc[0]
    assert (row.entry_date, row.entry_price) == (rising.index[1], 100.5)
    assert row.r2 == pytest.approx(102.0 / 100.5 - 1)
    assert row.r5 == pytest.approx(105.0 / 100.5 - 1)


def test_the_adverse_and_favourable_moves_are_intraday_over_the_longest_horizon(rising):
    rv = forward_outcomes(rising, [rising.index[0]], horizons=(2,))

    row = rv.iloc[0]
    assert row.mae == pytest.approx(100.0 / 100.5 - 1)   # the low of the entry day
    assert row.mfe == pytest.approx(103.0 / 100.5 - 1)   # the high two days on
    assert row.complete


def test_a_horizon_the_data_does_not_reach_is_missing_and_the_row_is_incomplete(rising):
    rv = forward_outcomes(rising, [rising.index[7]], horizons=(2, 5))

    assert rv.iloc[0].r2 == pytest.approx(109.0 / 107.5 - 1)
    assert pd.isna(rv.iloc[0].r5)
    assert not rv.iloc[0].complete


def test_a_day_with_nothing_left_to_trade_into_yields_no_row(rising):
    assert forward_outcomes(rising, [rising.index[-1]]).empty
    assert forward_outcomes(rising, [date(1990, 1, 1)]).empty


def test_the_forward_path_is_one_column_per_trading_day_after_the_signal(rising):
    rv = forward_paths(rising, [rising.index[0]], horizon=3)

    assert rv.columns.tolist() == [1, 2, 3]
    assert rv.iloc[0].tolist() == pytest.approx([101.0 / 100.5 - 1, 102.0 / 100.5 - 1,
                                                 103.0 / 100.5 - 1])


def test_the_naive_baseline_is_the_gain_and_the_volume_without_the_state_machine():
    # day 2 gains 3% on higher volume, day 3 gains 3% on lower volume, day 4 is flat
    ohlcv = bars([100.0, 100.0, 103.0, 106.09, 106.09],
                 volumes=[100.0, 100.0, 200.0, 150.0, 300.0])

    rv = baseline_dates(ohlcv, ftd())

    assert rv['all days'] == list(ohlcv.index)
    assert rv['naive FTD days'] == [ohlcv.index[2]]


def test_a_signal_without_a_gain_threshold_gets_no_naive_baseline(rising):
    class Bare:
        pass

    assert list(baseline_dates(rising, Bare())) == ['all days']


def test_the_forward_table_has_a_row_per_named_set_of_days(rising):
    rv = forward_table(rising, {'signal': [rising.index[0]],
                                'all days': list(rising.index)}, horizons=(2,))

    assert rv.index.tolist() == ['signal', 'all days']
    assert rv.loc['signal', 'days'] == 1
    assert rv.loc['signal', 'mean r2'] == pytest.approx(102.0 / 100.5 - 1)
    assert rv.loc['signal', 'positive at 2d'] == 1.0


# --- the reference dates ---------------------------------------------------------------

def reference(day, kind='reference', **kwargs):
    rv = {'date': day, 'episode': 'test', 'day1': None, 'kind': kind,
          'confidence': None, 'notes': None}
    rv.update(kwargs)
    return rv


def test_a_reference_within_the_tolerance_is_a_hit_and_carries_the_gap():
    ohlcv, sv = run(DECLINE_AND_RALLY, FTD_VOLUMES)       # the pulse is on day 7

    rv = match_references(sv, [reference(ohlcv.index[5])], tolerance=3)

    assert rv.iloc[0].hit
    assert (rv.iloc[0].detected, rv.iloc[0].gap) == (ohlcv.index[7], 2)
    assert rv.iloc[0].in_data


def test_a_reference_outside_the_tolerance_is_a_miss_that_carries_the_state_and_the_blocks():
    ohlcv, sv = run(DECLINE_AND_RALLY, FTD_VOLUMES)

    rv = match_references(sv, [reference(ohlcv.index[1])], tolerance=2)

    assert not rv.iloc[0].hit
    assert rv.iloc[0].state == FTDSignal.WATCHING
    assert 'PEAK_AGE' in rv.iloc[0].day0_blocks


def test_a_rally_day_blocked_from_being_a_follow_through_day_shows_in_the_window():
    ohlcv, sv = run(DECLINE_AND_RALLY)      # flat volume, so day 7 is blocked

    rv = match_references(sv, [reference(ohlcv.index[7])], tolerance=2)

    assert not rv.iloc[0].hit               # no pulse anywhere in the data
    assert rv.iloc[0].ftd_blocks == 'VOLUME'


def test_a_reference_the_data_does_not_cover_is_neither_a_hit_nor_a_miss():
    _, sv = run(DECLINE_AND_RALLY, FTD_VOLUMES)

    rv = match_references(sv, [reference(date(1990, 1, 1))])

    assert not rv.iloc[0].in_data
    assert not rv.iloc[0].hit
    assert pd.isna(rv.iloc[0].gap)


def test_the_match_table_keeps_the_reference_fields_for_the_report():
    ohlcv, sv = run(DECLINE_AND_RALLY, FTD_VOLUMES)
    ref = reference(ohlcv.index[7], kind='candidate', episode='Covid crash',
                    confidence='medium', notes='off the low')

    rv = match_references(sv, [ref])

    assert rv.iloc[0].episode == 'Covid crash'
    assert (rv.iloc[0].kind, rv.iloc[0].confidence, rv.iloc[0].notes) == \
        ('candidate', 'medium', 'off the low')


# --- the reference file ----------------------------------------------------------------

def write_references(tmp_path, entries):
    path = tmp_path / "refs.yaml"
    path.write_text(yaml.safe_dump({'follow_through_days': entries}, sort_keys=False))
    return str(path)


def test_the_reference_file_is_read_as_dates(tmp_path):
    path = write_references(tmp_path, [
        {'date': date(2020, 4, 2), 'episode': 'Covid crash', 'day1': date(2020, 3, 23),
         'kind': 'reference', 'confidence': None, 'notes': 'Covid',
         'unreachable': {'reason': 'because', 'low': date(2020, 3, 23), 'block': 'PEAK_AGE',
                         'detected_instead': date(2020, 3, 26)}}])

    rv = load_references(path)

    assert rv[0]['date'] == date(2020, 4, 2)
    assert rv[0]['day1'] == date(2020, 3, 23)
    assert rv[0]['unreachable']['low'] == date(2020, 3, 23)
    assert rv[0]['unreachable']['detected_instead'] == date(2020, 3, 26)


def test_the_reference_file_must_name_its_kinds(tmp_path):
    path = write_references(tmp_path, [{'date': date(2020, 4, 2), 'kind': 'guess'}])

    with pytest.raises(ValueError, match="'reference' or 'candidate'"):
        load_references(path)


def test_the_reference_file_must_hold_a_follow_through_days_list(tmp_path):
    path = tmp_path / "refs.yaml"
    path.write_text("dates: []\n")

    with pytest.raises(ValueError, match="follow_through_days"):
        load_references(str(path))


# --- the parameter grid ----------------------------------------------------------------

def test_the_sensitivity_grid_runs_the_base_setting_and_one_parameter_away_from_it():
    ohlcv = bars(DECLINE_AND_RALLY, volumes=FTD_VOLUMES)
    base = dict(index=None, min_decline=0.08, min_peak_age_days=3, day0_window=3,
                ftd_min_gain=0.02, ftd_min_days=4, ftd_max_days=6)

    rv = sensitivity(ohlcv, {'ftd_min_gain': [0.02, 0.05]},
                     [reference(ohlcv.index[7])], base_params=base, horizon=2)

    assert rv.setting.tolist() == ['base', 'ftd_min_gain=0.05']   # the base value is not re-run
    assert rv.pulses.tolist() == [1, 0]
    assert rv.references_hit.tolist() == [1, 0]
    assert rv.references.tolist() == [1, 1]
    assert 'median r2' in rv.columns


def test_the_sensitivity_grid_counts_only_the_references_not_the_candidates():
    ohlcv = bars(DECLINE_AND_RALLY, volumes=FTD_VOLUMES)
    refs = [reference(ohlcv.index[7]), reference(ohlcv.index[7], kind='candidate')]

    rv = sensitivity(ohlcv, {}, refs,
                     base_params=dict(index=None, min_decline=0.08, min_peak_age_days=3,
                                      day0_window=3, ftd_min_gain=0.02, ftd_min_days=4,
                                      ftd_max_days=6))

    assert rv.references.tolist() == [1]
