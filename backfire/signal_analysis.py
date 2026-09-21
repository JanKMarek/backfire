"""
    Analysis of what a signal did and of what the market did next.

    Pure pandas: nothing here draws or writes anything, so the same functions serve
    backfire/report_signal.py and a notebook. Two questions are answered separately:

      - does the signal follow its own rules? extract_episodes and match_references read the
        diagnostics the signal records day by day (see FTDSignal) and line them up against the
        reference dates in docs/ftd_reference.yaml.
      - does a firing mark a real turnaround? forward_outcomes, forward_paths and
        baseline_dates measure what happened after each firing and against what.

    Only extract_episodes is specific to the Follow Through Day signal - it reads the 'event'
    column of FTDSignal. Everything else works off the rising edges of any signal's 'es'.
"""

import math
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml

from .signals import FTDSignal

# docs/ftd_reference.yaml, next to this package
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "docs", "ftd_reference.yaml")

# how a rally attempt ended, in the order the report lists them
OUTCOMES = ('FTD', 'UNDERCUT', 'TIMEOUT', 'OPEN')


# ----------------------------------------------------------------------------------
# Reading the day by day diagnostics
# ----------------------------------------------------------------------------------

def _tokens(value):
    """ The marks of one day - 'UNDERCUT+DAY0' - as a list, empty when there are none. """
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == '':
        return []
    return str(value).split('+')


def as_date(value):
    """ A YAML or pandas date as a datetime.date, so that it compares with the price index. """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def pulse_dates(sv):
    """
        The days a signal turns on - the rising edges of 'es'. A one day pulse such as the
        Follow Through Day has one per firing; a signal that stays on has one per block.
    :param sv: the signal values, as Signal.__call__ returns them
    :return: list of days
    """
    es = sv.es.fillna(False).astype(bool)
    return list(sv.index[es & ~es.shift(1, fill_value=False)])


def load_references(path=None):
    """
        Reads the reference and candidate Follow Through Days.
    :param path: the YAML file, docs/ftd_reference.yaml by default
    :return: list of dicts with 'date' (a datetime.date), 'episode', 'day1', 'kind',
             'confidence', 'notes' and, for a reference the rules cannot reach,
             'unreachable' - see the header of the file
    """
    with open(path or REFERENCE_PATH) as f:
        conf = yaml.safe_load(f)
    if not isinstance(conf, dict) or 'follow_through_days' not in conf:
        raise ValueError(f"'{path or REFERENCE_PATH}' must hold a 'follow_through_days' list.")

    rv = []
    for entry in conf['follow_through_days']:
        entry = dict(entry)
        entry['date'] = as_date(entry.get('date'))
        entry['day1'] = as_date(entry.get('day1'))
        if entry['date'] is None:
            raise ValueError(f"A reference entry has no date: {entry!r}")
        if entry.get('kind') not in ('reference', 'candidate'):
            raise ValueError(f"{entry['date']}: kind must be 'reference' or 'candidate', "
                             f"got {entry.get('kind')!r}")
        unreachable = entry.get('unreachable')
        if unreachable:
            unreachable = dict(unreachable)
            unreachable['low'] = as_date(unreachable.get('low'))
            unreachable['detected_instead'] = as_date(unreachable.get('detected_instead'))
            entry['unreachable'] = unreachable
        rv.append(entry)
    return rv


# ----------------------------------------------------------------------------------
# The rally attempts
# ----------------------------------------------------------------------------------

EPISODE_COLUMNS = ['day0_date', 'day1_date', 'peak', 'peak_date', 'peak_age', 'rally_low',
                   'decline_pct', 'outcome', 'end_date', 'ftd_date', 'rally_day', 'ftd_gain',
                   'ftd_volume_ratio', 'uptrend_end', 'uptrend_end_date', 'days_to_failure']


def extract_episodes(ohlcv, sv):
    """
        One row per rally attempt, read off the 'event' column of FTDSignal.

        An attempt opens on a day 0 and ends in a Follow Through Day, an undercut or a
        timeout; one still under way at the end of the data ends 'OPEN'. An attempt confirmed
        by a Follow Through Day also carries what became of the uptrend it started: the
        Follow Through Day failed ('FTD_FAILED', with the trading days it took), a new
        correction re-armed the signal ('NEW_CORRECTION'), or the uptrend is still 'OPEN'.

    :param ohlcv: the underlying's prices, on the same calendar as sv
    :param sv: the signal values of an FTDSignal over them
    :return: DataFrame with EPISODE_COLUMNS, one row per attempt, in time order
    """
    if 'event' not in sv.columns:
        raise ValueError("The signal values carry no 'event' column: extract_episodes needs "
                         "an FTDSignal (see docs/SIGNALS.md).")

    days = list(sv.index)
    events = sv['event'].fillna('').tolist()
    peak, peak_date = sv.peak.tolist(), sv.peak_date.tolist()
    rally_low, decline_pct = sv.rally_low.tolist(), sv.decline_pct.tolist()
    rally_day = sv.rally_day.tolist()
    close = sv.index_close.tolist()
    volume = ohlcv.V.reindex(sv.index).tolist()

    def record_day0(attempt, i):
        """ The day 0 of an attempt; a day 0 that slides to a lower low overwrites it. """
        age = None
        if peak_date[i] is not None and not pd.isna(peak_date[i]):
            age = i - int(sv.index.searchsorted(peak_date[i], side='left'))
        attempt.update(day0_date=days[i], peak=peak[i], peak_date=peak_date[i], peak_age=age,
                       rally_low=rally_low[i], decline_pct=decline_pct[i])

    rows, attempt, uptrend, uptrend_i = [], None, None, None
    for i, day in enumerate(days):
        for event in _tokens(events[i]):
            if event in ('UNDERCUT', 'TIMEOUT') and attempt is not None:
                attempt['outcome'], attempt['end_date'] = event, day
                attempt = None
            elif event == 'FTD' and attempt is not None:
                attempt.update(outcome='FTD', end_date=day, ftd_date=day,
                               rally_day=rally_day[i], uptrend_end='OPEN',
                               ftd_gain=_change(close, i), ftd_volume_ratio=_ratio(volume, i))
                uptrend, uptrend_i, attempt = attempt, i, None
            elif event in ('FTD_FAILED', 'NEW_CORRECTION') and uptrend is not None:
                uptrend['uptrend_end'], uptrend['uptrend_end_date'] = event, day
                if event == 'FTD_FAILED':
                    uptrend['days_to_failure'] = i - uptrend_i
                uptrend = None
            elif event == 'DAY0':
                attempt = dict.fromkeys(EPISODE_COLUMNS)
                attempt['outcome'] = 'OPEN'
                rows.append(attempt)
                record_day0(attempt, i)
            elif event == 'DAY0_LOWERED' and attempt is not None:
                record_day0(attempt, i)
            elif event == 'DAY1' and attempt is not None:
                attempt['day1_date'] = day

    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def _ratio(values, i):
    """ The day's value over the previous day's; None when there is no previous day. """
    if i == 0 or values[i - 1] is None or pd.isna(values[i - 1]) or values[i - 1] <= 0:
        return None
    return values[i] / values[i - 1]


def _change(values, i):
    """ The day's value over the previous day's, as a return. """
    ratio = _ratio(values, i)
    return None if ratio is None else ratio - 1.0


def attempt_summary(episodes):
    """
        How the rally attempts ended, and how many of the confirmed ones did not hold.
    :return: pandas Series of counts
    """
    rv = {outcome: int((episodes.outcome == outcome).sum()) for outcome in OUTCOMES}
    ftds = episodes[episodes.outcome == 'FTD']
    rv['FTD failed later'] = int((ftds.uptrend_end == 'FTD_FAILED').sum())
    rv['FTD ended in a new correction'] = int((ftds.uptrend_end == 'NEW_CORRECTION').sum())
    rv['attempts'] = len(episodes)
    return pd.Series(rv, name='attempts')


def failure_rate(episodes):
    """ The share of Follow Through Days that the index later closed back below. """
    ftds = episodes[episodes.outcome == 'FTD']
    if ftds.empty:
        return float('nan')
    return float((ftds.uptrend_end == 'FTD_FAILED').sum()) / len(ftds)


# ----------------------------------------------------------------------------------
# What the market did next
# ----------------------------------------------------------------------------------

def forward_outcomes(ohlcv, dates, horizons=(5, 20, 60)):
    """
        What the market did after each of the given days.

        The position is opened at the next day's open, the way SignalDrivenStrategy executes
        a signal, so a day that has no next day yields nothing. The return at horizon h is
        measured from that open to the close h trading days after the signal day; the largest
        adverse and favourable moves are measured intraday over the longest horizon.

    :param ohlcv: the underlying's prices
    :param dates: the days to measure from
    :param horizons: the horizons, in trading days
    :return: DataFrame indexed by the signal day, with 'entry_date', 'entry_price',
             'r{h}' per horizon, 'mae', 'mfe' and 'complete' - False when the data ran out
             before the longest horizon did
    """
    window = max(horizons)
    o, h, l, c = (ohlcv.O.to_numpy(), ohlcv.H.to_numpy(), ohlcv.L.to_numpy(),
                  ohlcv.C.to_numpy())
    n = len(ohlcv)

    index, rows = [], []
    for day in dates:
        i = int(ohlcv.index.searchsorted(day, side='left'))
        if i >= n or ohlcv.index[i] != day or i + 1 >= n:
            continue                    # not a trading day, or nothing left to trade into
        entry = o[i + 1]
        row = {'entry_date': ohlcv.index[i + 1], 'entry_price': entry}
        for horizon in horizons:
            row[f'r{horizon}'] = c[i + horizon] / entry - 1.0 if i + horizon < n else np.nan
        last = min(i + window, n - 1)
        row['mae'] = l[i + 1:last + 1].min() / entry - 1.0
        row['mfe'] = h[i + 1:last + 1].max() / entry - 1.0
        row['complete'] = i + window < n
        index.append(day)
        rows.append(row)

    columns = (['entry_date', 'entry_price'] + [f'r{h}' for h in horizons]
               + ['mae', 'mfe', 'complete'])
    return pd.DataFrame(rows, index=pd.Index(index, name='date'), columns=columns)


def forward_paths(ohlcv, dates, horizon=60):
    """
        The path of the return after each of the given days, for the average path chart:
        one row per day, one column per trading day 1..horizon after it, measured from the
        next day's open the way forward_outcomes measures.
    :return: DataFrame indexed by the signal day, columns 1..horizon
    """
    o, c = ohlcv.O.to_numpy(), ohlcv.C.to_numpy()
    n = len(ohlcv)

    index, rows = [], []
    for day in dates:
        i = int(ohlcv.index.searchsorted(day, side='left'))
        if i >= n or ohlcv.index[i] != day or i + 1 >= n:
            continue
        entry = o[i + 1]
        rows.append([c[i + k] / entry - 1.0 if i + k < n else np.nan
                     for k in range(1, horizon + 1)])
        index.append(day)
    return pd.DataFrame(rows, index=pd.Index(index, name='date'),
                        columns=range(1, horizon + 1))


def baseline_dates(ohlcv, signal):
    """
        The comparison sets a signal's own days are judged against.

        'all days' is the market itself. 'naive FTD days' applies the Follow Through Day's
        last two clauses - the gain and the higher volume - to every day, with none of the
        correction, day 0 or day count logic around them, and so says whether that logic adds
        anything. It is only offered for a signal that has an ftd_min_gain.

    :return: dict of name -> list of days
    """
    rv = {'all days': list(ohlcv.index)}
    threshold = getattr(signal, 'ftd_min_gain', None)
    if threshold is not None:
        gain = ohlcv.C / ohlcv.C.shift(1) - 1.0
        naive = (gain >= threshold) & (ohlcv.V > ohlcv.V.shift(1))
        rv['naive FTD days'] = list(ohlcv.index[naive.fillna(False)])
    return rv


def forward_table(ohlcv, groups, horizons=(5, 20, 60)):
    """
        One row per named set of days, comparing what followed them.
    :param groups: dict of name -> list of days, e.g. the signal's own days next to
                   baseline_dates(...)
    :return: DataFrame with the count, the mean and median return per horizon, the share of
             days that were positive at the longest horizon, and the median mae/mfe
    """
    longest = max(horizons)
    rows, index = [], []
    for name, dates in groups.items():
        outcomes = forward_outcomes(ohlcv, dates, horizons)
        row = {'days': len(outcomes)}
        for horizon in horizons:
            row[f'mean r{horizon}'] = outcomes[f'r{horizon}'].mean()
            row[f'median r{horizon}'] = outcomes[f'r{horizon}'].median()
        row[f'positive at {longest}d'] = (outcomes[f'r{longest}'] > 0).sum() / max(
            outcomes[f'r{longest}'].notna().sum(), 1)
        row['median mae'] = outcomes.mae.median()
        row['median mfe'] = outcomes.mfe.median()
        rows.append(row)
        index.append(name)
    return pd.DataFrame(rows, index=index)


# ----------------------------------------------------------------------------------
# The reference dates
# ----------------------------------------------------------------------------------

MATCH_COLUMNS = ['date', 'episode', 'kind', 'confidence', 'day1', 'detected', 'gap', 'hit',
                 'in_data', 'state', 'day0_blocks', 'ftd_blocks', 'notes']


def match_references(sv, refs, tolerance=10):
    """
        Lines the signal's firings up against the reference and candidate dates.

        A date is hit when a firing is within 'tolerance' trading days of it - the reference
        calls are discretionary and made on the Nasdaq Composite, so an exact match is not
        expected (see docs/SIGNALS.md). A date the data does not cover is neither a hit nor a
        miss and is marked in_data False.

        For a miss, 'state' is what the machine was doing on the date and the block columns
        collect the reasons it recorded within the tolerance window - between them they say
        why nothing fired.

    :param sv: the signal values
    :param refs: the reference list, as load_references returns it
    :param tolerance: how many trading days away a firing still counts as a hit
    :return: DataFrame with MATCH_COLUMNS, in the order of refs
    """
    bars = sv.index
    pulse_bars = [int(bars.searchsorted(day, side='left')) for day in pulse_dates(sv)]

    rows = []
    for ref in refs:
        day = as_date(ref['date'])
        row = {'date': day, 'episode': ref.get('episode'), 'kind': ref.get('kind'),
               'confidence': ref.get('confidence'), 'day1': as_date(ref.get('day1')),
               'notes': ref.get('notes'), 'detected': None, 'gap': np.nan, 'hit': False,
               'in_data': False, 'state': None, 'day0_blocks': '', 'ftd_blocks': ''}
        if len(bars) and bars[0] <= day <= bars[-1]:
            target = min(int(bars.searchsorted(day, side='left')), len(bars) - 1)
            lo, hi = max(0, target - tolerance), min(len(bars), target + tolerance + 1)
            # the state and the blocks are worth having whether or not anything fired -
            # a signal that never fires is the case that most needs explaining
            row.update(in_data=True, state=sv.state.iloc[target],
                       day0_blocks=_distinct(sv.day0_block.iloc[lo:hi]),
                       ftd_blocks=_distinct(sv.ftd_block.iloc[lo:hi]))
            if pulse_bars:
                gap = min((bar - target for bar in pulse_bars), key=abs)
                row.update(detected=bars[target + gap], gap=gap,
                           hit=abs(gap) <= tolerance)
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _distinct(marks):
    """ The distinct tokens of a window of mark values, in the order they first appear. """
    rv = []
    for value in marks:
        for token in _tokens(value):
            if token not in rv:
                rv.append(token)
    return ', '.join(rv)


# ----------------------------------------------------------------------------------
# Sensitivity to the parameters
# ----------------------------------------------------------------------------------

SENSITIVITY_COLUMNS = ['setting', 'pulses', 'references_hit', 'references', 'failure_rate']


def sensitivity(ohlcv, grid, refs, base_params=None, factory=FTDSignal, env=None,
                tolerance=10, horizon=20):
    """
        Re-runs the signal over a small grid of parameter settings, one parameter away from
        the base setting at a time, and reports what each setting buys.

    :param ohlcv: the underlying's prices
    :param grid: mapping of parameter name -> the values to try, e.g.
                 {'min_decline': [0.08, 0.10], 'min_peak_age_days': [15, 20, 25]}
    :param refs: the reference list, as load_references returns it; only the references
                 count towards 'references_hit', the candidates are not expectations
    :param base_params: the constructor arguments of the signal the report is about; the
                        factory's own defaults when None
    :param factory: builds the signal from the parameters, FTDSignal by default
    :param env: the Environment the signal loads its index prices from, when it needs one
    :param horizon: the forward horizon whose median return is reported
    :return: DataFrame, one row per setting, the base setting first
    """
    base_params = dict(base_params or {})
    references = [ref for ref in refs if ref.get('kind') == 'reference']
    median_column = f'median r{horizon}'

    settings = [('base', base_params)]
    for name, values in (grid or {}).items():
        for value in values:
            if base_params.get(name) != value:
                settings.append((f"{name}={value}", {**base_params, name: value}))

    rows = []
    for label, params in settings:
        signal = factory(**params)
        if env is not None:
            signal.bind(env)
        sv = signal(ohlcv)
        episodes = extract_episodes(ohlcv, sv)
        matched = match_references(sv, references, tolerance)
        outcomes = forward_outcomes(ohlcv, pulse_dates(sv), (horizon,))
        rows.append({'setting': label, 'pulses': int(sv.es.sum()),
                     'references_hit': int(matched.hit.sum()), 'references': len(references),
                     'failure_rate': failure_rate(episodes),
                     median_column: outcomes[f'r{horizon}'].median()})
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS + [median_column])
