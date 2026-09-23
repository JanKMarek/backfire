"""
    Analysis of what a signal did and of what the market did next.

    Pure pandas: nothing here draws or writes anything, so the same functions serve
    backfire/ftd_signal_verification_report.py and a notebook. Two questions are answered separately:

      - does the signal fire where it should? load_ground_truth reads a set of dates the
        signal is expected to fire on - the turnaround points of docs/turnaround_points.csv
        or the IBD calls of docs/ftd_reference.yaml - match_ground_truth lines the firings up
        against them with the tolerance docs/SIGNALS.md gives, and verification_stats turns
        the result into precision, recall and F1. extract_episodes reads the diagnostics the
        signal records day by day (see FTDSignal) so that every miss can be explained.
      - does a firing mark a real turnaround? forward_outcomes, forward_paths and
        baseline_dates measure what happened after each firing and against what;
        ftd_dates_by_fate and path_table split that by whether the Follow Through Day held.

    Only extract_episodes and the miss reasons are specific to the Follow Through Day signal -
    they read the 'event' and block columns of FTDSignal. Everything else works off the
    rising edges of any signal's 'es'.
"""

import math
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml

from .signals import FTDSignal

# the ground truth files, in docs/ next to this package
_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
REFERENCE_PATH = os.path.join(_DOCS, "ftd_reference.yaml")
TURNAROUND_PATH = os.path.join(_DOCS, "turnaround_points.csv")
# the ground truth a report can be scored against, by name - see load_ground_truth
GROUND_TRUTH = {'turnarounds': TURNAROUND_PATH, 'ibd': REFERENCE_PATH}

# the matching tolerance docs/SIGNALS.md gives: a firing counts for a ground truth date when
# it is at most EARLY_DAYS trading days before it or LATE_DAYS after it
EARLY_DAYS, LATE_DAYS = 1, 3
# a firing this close to a ground truth date it did not match is reported as the reason
NEAR_MISS_DAYS = 10

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


def load_turnaround_points(path=None):
    """
        Reads the turnaround points: the dates on which, with hindsight, the market turned -
        not IBD calls, but what the signal is meant to detect (docs/SIGNALS.md, 'Signal
        verification').
    :param path: the CSV file, docs/turnaround_points.csv by default; '#' lines are comments
                 and the columns are 'Date' and 'Notes'
    :return: list of dicts in the shape load_references returns, with kind 'turnaround'
    """
    path = path or TURNAROUND_PATH
    frame = pd.read_csv(path, comment='#', skip_blank_lines=True)
    if 'Date' not in frame.columns:
        raise ValueError(f"'{path}' must have a 'Date' column.")
    rv = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.Date):
            continue
        notes = getattr(row, 'Notes', None)
        rv.append({'date': as_date(row.Date), 'episode': None, 'day1': None,
                   'kind': 'turnaround', 'confidence': None,
                   'notes': None if notes is None or pd.isna(notes) else str(notes)})
    return rv


def ground_truth_path(source='turnarounds'):
    """ The file behind a ground truth name ('turnarounds', 'ibd'), or the path given. """
    return GROUND_TRUTH.get(source, source)


def load_ground_truth(source='turnarounds'):
    """
        Reads the dates a report scores the signal against.

        'turnarounds' is docs/turnaround_points.csv: does the signal meet its intent? 'ibd' is
        docs/ftd_reference.yaml: does the signal fire when IBD called a Follow Through Day?
        Only the 'reference' entries of that file are ground truth - its candidates are
        recalled but unconfirmed dates, not expectations. A path is read as one or the other
        by its extension.

    :param source: 'turnarounds', 'ibd', or the path of a .csv or .yaml file
    :return: list of dicts with 'date' (a datetime.date), 'episode', 'day1', 'kind',
             'confidence' and 'notes', in date order
    """
    path = ground_truth_path(source)
    lower = str(path).lower()
    if lower.endswith('.csv'):
        rv = load_turnaround_points(path)
    elif lower.endswith(('.yaml', '.yml')):
        rv = [ref for ref in load_references(path) if ref['kind'] == 'reference']
    else:
        raise ValueError(f"Ground truth must be 'turnarounds', 'ibd' or a .csv/.yaml file, "
                         f"got {source!r}.")
    return sorted(rv, key=lambda entry: entry['date'])


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


def ftd_dates_by_fate(episodes):
    """
        The Follow Through Days grouped by what became of the uptrend each one started: 'all'
        of them, the 'successful' ones - the uptrend held until a new correction re-armed the
        signal - and the 'failed' ones the index closed back below the rally low. An uptrend
        still open at the end of the data is neither, and counts only under 'all'.
    :param episodes: the rally attempts, as extract_episodes returns them
    :return: dict of name -> list of days
    """
    ftds = episodes[episodes.outcome == 'FTD']
    return {'all': list(ftds.ftd_date),
            'successful': list(ftds.ftd_date[ftds.uptrend_end == 'NEW_CORRECTION']),
            'failed': list(ftds.ftd_date[ftds.uptrend_end == 'FTD_FAILED'])}


def path_table(paths, groups, days=(5, 10, 20, 40, 60)):
    """
        The median of the forward paths at the given days, one row per named set of days.
    :param paths: the forward paths, as forward_paths returns them
    :param groups: dict of name -> list of days, e.g. ftd_dates_by_fate(...)
    :param days: trading days after the signal day; a day past the horizon of the paths is
                 left out
    :return: DataFrame with the 'count' of paths and 'median r{d}' per day
    """
    days = [d for d in days if d in paths.columns]
    rows, index = [], []
    for name, dates in groups.items():
        group = paths[paths.index.isin(dates)]
        row = {'count': len(group)}
        for d in days:
            row[f'median r{d}'] = group[d].median()
        rows.append(row)
        index.append(name)
    return pd.DataFrame(rows, index=index, columns=['count'] + [f'median r{d}' for d in days])


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

MATCH_COLUMNS = ['date', 'episode', 'kind', 'confidence', 'day1', 'in_data', 'hit',
                 'detected', 'gap', 'nearest', 'nearest_gap', 'state', 'day0_blocks',
                 'ftd_blocks', 'reason', 'notes']


def match_ground_truth(sv, truth, early=EARLY_DAYS, late=LATE_DAYS):
    """
        Lines the signal's firings up against the ground truth dates.

        A date is hit when a firing falls at most 'early' trading days before it or 'late'
        after it - the signal may confirm a turnaround a few days off the date recorded for
        it (docs/SIGNALS.md, 'Signal verification'). Each firing counts for at most one date:
        the dates are taken in order and each claims the nearest unclaimed firing in its
        window, so the hits are the true positives and every other firing is a false
        positive. A date the data does not cover is neither a hit nor a miss and is marked
        in_data False.

        A miss carries a reason: the nearest firing when there is one close by, otherwise
        what the machine was doing on the date - the uptrend it was still in, the clause that
        stopped the day 0, or the clause that stopped the follow through - read off the
        state and the block marks within the window.

    :param sv: the signal values
    :param truth: the ground truth, as load_ground_truth returns it
    :param early: how many trading days before the date a firing still counts
    :param late: how many trading days after the date a firing still counts
    :return: DataFrame with MATCH_COLUMNS, one row per ground truth date, in date order.
             'detected' and 'gap' are the matched firing and its distance in trading days
             (positive when it fired later); 'nearest' and 'nearest_gap' the closest firing
             whether or not it matched.
    """
    bars = sv.index
    pulse_bars = [int(bars.searchsorted(day, side='left')) for day in pulse_dates(sv)]
    claimed = {}                                    # pulse bar -> the date that took it

    rows = []
    for entry in sorted(truth, key=lambda e: as_date(e['date'])):
        day = as_date(entry['date'])
        row = {'date': day, 'episode': entry.get('episode'), 'kind': entry.get('kind'),
               'confidence': entry.get('confidence'), 'day1': as_date(entry.get('day1')),
               'notes': entry.get('notes'), 'in_data': False, 'hit': False, 'detected': None,
               'gap': np.nan, 'nearest': None, 'nearest_gap': np.nan, 'state': None,
               'day0_blocks': '', 'ftd_blocks': '', 'reason': 'outside the data'}
        if len(bars) and bars[0] <= day <= bars[-1]:
            # the date itself, or the next trading day when the underlying did not trade
            target = min(int(bars.searchsorted(day, side='left')), len(bars) - 1)
            lo, hi = max(0, target - early), min(len(bars) - 1, target + late)
            row.update(in_data=True, reason='', state=sv.state.iloc[target],
                       day0_blocks=_distinct(sv.day0_block.iloc[lo:hi + 1]),
                       ftd_blocks=_distinct(sv.ftd_block.iloc[lo:hi + 1]))
            if pulse_bars:
                nearest = min(pulse_bars, key=lambda bar: (abs(bar - target), bar))
                row.update(nearest=bars[nearest], nearest_gap=nearest - target)
                free = [bar for bar in pulse_bars if lo <= bar <= hi and bar not in claimed]
                if free:
                    best = min(free, key=lambda bar: (abs(bar - target), bar))
                    claimed[best] = day
                    row.update(hit=True, detected=bars[best], gap=best - target)
            if not row['hit']:
                row['reason'] = _miss_reason(sv, target, row, claimed)
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _miss_reason(sv, target, row, claimed):
    """ Why nothing fired for a ground truth date, in words - see match_ground_truth. """
    gap = row['nearest_gap']
    if row['nearest'] is not None and abs(gap) <= NEAR_MISS_DAYS:
        when = f"{abs(gap):.0f}d {'late' if gap > 0 else 'early'}" if gap else "on the day"
        rv = f"fired {when} ({row['nearest']})"
        taken = claimed.get(int(sv.index.searchsorted(row['nearest'], side='left')))
        return rv + (f", already matched to {taken}" if taken else "")

    state = row['state']
    if state == FTDSignal.UPTREND:
        return f"still in the uptrend of the {sv.ftd_date.iloc[target]} follow through day"
    if state == FTDSignal.WATCHING:
        return (f"no day 0: {row['day0_blocks']}" if row['day0_blocks']
                else "no new low to test as a day 0")
    if state == FTDSignal.DAY0:
        return "day 0 made, waiting for day 1"
    rally_day = sv.rally_day.iloc[target]
    where = "in the rally attempt" + ("" if pd.isna(rally_day) else f" (day {rally_day:.0f})")
    if row['ftd_blocks']:
        return f"{where}: follow through blocked by {row['ftd_blocks']}"
    return f"{where}: no day gained the minimum"


def false_positive_dates(matches, pulses):
    """ The firings no ground truth date claimed, in time order. """
    detected = {as_date(day) for day in matches.detected[matches.hit]}
    return [day for day in pulses if as_date(day) not in detected]


def verification_stats(matches, pulses, trading_days):
    """
        The verification statistics of docs/SIGNALS.md: a positive is a firing, a true
        positive one that matched a ground truth date, and a ground truth date counts only
        when the data covers it.
    :param matches: the match table, as match_ground_truth returns it
    :param pulses: the firing dates, as pulse_dates returns them
    :param trading_days: the length of the simulation range
    :return: dict with 'ground_truth' (in the data), 'ground_truth_total', 'positives',
             'true_positives', 'false_positives', 'trading_days', 'precision', 'recall' and
             'f1' - the rates NaN where undefined
    """
    scored = matches[matches.in_data.astype(bool)]
    tp, positives = int(scored.hit.sum()), len(pulses)
    precision = tp / positives if positives else np.nan
    recall = tp / len(scored) if len(scored) else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if positives and len(scored) and precision + recall else np.nan)
    return {'ground_truth': len(scored), 'ground_truth_total': len(matches),
            'positives': positives, 'true_positives': tp, 'false_positives': positives - tp,
            'trading_days': trading_days, 'precision': precision, 'recall': recall, 'f1': f1}


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

SENSITIVITY_COLUMNS = ['setting', 'pulses', 'hits', 'ground_truth', 'precision', 'recall',
                       'failure_rate']


def sensitivity(ohlcv, grid, truth, base_params=None, factory=FTDSignal, env=None,
                early=EARLY_DAYS, late=LATE_DAYS, horizon=20):
    """
        Re-runs the signal over a small grid of parameter settings, one parameter away from
        the base setting at a time, and reports what each setting buys.

    :param ohlcv: the underlying's prices
    :param grid: mapping of parameter name -> the values to try, e.g.
                 {'min_decline': [0.08, 0.10], 'min_peak_age_days': [15, 20, 25]}
    :param truth: the ground truth, as load_ground_truth returns it
    :param base_params: the constructor arguments of the signal the report is about; the
                        factory's own defaults when None
    :param factory: builds the signal from the parameters, FTDSignal by default
    :param env: the Environment the signal loads its index prices from, when it needs one
    :param early, late: the matching tolerance, as match_ground_truth takes it
    :param horizon: the forward horizon whose median return is reported
    :return: DataFrame, one row per setting, the base setting first
    """
    base_params = dict(base_params or {})
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
        pulses = pulse_dates(sv)
        episodes = extract_episodes(ohlcv, sv)
        stats = verification_stats(match_ground_truth(sv, truth, early, late), pulses, len(sv))
        outcomes = forward_outcomes(ohlcv, pulses, (horizon,))
        rows.append({'setting': label, 'pulses': stats['positives'],
                     'hits': stats['true_positives'], 'ground_truth': stats['ground_truth'],
                     'precision': stats['precision'], 'recall': stats['recall'],
                     'failure_rate': failure_rate(episodes),
                     median_column: outcomes[f'r{horizon}'].median()})
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS + [median_column])
