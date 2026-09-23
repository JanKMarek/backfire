"""
    Command line driver for the FTD signal verification report.

    Runs one signal over one underlying, scores its firings against a set of ground truth
    dates and writes a static HTML report - docs/SIGNALS.md, 'Signal verification', is the
    specification.

      uv run python backfire/ftd_signal_verification_report.py --underlying QQQ --start_date 1999-03-10 \
          --signal strategies/signals/ftd.yaml --ground_truth turnarounds \
          --out out/ftd_verification

    The signal file names the class and its constructor arguments, the same catalog
    backtest.py builds a strategy from:

      signal:
        name: FTDSignal
        index: null
        min_decline: 0.08
        min_peak_age_days: 20
        day0_window: 5
        ftd_min_gain: 0.0125
        ftd_min_days: 4
        ftd_max_days: 25

    Any entry can be overridden per run with --signal.<name>=<value>, read as YAML so types
    are kept; with no --signal file the overrides define the signal on their own.

    The ground truth is one of two things. 'turnarounds' - docs/turnaround_points.csv, the
    market turnarounds picked with hindsight - asks whether the signal meets its intent;
    'ibd' - the reference calls of docs/ftd_reference.yaml - asks whether it fires when IBD
    called a Follow Through Day. A firing matches a ground truth date when it is at most
    --early_days trading days before it or --late_days after it (1 and 3 by default); a
    positive is a firing, a true positive a firing that matched, and each firing can match
    only one date.

    The report lists:
      - the parameters of the signal run and of the verification
      - the statistics: ground truth dates, positives, trading days, true and false
        positives, precision, recall and F1
      - the scorecard: the ground truth dates and the false positives in one table, in date
        order - a ground truth date with the firing that matched it, or the reason nothing
        did, and the note recorded for the date; a false positive with its firing
      - what followed a firing: the average path of the underlying up to 60 trading days out
        for all, the successful and the failed Follow Through Days, their median return at 5,
        10, 20, 40 and 60 days, and the forward returns against two baselines
      - the rally attempts: how they ended, the rally day the Follow Through Day lands on,
        and the attempts that were never confirmed
      - the sensitivity of the statistics to the parameters, one parameter away at a time
      - the gallery: one annotated chart per ground truth date and per false positive, each
        linked from the scorecard

    Next to ftd_signal_verification_report.html the run writes scorecard.csv, episodes.csv
    and the signal values.
"""

import argparse
import html
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from backfire import signal_analysis as sa
from backfire.backtest import SIGNALS, apply_overrides, build_component
from backfire.base import Environment
from backfire.signals import FTDSignal

REPORT_NAME = "ftd_signal_verification_report"
# how many bars of context a gallery panel shows on each side of its episode
CONTEXT_BARS = 60
# the horizons the forward returns are measured at, in trading days; the average path is
# followed to the longest and a panel caption quotes the middle one
FORWARD_HORIZONS = (5, 20, 60)
CAPTION_HORIZON = FORWARD_HORIZONS[1]
# the days along the path the successful and the failed Follow Through Days are compared at
PATH_DAYS = (5, 10, 20, 40, 60)
# the groups of Follow Through Days the paths are split into - see
# signal_analysis.ftd_dates_by_fate - with the label and the colour each is shown in
_FATES = {'all': ("all FTDs", '#1f4e79'),
          'successful': ("successful FTDs", '#2e7d32'),
          'failed': ("failed FTDs", '#c62828')}
# the grid the sensitivity table walks, one parameter away from the run's own setting
SENSITIVITY_GRID = {'min_decline': [0.08, 0.10],
                    'min_peak_age_days': [15, 20, 25],
                    'day0_window': [5, 10],
                    'ftd_min_gain': [0.0125, 0.015]}

_STATE_COLORS = {FTDSignal.WATCHING: '#c9ccd1', FTDSignal.DAY0: '#e8a33d',
                 FTDSignal.RALLY: '#4a90d9', FTDSignal.UPTREND: '#4caf50'}
_UP, _DOWN = '#26a69a', '#ef5350'
_PEAK, _FTD_MARK, _REFERENCE = '#8e44ad', '#2e7d32', '#555555'
_DAY0_BLOCK, _FTD_BLOCK = '#e8a33d', '#c62828'

_GROUND_TRUTH_LABELS = {'turnarounds': "turnaround points - does the signal meet its intent?",
                        'ibd': "IBD reference calls - does the signal fire when IBD did?"}


# ----------------------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------------------

def split_signal_overrides(argv):
    """
        Pulls the --signal.<path>=<value> arguments out of the command line, leaving the
        rest for argparse. The path is dotted from the root of the signal file, so
        --signal.ftd_min_gain=0.015 sets signal.ftd_min_gain.
    :return: (the remaining arguments, the overrides in apply_overrides form)
    """
    rest, overrides = [], []
    for arg in argv:
        if arg.startswith("--signal."):
            if "=" not in arg:
                raise ValueError(f"'{arg}' must be written --signal.<name>=<value>.")
            overrides.append(arg[len("--"):])
        else:
            rest.append(arg)
    return rest, overrides


def load_signal_conf(path, overrides=()):
    """
        Reads a signal definition file and applies the command line overrides. With no file
        the overrides define the signal on their own.
    :return: the parsed definition, a mapping with a 'signal' key
    """
    conf = {}
    if path:
        with open(path) as f:
            conf = yaml.safe_load(f)
        if not isinstance(conf, dict):
            raise ValueError(f"Signal file '{path}' does not contain a YAML mapping.")
    conf = apply_overrides(conf, overrides)
    if not isinstance(conf.get('signal'), dict):
        raise ValueError("No signal was defined: give a --signal file with a 'signal' "
                         "mapping, or name the class with --signal.name=FTDSignal.")
    return conf


def _valid_date(value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a date in YYYY-MM-DD format.")
    return value


def _days(value):
    days = int(value)
    if days < 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be a number of days, 0 or more.")
    return days


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, overrides = split_signal_overrides(argv)

    p = argparse.ArgumentParser(
        prog="ftd_signal_verification_report.py",
        description="Write the FTD signal verification report: one signal over one "
                    "underlying, scored against a set of ground truth dates.")
    p.add_argument("--underlying", required=True, help="ticker the signal is run over, e.g. QQQ")
    p.add_argument("--start_date", required=True, type=_valid_date,
                   help="first day of the period, YYYY-MM-DD")
    p.add_argument("--end_date", default=None, type=_valid_date,
                   help="last day of the period, YYYY-MM-DD (default: the most recent data)")
    p.add_argument("-md", "--md", dest="md", default="./md",
                   help="market data directory (default: ./md)")
    p.add_argument("--signal", default=None,
                   help="path to the signal definition YAML file")
    p.add_argument("--ground_truth", default="turnarounds",
                   help="'turnarounds' (docs/turnaround_points.csv), 'ibd' (the reference "
                        "calls of docs/ftd_reference.yaml), or the path of a file in either "
                        "format (default: turnarounds)")
    p.add_argument("--early_days", default=sa.EARLY_DAYS, type=_days,
                   help=f"a firing this many trading days before a ground truth date still "
                        f"matches it (default: {sa.EARLY_DAYS})")
    p.add_argument("--late_days", default=sa.LATE_DAYS, type=_days,
                   help=f"a firing this many trading days after a ground truth date still "
                        f"matches it (default: {sa.LATE_DAYS})")
    p.add_argument("--out", required=True, help="output folder for the report")

    args = p.parse_args(argv)
    args.overrides = overrides
    return args


# ----------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------

_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def fmt_date(day, year=True):
    """ A date the way the captions read it: 'Apr 2 2020'. """
    if day is None or (isinstance(day, float) and pd.isna(day)):
        return "-"
    day = sa.as_date(day)
    return f"{_MONTHS[day.month - 1]} {day.day}" + (f" {day.year}" if year else "")


def fmt_pct(value, digits=1, sign=True):
    """ A fraction as a percentage; '-' when it is missing. """
    if value is None or pd.isna(value):
        return "-"
    return f"{value:{'+' if sign else ''}.{digits}%}"


def fmt_num(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def esc(value):
    return html.escape("" if value is None or pd.isna(value) else str(value))


def html_table(headers, rows, css_class="tbl"):
    """ A table of already formatted cells; a cell may carry markup. """
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                     for row in rows)
    return (f'<table class="{css_class}">\n<thead><tr>{head}</tr></thead>\n'
            f'<tbody>\n{body}\n</tbody>\n</table>')


# ----------------------------------------------------------------------------------
# The gallery entries
# ----------------------------------------------------------------------------------

def panel_key(kind, day):
    return f"{kind}-{sa.as_date(day).isoformat()}"


def gallery_entries(matches, episodes, pulses):
    """
        The panels of the gallery and the order they are shown in: one per ground truth date
        the data covers, in the order of the scorecard, then one per false positive.

    :param matches: the match table, as signal_analysis.match_ground_truth returns it
    :param episodes: the rally attempts, as signal_analysis.extract_episodes returns them
    :param pulses: the firing dates, as signal_analysis.pulse_dates returns them
    :return: list of dicts with 'key', 'group' ('truth' or 'fp'), 'date', 'match' (the match
             row or None) and 'episode' (the confirmed attempt shown, or None)
    """
    ftds = episodes[episodes.outcome == 'FTD']
    by_ftd_date = {sa.as_date(row.ftd_date): row for row in ftds.itertuples()}

    rv = []
    for match in matches[matches.in_data].itertuples():
        episode = by_ftd_date.get(sa.as_date(match.detected)) if match.hit else None
        rv.append({'key': panel_key('truth', match.date), 'group': 'truth',
                   'date': match.date, 'match': match, 'episode': episode})
    for day in sa.false_positive_dates(matches, pulses):
        rv.append({'key': panel_key('fp', day), 'group': 'fp', 'date': sa.as_date(day),
                   'match': None, 'episode': by_ftd_date.get(sa.as_date(day))})
    return rv


def caption(entry, outcomes, horizon=CAPTION_HORIZON):
    """
        The one line verdict under a panel, e.g.
        'Apr 2 2020 - fired Apr 2 2020 (+0d) - day 4 of the attempt - +18.3% @20d - held'.
    :param outcomes: the forward returns, as signal_analysis.forward_outcomes returns them
    :param horizon: the horizon whose return the caption quotes
    """
    match, episode = entry['match'], entry['episode']
    parts = []
    if match is not None:
        parts.append(f"ground truth {fmt_date(match.date)}")
        if match.hit:
            parts.append(f"fired {fmt_date(match.detected)} ({match.gap:+.0f}d)")
        else:
            parts.append(f"not fired - {match.reason}")
    else:
        parts.append(f"false positive {fmt_date(entry['date'])}")

    if episode is not None:
        parts.append(f"day {episode.rally_day:.0f} of the attempt")
        column = f'r{horizon}'
        if episode.ftd_date in outcomes.index and column in outcomes.columns:
            parts.append(f"{fmt_pct(outcomes.loc[episode.ftd_date, column])} @{horizon}d")
        if episode.uptrend_end == 'FTD_FAILED':
            parts.append(f"failed after {episode.days_to_failure:.0f}d")
        elif episode.uptrend_end == 'NEW_CORRECTION':
            parts.append("held to the next correction")
        else:
            parts.append("held")
    return " &middot; ".join(parts)


def panel_window(entry, bar_of, bars):
    """
        The bars a panel covers: the attempt, the ground truth date and, when the Follow
        Through Day failed, the failure - with CONTEXT_BARS on each side. A new correction is
        not drawn in: it can be years away.
    """
    days = [entry['date']]
    episode = entry['episode']
    if episode is not None:
        days += [episode.day0_date, episode.end_date]
        if episode.uptrend_end == 'FTD_FAILED':
            days.append(episode.uptrend_end_date)
    positions = [bar_of[day] for day in days if day is not None and day in bar_of]
    if not positions:
        positions = [min(bars.searchsorted(entry['date'], side='left'), len(bars) - 1)]
    return (max(0, min(positions) - CONTEXT_BARS),
            min(len(bars) - 1, max(positions) + CONTEXT_BARS))


# ----------------------------------------------------------------------------------
# The figures
# ----------------------------------------------------------------------------------

def fig_json(fig):
    """ A figure as the JSON the page renders it from, safe to embed in a script tag. """
    return pio.to_json(fig, validate=False).replace("</", "<\\/")


def _round(values, digits=2):
    return [None if pd.isna(v) else round(float(v), digits) for v in values]


def _state_runs(states):
    """ The contiguous runs of one state, as (first position, last position, state). """
    rv, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            rv.append((start, i - 1, states[start]))
            start = i
    return rv


def panel_figure(ohlcv, sv, entry, first, last):
    """
        One annotated episode: the bars and their volume, the peak and the decline to day 0,
        the rally low, the rally day numbers, the Follow Through Day and the days that were
        blocked from being one, the ground truth date, and a band of the machine's state.
    """
    window, svw = ohlcv.iloc[first:last + 1], sv.iloc[first:last + 1]
    days = list(window.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=days, open=_round(window.O), high=_round(window.H), low=_round(window.L),
        close=_round(window.C), name='price', showlegend=False,
        increasing_line_color=_UP, decreasing_line_color=_DOWN), row=1, col=1)
    fig.add_trace(go.Bar(
        x=days, y=_round(window.V, 0), name='volume', showlegend=False, marker_line_width=0,
        marker_color=[_UP if c >= o else _DOWN for o, c in zip(window.O, window.C)],
        hoverinfo='skip'), row=2, col=1)

    # the diagnostic row of every bar, on hover over the close
    fig.add_trace(go.Scatter(
        x=days, y=_round(window.C), mode='markers', showlegend=False,
        marker=dict(size=6, color='rgba(0,0,0,0)'), name='',
        customdata=[[state, _text(event), _text(d0), _text(ftdb), _rally(day), decline]
                    for state, event, d0, ftdb, day, decline
                    in zip(svw.state, svw.event, svw.day0_block, svw.ftd_block,
                           svw.rally_day, svw.decline_pct)],
        hovertemplate=("%{x|%b %d %Y}<br>close %{y}<br>state %{customdata[0]}"
                       "<br>rally day %{customdata[4]}"
                       "<br>from peak %{customdata[5]:.1%}"
                       "<br>event %{customdata[1]}"
                       "<br>day 0 blocked %{customdata[2]}"
                       "<br>follow through blocked %{customdata[3]}<extra></extra>")),
        row=1, col=1)

    episode = entry['episode']
    if episode is not None:
        _draw_episode(fig, episode, window, days)
    else:
        # nothing fired here: show what the correction test was measured against instead
        _draw_correction_context(fig, svw, window, entry['date'])
    _draw_rally_numbers(fig, svw, window, days)
    _draw_blocks(fig, svw, window, days)

    if entry['match'] is not None and entry['date'] in window.index:
        fig.add_vline(x=entry['date'], line=dict(color=_REFERENCE, width=1, dash='dash'),
                      row=1, col=1)

    # a thin band of the machine's state along the bottom of the price pane
    states = list(svw.state)
    for start, end, state in _state_runs(states):
        fig.add_shape(type='rect', xref='x', yref='y domain',
                      x0=days[start], x1=days[min(end + 1, len(days) - 1)], y0=0, y1=0.03,
                      fillcolor=_STATE_COLORS.get(state, '#ffffff'), line_width=0,
                      layer='below', row=1, col=1)

    fig.update_layout(height=430, margin=dict(l=48, r=16, t=28, b=24),
                      xaxis_rangeslider_visible=False, hovermode='closest',
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                      font=dict(size=11), dragmode='pan')
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    return fig


def _text(value):
    return '-' if value is None or pd.isna(value) or value == '' else str(value)


def _rally(value):
    return '-' if value is None or pd.isna(value) else f"{value:.0f}"


def _draw_episode(fig, episode, window, days):
    """ The peak, the decline to day 0, the rally low, the rally days and the pulse. """
    if episode.peak_date in window.index:
        fig.add_trace(go.Scatter(
            x=[episode.peak_date], y=[round(float(episode.peak), 2)], mode='markers+text',
            marker=dict(symbol='triangle-down', size=10, color=_PEAK), text=['peak'],
            textposition='top center', textfont=dict(color=_PEAK), showlegend=False,
            hovertemplate=f"peak {fmt_num(episode.peak)} on %{{x|%b %d %Y}}<extra></extra>"),
            row=1, col=1)
        if episode.day0_date in window.index:
            fig.add_shape(type='line', x0=episode.peak_date, x1=episode.day0_date,
                          y0=float(episode.peak), y1=float(episode.rally_low),
                          line=dict(color=_PEAK, width=1, dash='dot'), row=1, col=1)
            fig.add_annotation(x=episode.day0_date, y=float(episode.rally_low),
                               text=f"day 0, {fmt_pct(episode.decline_pct)} in "
                                    f"{episode.peak_age:.0f}d",
                               showarrow=False, yshift=-16, font=dict(size=10, color=_PEAK),
                               row=1, col=1)

    # the rally low holds from day 0 until the attempt ends, or until the pulse fails
    end = (episode.uptrend_end_date if episode.uptrend_end == 'FTD_FAILED'
           else episode.end_date) or days[-1]
    if episode.day0_date is not None and episode.rally_low is not None:
        fig.add_shape(type='line', x0=max(episode.day0_date, days[0]), x1=min(end, days[-1]),
                      y0=float(episode.rally_low), y1=float(episode.rally_low),
                      line=dict(color=_FTD_MARK, width=1), row=1, col=1)

    if episode.ftd_date in window.index:
        close = float(window.C[episode.ftd_date])
        fig.add_trace(go.Scatter(
            x=[episode.ftd_date], y=[round(close, 2)], mode='markers+text',
            marker=dict(symbol='triangle-up', size=13, color=_FTD_MARK), text=['FTD'],
            textposition='bottom center', textfont=dict(color=_FTD_MARK), showlegend=False,
            hovertemplate=(f"follow through day, day {episode.rally_day:.0f}"
                           f"<br>gain {fmt_pct(episode.ftd_gain)}"
                           f"<br>volume x{fmt_num(episode.ftd_volume_ratio)}<extra></extra>")),
            row=1, col=1)


def _draw_blocks(fig, svw, window, days):
    """
        The days that came close to a transition and the clause that stopped them: a new low
        that was not a day 0, marked at the low, and a rally day that gained enough but was
        not a Follow Through Day, marked at the close.
    """
    for column, prices, symbol, color, label in [
            ('day0_block', window.L, 'circle-open', _DAY0_BLOCK, 'not a day 0'),
            ('ftd_block', window.C, 'x', _FTD_BLOCK, 'not a follow through day')]:
        marked = [(day, price, reason) for day, price, reason
                  in zip(days, prices, svw[column]) if _text(reason) != '-']
        if not marked:
            continue
        fig.add_trace(go.Scatter(
            x=[m[0] for m in marked], y=_round([m[1] for m in marked]), mode='markers',
            marker=dict(symbol=symbol, size=8, color=color, line=dict(width=1.5, color=color)),
            showlegend=False, name='', customdata=[m[2] for m in marked],
            hovertemplate="%{x|%b %d %Y}<br>" + label + ": %{customdata}<extra></extra>"),
            row=1, col=1)


def _draw_correction_context(fig, svw, window, day):
    """
        The peak the correction test was being measured against on the day, and the deepest
        low of the window against it. This is the whole of the explanation for a ground truth
        date the signal missed: the decline was too shallow, or the peak too young, or both.
    """
    if day not in svw.index or window.empty:
        return
    peak, peak_date = svw.peak[day], svw.peak_date[day]
    if peak_date is None or pd.isna(peak_date) or peak_date not in window.index:
        return
    # the deepest low since the peak: the best shot the correction test had in this window
    since = window.L[window.index >= peak_date]
    if since.empty:
        return
    low_day = since.idxmin()
    low = float(since[low_day])
    age = window.index.searchsorted(low_day) - window.index.searchsorted(peak_date)

    fig.add_trace(go.Scatter(
        x=[peak_date], y=[round(float(peak), 2)], mode='markers+text',
        marker=dict(symbol='triangle-down', size=10, color=_PEAK), text=['peak'],
        textposition='top center', textfont=dict(color=_PEAK), showlegend=False,
        hovertemplate=f"peak {fmt_num(peak)} on %{{x|%b %d %Y}}<extra></extra>"), row=1, col=1)
    fig.add_shape(type='line', x0=peak_date, x1=low_day, y0=float(peak), y1=low,
                  line=dict(color=_PEAK, width=1, dash='dot'), row=1, col=1)
    fig.add_annotation(x=low_day, y=low, showarrow=False, yshift=-16,
                       text=f"lowest low, {fmt_pct(low / float(peak) - 1)} in {age}d",
                       font=dict(size=10, color=_PEAK), row=1, col=1)


def _draw_rally_numbers(fig, svw, window, days):
    """ The rally day number over each bar of an attempt under way. """
    numbers = [(day, high, n) for day, high, n in zip(days, window.H, svw.rally_day)
               if n is not None and not pd.isna(n)]
    if numbers:
        fig.add_trace(go.Scatter(
            x=[n[0] for n in numbers], y=_round([n[1] for n in numbers]), mode='text',
            text=[f"{n[2]:.0f}" for n in numbers], textposition='top center',
            textfont=dict(size=9, color='#777777'), showlegend=False, hoverinfo='skip'),
            row=1, col=1)


def forward_path_figure(paths, groups):
    """
        The average path of the return after a Follow Through Day, one line per group - all
        of them, the successful ones and the failed ones - with the interquartile band of all.
    :param paths: the forward paths, as signal_analysis.forward_paths returns them
    :param groups: the firing days by group, as signal_analysis.ftd_dates_by_fate returns them
    """
    fig = go.Figure()
    x = list(paths.columns)
    everything = paths[paths.index.isin(groups.get('all', []))]
    if not everything.empty:
        low, high = everything.quantile(0.25), everything.quantile(0.75)
        fig.add_trace(go.Scatter(x=x + x[::-1],
                                 y=_round(high, 4) + _round(low, 4)[::-1],
                                 fill='toself', fillcolor='rgba(74,144,217,0.18)',
                                 line=dict(width=0), hoverinfo='skip',
                                 name='interquartile range, all FTDs'))
    for fate, dates in groups.items():
        group = paths[paths.index.isin(dates)]
        if group.empty:
            continue
        label, color = _FATES.get(fate, (fate, '#555555'))
        fig.add_trace(go.Scatter(x=x, y=_round(group.mean(), 4), mode='lines',
                                 line=dict(color=color, width=2),
                                 name=f"{label} ({len(group)})",
                                 hovertemplate=f"{label}, day %{{x}}: %{{y:.2%}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color='#999999', width=1))
    fig.update_layout(height=340, margin=dict(l=56, r=16, t=36, b=40),
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(size=11),
                      legend=dict(orientation='h', y=1.12),
                      xaxis_title="trading days after the firing",
                      yaxis_title="return from the next open", yaxis_tickformat='.0%')
    fig.update_yaxes(gridcolor='#eeeeee')
    fig.update_xaxes(gridcolor='#eeeeee')
    return fig


def rally_day_figure(episodes):
    """ How far into the attempt the Follow Through Day lands. """
    ftds = episodes[episodes.outcome == 'FTD']
    counts = ftds.rally_day.dropna().astype(int).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=list(counts.index), y=[int(v) for v in counts],
                           marker_color='#4a90d9',
                           hovertemplate="day %{x}: %{y} firings<extra></extra>"))
    fig.update_layout(height=280, margin=dict(l=56, r=16, t=16, b=40),
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(size=11),
                      xaxis_title="rally day of the follow through day",
                      yaxis_title="firings", bargap=0.15)
    fig.update_yaxes(gridcolor='#eeeeee')
    fig.update_xaxes(gridcolor='#eeeeee', dtick=1)
    return fig


# ----------------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; color: #222; background: #fafafa; font-size: 14px; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 36px 0 10px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
h3 { font-size: 14px; margin: 24px 0 8px; color: #444; }
.sub { color: #666; margin: 0 0 16px; }
.facts { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 6px; }
.fact { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 8px 14px; }
.fact b { display: block; font-size: 19px; font-weight: 600; }
.fact span { color: #777; font-size: 12px; }
table.tbl { border-collapse: collapse; width: 100%; background: #fff; font-size: 12.5px; }
table.tbl th, table.tbl td { border: 1px solid #e6e6e6; padding: 5px 8px; text-align: left;
                             vertical-align: top; }
table.tbl th { background: #f2f4f6; font-weight: 600; }
table.tbl tr.miss td { background: #fff6f5; }
table.kv { width: auto; }
table.kv td:first-child { color: #666; white-space: nowrap; }
.panel { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 10px 12px;
         margin: 16px 0; scroll-margin-top: 12px; }
.panel h4 { margin: 0 0 2px; font-size: 14px; }
.cap { color: #555; font-size: 12.5px; margin: 6px 2px 0; }
.plot { min-height: 300px; }
details { margin: 14px 0; }
summary { cursor: pointer; font-weight: 600; padding: 6px 0; }
.legend { color: #666; font-size: 12px; margin: 6px 0 0; }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px;
      vertical-align: -1px; margin: 0 3px 0 10px; }
.note { color: #666; font-size: 12.5px; }
"""

_SCRIPT = """
// Each panel is rendered only when it scrolls into view: the gallery holds around a hundred
// figures and drawing them all up front locks the browser up for a long time.
(function () {
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var el = entry.target;
      io.unobserve(el);
      var payload = document.getElementById(el.id + '-data');
      if (!payload) return;
      var fig = JSON.parse(payload.textContent);
      Plotly.newPlot(el, fig.data, fig.layout,
                     {displayModeBar: false, responsive: true, scrollZoom: true});
    });
  }, {rootMargin: '700px 0px'});
  document.querySelectorAll('.plot').forEach(function (el) { io.observe(el); });
})();
"""


def plot_div(key, fig):
    """ A placeholder the page renders the figure into when it scrolls into view. """
    return (f'<div class="plot" id="{key}"></div>\n'
            f'<script type="application/json" id="{key}-data">{fig_json(fig)}</script>')


def fact(value, label):
    return f'<div class="fact"><b>{value}</b><span>{esc(label)}</span></div>'


def _kv_table(pairs):
    return html_table(["parameter", "value"], [[esc(k), esc(v)] for k, v in pairs],
                      css_class="tbl kv")


def parameters_html(signal, conf, underlying, ohlcv, md, ground_truth, early, late):
    """ The signal run and the verification test, spelled out so a run can be repeated. """
    signal_pairs = [("signal", conf['signal']['name'])]
    signal_pairs += [(k, repr(v)) for k, v in sorted(conf['signal'].items()) if k != 'name']
    signal_pairs += [("underlying", underlying), ("market data", md),
                     ("period", f"{ohlcv.index[0]} .. {ohlcv.index[-1]}"),
                     ("trading days", f"{len(ohlcv):,}")]
    truth_pairs = [("ground truth", ground_truth),
                   ("file", sa.ground_truth_path(ground_truth)),
                   ("question", _GROUND_TRUTH_LABELS.get(ground_truth,
                                                          "the dates in the file given")),
                   ("tolerance", f"a firing matches a ground truth date from {early} trading "
                                 f"day{'s' if early != 1 else ''} before it to {late} after it; "
                                 f"each firing matches at most one date"),
                   ("forward path", f"{FORWARD_HORIZONS[-1]} trading days after the firing, "
                                    f"from the next day's open")]
    return ("<h3>Signal run</h3>" + _kv_table(signal_pairs)
            + "<h3>Verification test</h3>" + _kv_table(truth_pairs))


def header_html(signal, underlying, ground_truth, stats):
    truth = f"{stats['ground_truth']}"
    if stats['ground_truth_total'] != stats['ground_truth']:
        truth += f" of {stats['ground_truth_total']}"
    return (f"<h1>{esc(REPORT_NAME)}</h1>\n"
            f'<p class="sub">{esc(signal.name)} on {esc(underlying)}, scored against '
            f'{esc(_GROUND_TRUTH_LABELS.get(ground_truth, ground_truth))}</p>\n'
            '<div class="facts">'
            + fact(truth, "ground truth dates in the data")
            + fact(stats['positives'], "positives (firings)")
            + fact(f"{stats['trading_days']:,}", "trading days")
            + fact(stats['true_positives'], "true positives")
            + fact(stats['false_positives'], "false positives")
            + fact(fmt_pct(stats['precision'], sign=False), "precision")
            + fact(fmt_pct(stats['recall'], sign=False), "recall")
            + fact(fmt_num(stats['f1']), "F1")
            + "</div>")


def scorecard_html(matches, false_positives, episodes):
    """
        The ground truth dates and the false positives in one table, in date order: a ground
        truth date sorts by itself, a false positive by its firing date. A ground truth row
        carries the firing that matched it, or the reason nothing did, and the note recorded
        for the date; a false positive row only its firing. Every firing also shows the rally
        day it came on, its gain and what became of the uptrend it started. The dates link to
        their gallery panels.
    """
    ftds = {sa.as_date(row.ftd_date): row for row in
            episodes[episodes.outcome == 'FTD'].itertuples()}

    keyed = []
    for match in matches.itertuples():
        link = (f'<a href="#{panel_key("truth", match.date)}">{fmt_date(match.date)}</a>'
                if match.in_data else fmt_date(match.date))
        if match.hit:
            fired, result = f"{fmt_date(match.detected)} ({match.gap:+.0f}d)", "true positive"
        else:
            fired, result = "", "miss" if match.in_data else "outside the data"
        note = " - ".join(filter(None, [esc(match.episode), esc(match.notes)]))
        firing = _firing_cells(ftds.get(sa.as_date(match.detected)) if match.hit else None)
        reason = "" if match.hit or not match.in_data else esc(match.reason)
        keyed.append(((sa.as_date(match.date), 0),
                      [link, fired, result, *firing, reason, note]))
    for day in false_positives:
        link = f'<a href="#{panel_key("fp", day)}">{fmt_date(day)}</a>'
        keyed.append(((sa.as_date(day), 1),
                      ["", link, "false positive", *_firing_cells(ftds.get(sa.as_date(day))),
                       "", ""]))

    rows = [row for _, row in sorted(keyed, key=lambda item: item[0])]
    return html_table(["ground truth date", "firing date", "result", "rally day", "gain",
                       "what became of it", "reason for not firing", "note"], rows)


def _firing_cells(episode):
    """ The rally day, the gain and what became of the uptrend, for a firing's episode. """
    if episode is None:
        return ["", "", ""]
    held = {'FTD_FAILED': f"failed after {episode.days_to_failure:.0f}d"
            if not pd.isna(episode.days_to_failure) else "failed",
            'NEW_CORRECTION': "held to the next correction"}.get(episode.uptrend_end, "held")
    return [f"{episode.rally_day:.0f}", fmt_pct(episode.ftd_gain), held]


def outcomes_table_html(summary):
    rows = [[esc(name), f"{int(value)}"] for name, value in summary.items()]
    return html_table(["", "count"], rows)


def forward_table_html(table, horizons):
    headers = ["days measured", "count"]
    for horizon in horizons:
        headers += [f"mean @{horizon}d", f"median @{horizon}d"]
    headers += [f"positive @{max(horizons)}d", "median worst", "median best"]
    rows = []
    for name, row in table.iterrows():
        cells = [esc(name), f"{int(row['days'])}"]
        for horizon in horizons:
            cells += [fmt_pct(row[f'mean r{horizon}']), fmt_pct(row[f'median r{horizon}'])]
        cells += [fmt_pct(row[f'positive at {max(horizons)}d'], sign=False),
                  fmt_pct(row['median mae']), fmt_pct(row['median mfe'])]
        rows.append(cells)
    return html_table(headers, rows)


def path_table_html(table):
    """ The median return along the path, a row per group of Follow Through Days. """
    days = [column for column in table.columns if column.startswith('median r')]
    rows = [[esc(_FATES.get(name, (name,))[0]), f"{int(row['count'])}",
             *[fmt_pct(row[column]) for column in days]]
            for name, row in table.iterrows()]
    return html_table(["", "count", *[f"median @{column[len('median r'):]}d" for column in days]],
                      rows)


def sensitivity_table_html(table, horizon):
    rows = [[esc(row['setting']), f"{int(row['pulses'])}",
             f"{int(row['hits'])} / {int(row['ground_truth'])}",
             fmt_pct(row['precision'], sign=False), fmt_pct(row['recall'], sign=False),
             fmt_pct(row['failure_rate'], sign=False), fmt_pct(row[f'median r{horizon}'])]
            for _, row in table.iterrows()]
    return html_table(["setting", "firings", "hits", "precision", "recall", "failure rate",
                       f"median @{horizon}d"], rows)


def unconfirmed_table_html(episodes):
    """ The attempts that never produced a Follow Through Day, kept out of the gallery. """
    rows = []
    for row in episodes[episodes.outcome != 'FTD'].itertuples():
        rows.append([fmt_date(row.day0_date), fmt_date(row.day1_date), fmt_num(row.peak),
                     fmt_date(row.peak_date), fmt_pct(row.decline_pct), fmt_num(row.rally_low),
                     row.outcome, fmt_date(row.end_date)])
    return html_table(["day 0", "day 1", "peak", "peak made", "decline", "rally low",
                       "outcome", "ended"], rows)


def state_legend_html():
    swatches = "".join(f'<span class="sw" style="background:{color}"></span>{esc(state)}'
                       for state, color in _STATE_COLORS.items())
    return f'<p class="legend">state band:{swatches}</p>'


def gallery_html(ohlcv, sv, entries, outcomes, bar_of, bars, horizon=CAPTION_HORIZON):
    """ The panels: the ground truth dates, then the false positives folded away. """
    parts = []
    truth = [entry for entry in entries if entry['group'] == 'truth']
    fps = [entry for entry in entries if entry['group'] == 'fp']

    parts.append("<h3>Ground truth dates</h3><p class='note'>One chart per ground truth "
                 "date in the scorecard, in date order.</p>")
    parts += [_panel_html(ohlcv, sv, entry, outcomes, bar_of, bars, horizon)
              for entry in truth]
    if fps:
        parts.append(f"<details><summary>False positives ({len(fps)}) - firings no ground "
                     f"truth date accounts for</summary>")
        parts += [_panel_html(ohlcv, sv, entry, outcomes, bar_of, bars, horizon)
                  for entry in fps]
        parts.append("</details>")
    return "\n".join(parts)


def _panel_html(ohlcv, sv, entry, outcomes, bar_of, bars, horizon):
    label = fmt_date(entry['date'])
    if entry['match'] is not None and entry['match'].episode:
        label = f"{entry['match'].episode} - {label}"
    first, last = panel_window(entry, bar_of, bars)
    chart = plot_div(entry['key'] + "-fig", panel_figure(ohlcv, sv, entry, first, last))
    return (f'<div class="panel" id="{entry["key"]}"><h4>{esc(label)}</h4>'
            f'{chart}<p class="cap">{caption(entry, outcomes, horizon)}</p></div>')


def build_page(title, body):
    return (f"<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n"
            f"<script>{get_plotlyjs()}</script>\n</head>\n<body>\n<main>\n{body}\n</main>\n"
            f"<script>{_SCRIPT}</script>\n</body></html>\n")


# ----------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------

def run_report(signal_conf, underlying, start_date, end_date=None, md="./md", out="",
               ground_truth="turnarounds", early=sa.EARLY_DAYS, late=sa.LATE_DAYS,
               horizons=FORWARD_HORIZONS, grid=None):
    """
        Runs one signal over one underlying, scores it against the ground truth and writes
        the report into 'out'.
    :param ground_truth: 'turnarounds', 'ibd' or a file path - see
                         signal_analysis.load_ground_truth
    :param early, late: the matching tolerance in trading days, before and after the date
    :param horizons: the forward horizons, in trading days; the average path runs to the
                     longest and the captions quote the middle one
    :param grid: the parameter grid of the sensitivity table, SENSITIVITY_GRID by default
    :return: dict with 'paths' (the files written, by name), 'signal', 'ohlcv',
             'signal_values', 'episodes', 'scorecard' (the match table), 'stats',
             'false_positives', 'paths_after' (the forward paths), 'path_stats' (their medians
             by what became of the Follow Through Day), 'sensitivity' and 'html'
    """
    env = Environment(md=md, out_dir=out)
    signal = build_component(signal_conf['signal'], SIGNALS, 'signal')
    signal.bind(env)

    ohlcv = env.load_ohlcv(underlying, start_date, end_date)
    if ohlcv.empty:
        raise ValueError(f"No {underlying} prices between {start_date} and "
                         f"{end_date or 'the most recent data'}.")
    sv = signal(ohlcv)
    if 'event' not in sv.columns:
        raise ValueError(f"{signal.name} does not record the day by day diagnostics this "
                         f"report reads. It is written for FTDSignal - see docs/SIGNALS.md.")

    truth = sa.load_ground_truth(ground_truth)
    episodes = sa.extract_episodes(ohlcv, sv)
    pulses = sa.pulse_dates(sv)
    matches = sa.match_ground_truth(sv, truth, early, late)
    stats = sa.verification_stats(matches, pulses, len(ohlcv))
    false_positives = sa.false_positive_dates(matches, pulses)
    horizons = tuple(sorted(horizons))
    forward_days, quoted = horizons[-1], horizons[len(horizons) // 2]
    outcomes = sa.forward_outcomes(ohlcv, pulses, horizons)
    paths = sa.forward_paths(ohlcv, pulses, forward_days)
    fates = sa.ftd_dates_by_fate(episodes)
    path_stats = sa.path_table(paths, fates, PATH_DAYS)
    table = sa.forward_table(ohlcv, {f"{signal.name} firings": pulses,
                                     **sa.baseline_dates(ohlcv, signal)}, horizons)
    base = {k: v for k, v in signal_conf['signal'].items() if k != 'name'}
    settings = sa.sensitivity(ohlcv, SENSITIVITY_GRID if grid is None else grid, truth,
                              base_params=base, factory=SIGNALS[signal_conf['signal']['name']],
                              env=env, early=early, late=late, horizon=quoted)

    bars = ohlcv.index
    bar_of = {day: i for i, day in enumerate(bars)}
    entries = gallery_entries(matches, episodes, pulses)

    body = "\n".join([
        header_html(signal, underlying, ground_truth, stats),
        "<h2>Parameters</h2>",
        parameters_html(signal, signal_conf, underlying, ohlcv, md, ground_truth, early, late),
        "<h2>Scorecard</h2>",
        "<p class='note'>The ground truth dates and the false positives - firings no ground "
        "truth date accounts for - in date order. A ground truth date is hit when a firing "
        "falls within the tolerance; a date outside the data is listed but not scored. The "
        "reason for a miss is read off the signal's own diagnostics on the date.</p>",
        scorecard_html(matches, false_positives, episodes),
        "<h2>What followed a firing</h2>",
        "<h3>The average path after a firing</h3>",
        f"<p class='note'>The mean return for up to {forward_days} trading days after a "
        f"firing: over all {len(fates['all'])} follow through days, with their interquartile "
        f"band, over the {len(fates['successful'])} successful ones - the uptrend held until "
        f"a new correction - and over the {len(fates['failed'])} failed ones the index closed "
        f"back below the rally low. An uptrend still open at the end of the data counts only "
        f"under all. Measured from the next day's open, the way SignalDrivenStrategy executes "
        f"a signal. Whether a follow through day failed is only known afterwards, so the split "
        f"is hindsight, not something to trade on.</p>",
        plot_div("fwd-path", forward_path_figure(paths, fates)),
        "<h3>Along the path, successful against failed</h3>",
        "<p class='note'>The median return at each day after the firing; the count is the "
        "firings with a path, and a path the data cuts short drops out of the later days.</p>",
        path_table_html(path_stats),
        "<h3>Against the baselines</h3>",
        "<p class='note'>'naive FTD days' applies the gain and volume tests to every day "
        "with none of the correction, day 0 or day count logic around them.</p>",
        forward_table_html(table, horizons),
        "<h2>The rally attempts</h2>",
        "<h3>How they ended</h3>",
        outcomes_table_html(sa.attempt_summary(episodes)),
        "<h3>Where in the attempt the follow through day lands</h3>",
        plot_div("rally-day", rally_day_figure(episodes)),
        "<h3>Attempts that were never confirmed</h3>",
        "<p class='note'>Rally attempts that were undercut or timed out, so no chart.</p>",
        f"<details><summary>{len(episodes[episodes.outcome != 'FTD'])} attempts</summary>"
        f"{unconfirmed_table_html(episodes)}</details>",
        "<h2>Sensitivity to the parameters</h2>",
        "<p class='note'>One parameter away from this run's setting at a time, scored "
        "against the same ground truth with the same tolerance.</p>",
        sensitivity_table_html(settings, quoted),
        "<h2>Gallery</h2>",
        state_legend_html(),
        gallery_html(ohlcv, sv, entries, outcomes, bar_of, bars, quoted),
    ])

    paths_written = {}
    if out:
        os.makedirs(out, exist_ok=True)
        paths_written['report'] = os.path.join(out, f"{REPORT_NAME}.html")
        with open(paths_written['report'], "w", encoding="utf-8") as f:
            f.write(build_page(f"{REPORT_NAME}: {signal.name} on {underlying}", body))
        paths_written['scorecard'] = os.path.join(out, "scorecard.csv")
        matches.to_csv(paths_written['scorecard'], index=False)
        paths_written['episodes'] = os.path.join(out, "episodes.csv")
        episodes.to_csv(paths_written['episodes'], index=False)
        paths_written['signal_values'] = os.path.join(out, f"{signal.name}.csv")
        sv.to_csv(paths_written['signal_values'])

    return {'paths': paths_written, 'signal': signal, 'ohlcv': ohlcv, 'signal_values': sv,
            'episodes': episodes, 'scorecard': matches, 'stats': stats,
            'false_positives': false_positives, 'paths_after': paths,
            'path_stats': path_stats, 'sensitivity': settings, 'html': body}


def main(argv=None):
    args = parse_args(argv)
    try:
        conf = load_signal_conf(args.signal, args.overrides)
        rv = run_report(signal_conf=conf, underlying=args.underlying,
                        start_date=args.start_date, end_date=args.end_date, md=args.md,
                        out=args.out, ground_truth=args.ground_truth,
                        early=args.early_days, late=args.late_days)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"report failed: {e}", file=sys.stderr)
        return 2

    stats = rv['stats']
    print(f"Signal       : {rv['signal'].name}")
    print(f"Underlying   : {args.underlying}")
    print(f"Period       : {rv['ohlcv'].index[0]} .. {rv['ohlcv'].index[-1]} "
          f"({stats['trading_days']:,} trading days)")
    print(f"Ground truth : {args.ground_truth}, {stats['ground_truth']} dates in the data")
    print(f"Tolerance    : -{args.early_days} / +{args.late_days} trading days")
    print(f"Positives    : {stats['positives']} "
          f"({stats['true_positives']} true, {stats['false_positives']} false)")
    print(f"Precision    : {fmt_pct(stats['precision'], sign=False)}")
    print(f"Recall       : {fmt_pct(stats['recall'], sign=False)}")
    print(f"F1           : {fmt_num(stats['f1'])}")
    print(f"Attempts     : {len(rv['episodes'])}")
    for name, path in rv['paths'].items():
        print(f"  {name:<14} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
