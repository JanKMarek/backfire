"""
    Command line driver for the interactive strategy result dashboard.

    Reads the CSV files a backtest run wrote into its output folder and serves a
    Plotly Dash dashboard over them. The dashboard never re-runs the backtest:

      uv run python backfire/visualize.py -out "out/test/Index_50dMAvs200dMA"

    then point a browser at http://127.0.0.1:8050/ (override with --host/--port).

    The run folder is identified by its positions file, pos_{ticker}_{strategy}.csv;
    the matching stats_ and trades_ files are read alongside it, when present.
    Subdirectories of the run folder are ignored, so a folder holding several runs
    must be pointed at one level down.

    The dashboard has three parts:
      - the chart title carries the strategy, the backtest period, the total return,
        the CAGR and the three largest drawdowns
      - the chart pane plots the mark to market balance on a log left axis, the
        underlying OHLC on a log right axis, the entry/exit signal regimes as
        translucent bands and the executed trades as markers; it zooms and pans, and
        shows the day's OHLC and portfolio value when the cursor is near the underlying
      - the data pane tabulates monthly and annual strategy returns; clicking a cell
        marks the first and the last day of that period in the chart pane
"""

import argparse
import glob
import math
import os
import sys

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, dash_table, dcc, html

_MONTH_LABELS = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November',
                 12: 'December'}
_MONTH_NUMBERS = {name: number for number, name in _MONTH_LABELS.items()}
_MONTH_COLUMN = 'Month'
_ANNUAL_ROW = 'Annual'
_ROW_ORDER = list(_MONTH_LABELS.values()) + [_ANNUAL_ROW]

_ENTRY_FILL = 'rgba(44, 160, 44, 0.12)'
_EXIT_FILL = 'rgba(214, 39, 40, 0.12)'
_ENTRY_MARKER = '#2ca02c'
_EXIT_MARKER = '#d62728'
_HIGHLIGHT_COLOR = 'RoyalBlue'

_GAIN_THRESHOLD = 0.05
_LOSS_THRESHOLD = -0.05
# a signal block shorter than this is widened so it stays visible on a multi year axis
_MIN_BAND_DAYS = 3
# how close, in pixels, the cursor must be to the underlying for the hover label to show
_HOVER_DISTANCE_PX = 20
_PRICE_HOVER_NAME = 'price hover'
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8050


# ----------------------------------------------------------------------------------
# Loading the saved files of a run
# ----------------------------------------------------------------------------------

def find_run_files(folder):
    """
        Locates the CSV files of one backtest run.
    :param folder: the run's output folder, as passed to backtest.py --out. Only the
                   files directly in it are considered; subdirectories are ignored.
    :return: dict with keys 'folder', 'ticker', 'strategy_name', 'positions', 'stats',
             'trades' - the last two are None when the run did not write them
    :raise ValueError: when the folder is not a directory, holds no positions file, or
                       holds more than one
    """
    if not os.path.isdir(folder):
        raise ValueError(f"'{folder}' is not a folder.")

    paths = sorted(glob.glob(os.path.join(folder, "pos_*.csv")))
    if not paths:
        raise ValueError(f"No pos_*.csv file in '{folder}'. Is it a backtest output folder?")
    if len(paths) > 1:
        names = ', '.join(os.path.basename(p) for p in paths)
        raise ValueError(f"'{folder}' holds {len(paths)} runs ({names}). "
                         f"Point -out at one of them.")

    # strategy names may themselves contain underscores, so only the ticker is split off
    stem = os.path.basename(paths[0])[len("pos_"):-len(".csv")]
    ticker, _, strategy_name = stem.partition("_")

    def optional(prefix):
        path = os.path.join(folder, f"{prefix}_{stem}.csv")
        return path if os.path.isfile(path) else None

    return {'folder': folder, 'ticker': ticker, 'strategy_name': strategy_name,
            'positions': paths[0], 'stats': optional("stats"), 'trades': optional("trades")}


def load_positions(path):
    """
        Reads the daily positions and equity curve of a run.
    :param path: path to a pos_*.csv file
    :return: DataFrame indexed by a DatetimeIndex named 'Date', with the columns the
             backtest wrote: O, H, L, C, V, es, es_id, xs, pos, cash, action,
             delta_shares, memo, buy_price, balance and the unrealized_* columns
    :raise ValueError: when the file has no rows or no 'balance' column
    """
    # index_col=0 rather than index_col='Date': some older runs lost the 'Date' header
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    if df.empty:
        raise ValueError(f"'{path}' has no rows.")
    if 'balance' not in df.columns:
        raise ValueError(f"'{path}' has no 'balance' column; re-run the backtest.")
    return df


def load_stats(path):
    """
        Reads the performance metrics of a run.
    :param path: path to a stats_*.csv file, or None when the run wrote none
    :return: Series of raw strings indexed by metric name, empty when path is None.
             Values are strings because the file mixes counts, floats, NA and a
             timedelta; read individual metrics with stat_value.
    """
    if path is None:
        return pd.Series(dtype=object)
    return pd.read_csv(path, index_col=0).iloc[:, 0]


def stat_value(stats, key, default=None):
    """
        Reads one metric out of a stats Series as a float.
    :param stats: the Series returned by load_stats
    :param key: metric name, e.g. 'sharpe'
    :param default: returned when the metric is absent (stats files written before
                    the metric existed) or is not a number ('NA', an empty cell)
    :return: float, or default
    """
    value = pd.to_numeric(stats.get(key), errors='coerce')
    return default if pd.isna(value) else float(value)


def load_trades(path):
    """
        Reads the trades of a run.
    :param path: path to a trades_*.csv file, or None
    :return: DataFrame with entry_date and exit_date parsed as timestamps and a
             RangeIndex, empty with the expected columns when path is None
    """
    columns = ['ticker', 'entry_date', 'entry_price', 'shares', 'exit_date', 'exit_price',
              'memo', 'pnl', 'pnl_pcnt', 'hp']
    if path is None:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, index_col=0)
    for column in ('entry_date', 'exit_date'):
        df[column] = pd.to_datetime(df[column])
    return df


_RISK_MANAGEMENT_LABELS = ('sl',)


def signal_names(positions):
    """
        Recovers the entry and exit signal names from the trade memos, e.g.
        "bought:1153 shares;50dMAAbove200MA" names the entry signal. A "sold:" memo
        names either the exit signal or the risk management label ('sl'); the latter
        is skipped so a strategy that happens to be stopped out first still reports
        its real exit signal name. The standalone signal CSVs are not used for
        this: their filenames and column shapes are inconsistent across runs, while
        the memo is always written by the backtest.
    :param positions: the DataFrame from load_positions
    :return: (entry_name, exit_name); either is None when that side never triggered a
             trade (exit_name is also None when every exit was risk management driven)
    """
    def name_of(prefix):
        memo = positions['memo']
        # avoid the .str accessor: a run with no memos at all reads back as an
        # all-NaN float64 column, which .str rejects outright
        is_match = memo.map(lambda v: isinstance(v, str) and v.startswith(prefix))
        for value in memo[is_match]:
            label = value.rsplit(';', 1)[-1]
            if label not in _RISK_MANAGEMENT_LABELS:
                return label
        return None

    return name_of('bought:'), name_of('sold:')


class Run:
    """
        One backtest run, as read back from its output folder.

        Attributes:
        - folder, ticker, strategy_name
        - positions: daily O/H/L/C/V, signals and balance (see load_positions)
        - stats: the metrics the run wrote (see load_stats), possibly empty
        - trades: the trades the run executed (see load_trades), possibly empty
        - balance: shorthand for positions['balance']
        - start, end: first and last day of the trading period
        - entry_signal_name, exit_signal_name: see signal_names
    """
    def __init__(self, folder, ticker, strategy_name, positions, stats, trades):
        self.folder = folder
        self.ticker = ticker
        self.strategy_name = strategy_name
        self.positions = positions
        self.stats = stats
        self.trades = trades
        self.balance = positions['balance']
        self.start = positions.index[0]
        self.end = positions.index[-1]
        self.entry_signal_name, self.exit_signal_name = signal_names(positions)


def load_run(folder):
    """
        Reads a whole backtest run back from its output folder.
    :param folder: the run's output folder
    :return: Run
    :raise ValueError: when the folder is not a run folder or its positions file
                       cannot be used
    """
    files = find_run_files(folder)
    positions = load_positions(files['positions'])
    stats = load_stats(files['stats'])
    trades = load_trades(files['trades'])
    return Run(files['folder'], files['ticker'], files['strategy_name'], positions, stats, trades)


# ----------------------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------------------

def drawdown_episodes(balance, top=3):
    """
        Finds the deepest drawdowns of a mark to market balance curve. An episode is
        one contiguous stretch below a high water mark, from the day the balance first
        falls below it to the day it is recovered (or to the last day, for an episode
        still under water). Each episode contributes only its single deepest point, so
        one long slide is never reported more than once.
    :param balance: daily balance Series indexed by date
    :param top: how many episodes to report
    :return: list of at most 'top' dicts {'value', 'date', 'start', 'end'}, deepest
             first. 'value' is negative, e.g. -0.35. Empty for a curve that only rises.
    """
    peak = balance.cummax()
    drawdown = (balance - peak) / peak
    under = drawdown < 0
    if not under.any():
        return []

    episode = (~under).cumsum()[under]
    dates = pd.Series(balance.index, index=balance.index)[under]
    grouped = drawdown[under].groupby(episode)
    deepest = grouped.min()
    trough = grouped.idxmin()
    start = dates.groupby(episode).min()
    end = dates.groupby(episode).max()

    episodes = [{'value': deepest[e], 'date': trough[e], 'start': start[e], 'end': end[e]}
                for e in deepest.index]
    episodes.sort(key=lambda e: e['value'])
    return episodes[:top]


def key_statistics(run):
    """
        The numbers the dashboard header and stats strip report.
    :param run: the Run
    :return: dict with 'start', 'end', 'total_return', 'cagr', 'drawdowns' (see
             drawdown_episodes), plus 'no_trades', 'sharpe', 'calmar',
             'max_time_in_dd', 'max_dd_pcnt_realized' read from the stats file - None
             where that file predates the metric
    """
    stats = run.stats
    rtn = stat_value(stats, 'rtn')
    total_return = (rtn - 1 if rtn is not None
                    else run.balance.iloc[-1] / run.balance.iloc[0] - 1)
    cagr = stat_value(stats, 'cagr')
    if cagr is None:
        years = (run.end - run.start).days / 365
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else float('nan')

    return {
        'start': run.start, 'end': run.end,
        'total_return': total_return, 'cagr': cagr,
        'drawdowns': drawdown_episodes(run.balance, top=3),
        'no_trades': stat_value(stats, 'no_trades'),
        'sharpe': stat_value(stats, 'sharpe'),
        'calmar': stat_value(stats, 'calmar'),
        'max_time_in_dd': stat_value(stats, 'max_time_in_dd'),
        'max_dd_pcnt_realized': stat_value(stats, 'max_dd_pcnt_realized'),
    }


def monthly_returns(balance):
    """
        Compounds the daily balance into calendar month and calendar year returns.
    :param balance: daily balance Series indexed by date
    :return: (monthly, annual) Series indexed by period end. The first month/year is
             measured from the first trading day of the run, not from a full period.
    """
    growth = 1 + balance.pct_change()
    monthly = growth.resample('ME').prod() - 1
    annual = growth.resample('YE').prod() - 1
    return monthly, annual


def returns_pivot(balance):
    """
        Lays the monthly returns out as the data pane table: one row per month plus an
        'Annual' row along the bottom, one column per year.
    :param balance: daily balance Series indexed by date
    :return: DataFrame with a 'Month' column of row labels and one column per year,
             the year column names being strings so they can be DataTable column ids.
             Cells outside the trading period are NaN.
    """
    monthly, annual = monthly_returns(balance)
    pivot = pd.DataFrame({'Year': monthly.index.year, 'Month': monthly.index.month,
                          'Return': monthly.to_numpy()}).pivot(
        index='Month', columns='Year', values='Return')

    by_year = pd.Series(annual.to_numpy(), index=annual.index.year).reindex(pivot.columns)
    annual_row = pd.DataFrame([by_year.to_numpy()], index=[_ANNUAL_ROW], columns=pivot.columns)
    pivot = pd.concat([pivot, annual_row])

    pivot.index = pivot.index.map(lambda i: _MONTH_LABELS.get(i, i))
    pivot = pivot.reindex(_ROW_ORDER)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot.reset_index().rename(columns={'index': _MONTH_COLUMN})


def table_records(pivot):
    """
        Renders the returns pivot as DataTable rows, NaN cells becoming None so that
        months outside the trading period serialize as empty JSON cells.
    :param pivot: the DataFrame from returns_pivot
    :return: list of dicts, one per row
    """
    return pivot.astype(object).where(pd.notna(pivot), None).to_dict('records')


def signal_blocks(signal, min_days=_MIN_BAND_DAYS):
    """
        Collapses the days a signal is on into date ranges. 'es' and 'xs' hold the
        close price on the days the signal is on and 0.0 otherwise, so a day counts as
        on when the value is strictly positive.
    :param signal: the 'es' or 'xs' column of the positions frame
    :param min_days: a block shorter than this is widened symmetrically so that a
                     single day signal stays visible on a multi year x axis. This is a
                     deliberate visual distortion of a few days, not a data change.
    :return: list of (start, end) Timestamp pairs, in date order
    """
    on = signal > 0
    if not on.any():
        return []

    block = (~on).cumsum()[on]
    dates = pd.Series(signal.index, index=signal.index)[on]
    starts = dates.groupby(block).min()
    ends = dates.groupby(block).max()

    one_day = pd.Timedelta(days=1)
    blocks = []
    for start, end in zip(starts, ends):
        width_days = (end - start) / one_day + 1
        if width_days < min_days:
            pad = one_day * math.ceil((min_days - width_days) / 2)
            start, end = start - pad, end + pad
        blocks.append((start, end))
    return blocks


# ----------------------------------------------------------------------------------
# The chart pane
# ----------------------------------------------------------------------------------

def _band_trace(blocks, fill_color, name):
    xs, ys = [], []
    for start, end in blocks:
        xs += [start, start, end, end, None]
        ys += [0, 1, 1, 0, None]
    return go.Scatter(x=xs, y=ys, yaxis='y3', fill='toself', mode='none',
                      fillcolor=fill_color, hoverinfo='skip', name=name)


def _price_hover_trace(run):
    """
        An invisible marker on every daily close, on the price axis, that carries the
        chart's hover label: the day's OHLC and the portfolio value. The OHLC and
        balance traces themselves skip hover, so with hovermode 'closest' the label
        appears only when the cursor is within _HOVER_DISTANCE_PX of the underlying.
    """
    positions = run.positions
    return go.Scatter(
        x=positions.index, y=positions['C'], yaxis='y2', mode='markers',
        marker=dict(size=6, opacity=0), name=_PRICE_HOVER_NAME, showlegend=False,
        customdata=pd.concat([positions[['O', 'H', 'L', 'C']], run.balance], axis=1),
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                       "O: %{customdata[0]:,.2f}  H: %{customdata[1]:,.2f}<br>"
                       "L: %{customdata[2]:,.2f}  C: %{customdata[3]:,.2f}<br>"
                       "Portfolio: $%{customdata[4]:,.0f}"
                       f"<extra>{run.ticker}</extra>"))


def figure_title(run, stats):
    """
        The two line chart title that doubles as the dashboard header: strategy,
        underlying and backtest period on the first line; total return, CAGR and the
        three largest drawdowns on the second.
    :param run: the Run
    :param stats: the dict from key_statistics
    :return: str, with a <br><sup> break between the lines
    """
    line1 = (f"{run.strategy_name} on {run.ticker}   "
             f"{stats['start']:%Y-%m-%d} to {stats['end']:%Y-%m-%d}")
    line2 = f"Total return: {stats['total_return']:.2%}, CAGR: {stats['cagr']:.2%}"
    if stats['drawdowns']:
        dd = ', '.join(f"{d['value']:.1%} ({d['date']:%b %Y})" for d in stats['drawdowns'])
        line2 += f", Top drawdowns: {dd}"
    return f"{line1}<br><sup>{line2}</sup>"


def build_figure(run, stats=None):
    """
        Builds the chart pane: the strategy balance on a log left axis, the underlying
        OHLC on a log right axis, the entry and exit signal regimes as translucent
        full height bands on an invisible third axis, and the executed trades as
        entry/exit markers on the price axis.
    :param run: the Run
    :param stats: the dict from key_statistics; computed when None
    :return: go.Figure
    """
    if stats is None:
        stats = key_statistics(run)

    positions = run.positions
    fig = go.Figure()

    entry_blocks = signal_blocks(positions['es'])
    if entry_blocks:
        fig.add_trace(_band_trace(entry_blocks, _ENTRY_FILL,
                                  f"{run.entry_signal_name or 'entry signal'} on"))
    exit_blocks = signal_blocks(positions['xs'])
    if exit_blocks:
        fig.add_trace(_band_trace(exit_blocks, _EXIT_FILL,
                                  f"{run.exit_signal_name or 'exit signal'} on"))

    fig.add_trace(go.Ohlc(x=positions.index, open=positions['O'], high=positions['H'],
                          low=positions['L'], close=positions['C'],
                          name=f"{run.ticker} price", opacity=0.5, yaxis='y2',
                          hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=positions.index, y=run.balance,
                             name=f"{run.strategy_name} portfolio value",
                             line=dict(color='green', width=2), hoverinfo='skip'))

    fig.add_trace(_price_hover_trace(run))

    if not run.trades.empty:
        fig.add_trace(go.Scatter(
            x=run.trades['entry_date'], y=run.trades['entry_price'], yaxis='y2',
            mode='markers', name='entry',
            marker=dict(symbol='triangle-up', size=9, color=_ENTRY_MARKER,
                       line=dict(width=1, color='white')),
            hovertemplate="%{x|%Y-%m-%d}<br>price: %{y:,.2f}<extra>entry</extra>"))
        fig.add_trace(go.Scatter(
            x=run.trades['exit_date'], y=run.trades['exit_price'], yaxis='y2',
            mode='markers', name='exit',
            marker=dict(symbol='triangle-down', size=9, color=_EXIT_MARKER,
                       line=dict(width=1, color='white')),
            customdata=run.trades[['pnl_pcnt', 'hp']],
            hovertemplate="pnl: %{customdata[0]:.2%}<br>held: %{customdata[1]:.0f}d"
                         "<extra></extra>"))

    fig.update_layout(
        title=dict(text=figure_title(run, stats), font=dict(color='black')),
        yaxis=dict(title='Portfolio value ($)', type='log', side='left'),
        yaxis2=dict(title=f'{run.ticker} price', type='log', overlaying='y', side='right'),
        yaxis3=dict(overlaying='y', range=[0, 1], visible=False, fixedrange=True),
        xaxis=dict(dtick='M12', tickformat='%Y', tick0='2000-01-01',
                  minor=dict(dtick='M1', ticks='outside', ticklen=3, showgrid=False),
                  rangeslider=dict(visible=False),
                  showspikes=True, spikemode='across', spikethickness=1),
        template='plotly_white',
        # hover only near the underlying (see _price_hover_trace), not at any height
        hovermode='closest',
        hoverdistance=_HOVER_DISTANCE_PX,
        hoverlabel=dict(bgcolor='white', font_size=12),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=60, t=60, b=40),
        # keeps the user's zoom/pan across the click-to-highlight callback below
        uirevision='constant')

    return fig


# ----------------------------------------------------------------------------------
# Click on the returns table -> highlight in the chart pane
# ----------------------------------------------------------------------------------

def highlight_shapes(start, end):
    """
        The two dotted vertical markers that delimit a clicked table period.
    :param start: first day of the period
    :param end: last day of the period
    :return: list of two plotly shape dicts, referenced to the x axis and to paper y
             so that they span the full chart height regardless of the log axes
    """
    def vline(x):
        return {'type': 'line', 'xref': 'x', 'yref': 'paper',
                'x0': x, 'y0': 0, 'x1': x, 'y1': 1,
                'line': {'color': _HIGHLIGHT_COLOR, 'width': 2, 'dash': 'dot'}}

    return [vline(start.strftime('%Y-%m-%d')), vline(end.strftime('%Y-%m-%d'))]


def period_bounds(row_label, column_id):
    """
        The calendar period a clicked returns table cell stands for.
    :param row_label: the cell's 'Month' value - a month name, or 'Annual'
    :param column_id: the cell's column id - a year, as a string
    :return: (first_day, last_day) Timestamps, or None when the cell names no period
             (the Month label column, an unparseable year)
    """
    try:
        year = int(column_id)
    except (TypeError, ValueError):
        return None
    if row_label == _ANNUAL_ROW:
        return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)
    month = _MONTH_NUMBERS.get(row_label)
    if month is None:
        return None
    start = pd.Timestamp(year=year, month=month, day=1)
    return start, start + pd.offsets.MonthEnd(1)


def returns_style(year_columns):
    """
        The base conditional formatting of the returns table: gains above 5% green,
        losses below -5% red.
    :param year_columns: the table's year column ids
    :return: list of style_data_conditional rules
    """
    style = []
    for column in year_columns:
        style.append({'if': {'filter_query': f'{{{column}}} > {_GAIN_THRESHOLD}',
                             'column_id': str(column)},
                      'backgroundColor': 'lightgreen', 'color': 'black'})
        style.append({'if': {'filter_query': f'{{{column}}} < {_LOSS_THRESHOLD}',
                             'column_id': str(column)},
                      'backgroundColor': 'lightcoral', 'color': 'black'})
    return style


def highlight_for_cell(active_cell, rows, year_columns):
    """
        Decides what a click on the returns table does. Kept out of the Dash callback
        so that the click behaviour is testable without a browser.
    :param active_cell: the DataTable active_cell dict, or None
    :param rows: the table's data, as returned by table_records
    :param year_columns: the table's year column ids
    :return: (shapes, styles) - the chart shapes to install, empty to clear the
             highlight, and the table's style_data_conditional with the clicked cell
             marked
    """
    styles = returns_style(year_columns)
    if not active_cell:
        return [], styles

    row_index, column_id = active_cell['row'], active_cell['column_id']
    if column_id == _MONTH_COLUMN:
        return [], styles

    styles = styles + [{'if': {'row_index': row_index, 'column_id': column_id},
                        'backgroundColor': 'yellow', 'color': 'black'}]
    bounds = period_bounds(rows[row_index][_MONTH_COLUMN], column_id)
    return (highlight_shapes(*bounds) if bounds else []), styles


# ----------------------------------------------------------------------------------
# Dash application
# ----------------------------------------------------------------------------------

def _stats_strip(stats):
    def fmt(value, spec):
        return "NA" if value is None else format(value, spec)

    return (f"Trades: {fmt(stats['no_trades'], '.0f')}  |  "
           f"Sharpe: {fmt(stats['sharpe'], '.2f')}  |  "
           f"Calmar: {fmt(stats['calmar'], '.2f')}  |  "
           f"Max time in drawdown: {fmt(stats['max_time_in_dd'], '.0f')} days  |  "
           f"Max realized drawdown: {fmt(stats['max_dd_pcnt_realized'], '.1%')}")


def build_layout(run, figure, pivot, stats):
    """
        The dashboard page: the chart pane above, a one line strip of the supplementary
        run metrics, and the monthly returns table below.
    :param run: the Run
    :param figure: the go.Figure from build_figure
    :param pivot: the DataFrame from returns_pivot
    :param stats: the dict from key_statistics
    :return: html.Div
    """
    year_columns = [c for c in pivot.columns if c != _MONTH_COLUMN]
    columns = [{'name': _MONTH_COLUMN, 'id': _MONTH_COLUMN, 'type': 'text'}]
    columns += [{'name': c, 'id': c, 'type': 'numeric', 'format': {'specifier': '.2%'}}
               for c in year_columns]

    return html.Div([
        dcc.Graph(id='main-chart', figure=figure, style={'height': '55vh'},
                 config={'scrollZoom': True, 'displaylogo': False,
                         'modeBarButtonsToRemove': ['lasso2d', 'select2d']}),
        html.Div(_stats_strip(stats), id='run-stats',
                style={'padding': '6px 12px', 'color': '#444', 'fontSize': '13px'}),
        dash_table.DataTable(
            id='returns-table',
            columns=columns,
            data=table_records(pivot),
            style_cell={'textAlign': 'left', 'backgroundColor': 'white', 'color': 'black',
                       'padding': '2px 6px', 'fontFamily': 'monospace', 'minWidth': '70px'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold',
                         'color': 'white'},
            style_table={'overflowX': 'auto', 'width': '100%', 'minWidth': '100%'},
            fixed_columns={'headers': True, 'data': 1},
            style_data_conditional=returns_style(year_columns)),
    ], style={'fontFamily': 'sans-serif', 'margin': '10px'})


def create_app(run):
    """
        Builds the Dash application over one run and wires the returns table click to
        the chart highlight. Does not start a server.
    :param run: the Run
    :return: dash.Dash
    """
    stats = key_statistics(run)
    figure = build_figure(run, stats)
    pivot = returns_pivot(run.balance)
    year_columns = [c for c in pivot.columns if c != _MONTH_COLUMN]

    app = dash.Dash(__name__)
    app.layout = build_layout(run, figure, pivot, stats)

    @app.callback(
        Output('main-chart', 'figure'),
        Output('returns-table', 'style_data_conditional'),
        Input('returns-table', 'active_cell'),
        State('returns-table', 'data'))
    def _on_cell_click(active_cell, rows):
        shapes, styles = highlight_for_cell(active_cell, rows, year_columns)
        patch = Patch()
        patch['layout']['shapes'] = shapes
        return patch, styles

    return app


def serve(app, host=_DEFAULT_HOST, port=_DEFAULT_PORT, debug=False):
    """
        Starts the Dash development server. Blocks until interrupted.
    """
    app.run(host=host, port=port, debug=debug)


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------

def _port(value):
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a port number.")
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"Port {port} is outside 1..65535.")
    return port


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="visualize.py",
        description="Serve an interactive dashboard over a saved backtest run.")
    p.add_argument("-out", "--out", dest="out", required=True,
                  help="output folder of the run to visualize, "
                       'e.g. "out/test/Index_50dMAvs200dMA"')
    p.add_argument("--host", default=_DEFAULT_HOST,
                  help=f"interface to serve on (default: {_DEFAULT_HOST})")
    p.add_argument("--port", default=_DEFAULT_PORT, type=_port,
                  help=f"port to serve on (default: {_DEFAULT_PORT})")
    p.add_argument("--debug", action="store_true",
                  help="run the Dash development server with the reloader and the "
                       "in browser error pane")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        run = load_run(args.out)
        app = create_app(run)
    except (OSError, ValueError, KeyError) as e:
        print(f"visualize failed: {e}", file=sys.stderr)
        return 2

    print(f"Strategy   : {run.strategy_name}")
    print(f"Underlying : {run.ticker}")
    print(f"Period     : {run.start:%Y-%m-%d} .. {run.end:%Y-%m-%d}")
    print(f"Run folder : {args.out}")
    print(f"Serving    : http://{args.host}:{args.port}/")
    serve(app, host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
